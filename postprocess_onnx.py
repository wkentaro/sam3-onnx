#!/usr/bin/env python3

import argparse
import pathlib

import numpy as np
import onnx
from loguru import logger
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper
from onnxruntime.quantization import QuantType
from onnxruntime.quantization import quantize_dynamic


def postprocess(onnx_file: pathlib.Path) -> None:
    """Rewrite ops that onnxruntime's CoreML EP rejects or loads slowly, or that its
    WebGPU EP runs slowly, and inline the weights.

    Every rewrite is exact, so CPU and CUDA outputs stay the same. CoreML runs a
    graph it fully supports as one partition instead of tens, and its EP cannot
    read external data (it resolves the .data file against the .onnx file path).
    The external data file stays, as other models may share it.
    """
    model: onnx.ModelProto = onnx.load(onnx_file, load_external_data=False)
    # Before the weights load, as shape inference copies the model.
    shapes: dict[str, list[int]] = _static_shapes(model)
    onnx.load_external_data_for_model(model, str(onnx_file.parent))
    graph: onnx.GraphProto = model.graph

    # Only the indices and shapes, to avoid copying every weight.
    needed: set[str] = {
        n.input[1] for n in graph.node if n.op_type in ("Gather", "Reshape")
    }
    constants: dict[str, np.ndarray] = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.initializer
        if tensor.name in needed
    }
    for node in graph.node:
        if (
            node.op_type == "Constant"
            and node.output[0] in needed
            and node.attribute[0].name == "value"
        ):
            constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    initializers: dict[str, onnx.TensorProto] = {t.name: t for t in graph.initializer}
    outputs: set[str] = {o.name for o in graph.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def add_constant(name: str, value: np.ndarray) -> str:
        graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    changed: set[str] = set()
    fused: set[str] = set()
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.output[0] in fused:
            continue

        # MatMul(x, W) + Add(b) -> Reshape, Gemm, Reshape, as onnxruntime fuses
        # them for static shapes, so CoreML loaders can rewrite every linear
        # layer to transB=1 (see below).
        weight: onnx.TensorProto | None = (
            initializers.get(node.input[1]) if node.op_type == "MatMul" else None
        )
        (add, *others) = consumers.get(node.output[0], [None])
        # Only float, as onnxruntime has no integer Gemm.
        if (
            weight is not None
            and weight.data_type == onnx.TensorProto.FLOAT
            and add is not None
            and add.op_type == "Add"
            and not others
            and node.output[0] not in outputs
        ):
            x_shape: list[int] = shapes.get(node.input[0], [])
            bias: str = next((i for i in add.input if i != node.output[0]), "")
            if (
                len(weight.dims) != 2
                or len(x_shape) < 2
                or bias not in initializers
                or list(initializers[bias].dims) != [weight.dims[1]]
            ):
                nodes.append(node)
                continue
            (k, n) = weight.dims
            output = add.output[0]
            (x_2d, y_2d) = (f"{output}_2d", f"{output}_gemm")
            shape_2d = add_constant(f"{x_2d}_shape", np.array([-1, k]))
            shape_nd = add_constant(f"{output}_shape", np.array([*x_shape[:-1], n]))
            nodes += [
                helper.make_node(
                    "Reshape", [node.input[0], shape_2d], [x_2d], name=x_2d
                ),
                helper.make_node(
                    "Gemm", [x_2d, node.input[1], bias], [y_2d], name=y_2d
                ),
                helper.make_node("Reshape", [y_2d, shape_nd], [output], name=output),
            ]
            fused.add(output)
            changed.add(node.output[0])
            continue
        nodes.append(node)

        # Gemm(transB=1) -> Gemm(transB=0) with W {K, N}, as onnxruntime's WebGPU
        # EP runs transB=1 at half the speed. The CoreML EP loads transB=0 in
        # minutes, as it writes the transposed W into the model as text, so osam
        # rewrites it back to transB=1 for CoreML.
        trans_b = next((a for a in node.attribute if a.name == "transB"), None)
        weight = initializers.get(node.input[1]) if node.op_type == "Gemm" else None
        if trans_b is not None and trans_b.i == 1 and weight is not None:
            node.input[1] = add_constant(
                f"{node.output[0]}_weight", numpy_helper.to_array(weight).T
            )
            trans_b.i = 0
            continue

        indices: np.ndarray | None = (
            constants.get(node.input[1]) if node.op_type == "Gather" else None
        )
        if indices is not None and indices.ndim == 0:
            # Gather(scalar) == Squeeze(Gather([index])); CoreML rejects scalar indices.
            axis: int = next((a.i for a in node.attribute if a.name == "axis"), 0)
            output: str = node.output[0]
            node.input[1] = add_constant(f"{output}_indices", indices.reshape(1))
            node.output[0] = f"{output}_unsqueezed"
            axes: str = add_constant(f"{output}_axes", np.array([axis]))
            nodes.append(
                helper.make_node(
                    "Squeeze",
                    [node.output[0], axes],
                    [output],
                    name=f"{output}_squeeze",
                )
            )
            continue

        # Reshape to (1, ...) of rank 6, then Transpose that keeps axis 0, then
        # Reshape: drop the leading 1 so both are rank 5, which CoreML caps at.
        shape: np.ndarray | None = (
            constants.get(node.input[1]) if node.op_type == "Reshape" else None
        )
        # A 0 in a shape copies the input dimension at its index, which shifts.
        if shape is None or len(shape) != 6 or shape[0] != 1 or 0 in shape:
            continue
        (transpose, *others) = consumers.get(node.output[0], [None])
        if (
            others
            or transpose is None
            or transpose.op_type != "Transpose"
            or {node.output[0], transpose.output[0]} & outputs
        ):
            continue
        # Without perm, Transpose reverses the axes and so moves axis 0.
        perm = next((a for a in transpose.attribute if a.name == "perm"), None)
        if (
            perm is None
            or perm.ints[0] != 0
            or any(
                c.op_type != "Reshape"
                or c.input[1] not in constants
                or 0 in constants[c.input[1]]
                for c in consumers.get(transpose.output[0], [])
            )
        ):
            continue
        node.input[1] = add_constant(f"{node.output[0]}_shape", shape[1:])
        perm.ints[:] = [p - 1 for p in perm.ints[1:]]
        changed.update([node.output[0], transpose.output[0]])

    del graph.node[:]
    graph.node.extend(nodes)
    # Their recorded shapes no longer hold; onnxruntime infers them on load.
    value_info = [v for v in graph.value_info if v.name not in changed]
    del graph.value_info[:]
    graph.value_info.extend(value_info)
    # Drop the scalar indices, rank-6 shapes and {N, K} weights nothing reads.
    used: set[str] = {name for n in _all_nodes(graph) for name in n.input}
    used.update(o.name for o in graph.output)
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]

    # Raises for models over protobuf's 2 GB, which would need external data.
    onnx.save(model, onnx_file)
    logger.debug("postprocessed onnx model: {!r}", str(onnx_file))


_INTEGERS: frozenset[int] = frozenset(
    {
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    }
)


def convert_float16(onnx_file: pathlib.Path, output_file: pathlib.Path) -> None:
    """Write a copy of the model that computes in float16, for GPUs.

    Its inputs and outputs stay float32, so callers run it like the original.
    It halves the file and the memory it takes, and runs up to twice as fast
    on a GPU. CPUs lack float16 kernels, so keep the original for them.
    """
    model: onnx.ModelProto = onnx.load(onnx_file)
    graph: onnx.GraphProto = model.graph
    if any(a.g.node or a.graphs for n in graph.node for a in n.attribute):
        raise ValueError(f"{onnx_file} has a subgraph, which is not converted")

    # ONNX defines the scales of Resize for float32 alone.
    float32_only: set[str] = {
        n.input[2]
        for n in graph.node
        if n.op_type == "Resize" and len(n.input) > 2 and n.input[2]
    }

    def convert(tensor: onnx.TensorProto) -> None:
        if tensor.data_type != TensorProto.FLOAT:
            return
        value = numpy_helper.to_array(tensor)
        # Keep a large finite value, like an attention mask's, from becoming inf.
        limit = np.finfo(np.float16).max
        clipped = np.where(np.isfinite(value), np.clip(value, -limit, limit), value)
        tensor.CopyFrom(
            numpy_helper.from_array(clipped.astype(np.float16), tensor.name)
        )

    types: dict[str, int] = {
        value.name: value.type.tensor_type.elem_type
        for value in [
            *graph.input,
            *onnx.shape_inference.infer_shapes(model).graph.value_info,
            *graph.output,
        ]
    }
    types.update({tensor.name: tensor.data_type for tensor in graph.initializer})

    for tensor in graph.initializer:
        if tensor.name not in float32_only:
            convert(tensor)
    # A count of pixels can pass 65504, the largest float16. So an integer cast
    # to float stays float32 through the nodes that take such values alone,
    # like the division of two counts, and becomes float16 where it meets
    # another float.
    float32_values: set[str] = set()
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        to = next((a for a in node.attribute if a.name == "to"), None)
        float_inputs = [i for i in node.input if types.get(i) == TensorProto.FLOAT]
        if (
            node.op_type == "Cast"
            and to is not None
            and to.i == TensorProto.FLOAT
            and types.get(node.input[0]) in _INTEGERS
        ):
            float32_values.add(node.output[0])
        elif float_inputs and all(i in float32_values for i in float_inputs):
            float32_values.update(
                o for o in node.output if types.get(o) == TensorProto.FLOAT
            )
        else:
            for index, name in enumerate(node.input):
                if name not in float32_values:
                    continue
                half = f"{name}_half"
                if half not in types:
                    types[half] = TensorProto.FLOAT16
                    nodes.append(
                        helper.make_node(
                            "Cast", [name], [half], name=half, to=TensorProto.FLOAT16
                        )
                    )
                node.input[index] = half
            if node.op_type == "Cast" and to is not None:
                if to.i == TensorProto.FLOAT:
                    to.i = TensorProto.FLOAT16
            elif node.op_type in ("Constant", "ConstantOfShape"):
                if node.op_type == "Constant" and node.output[0] in float32_only:
                    pass
                elif any(
                    a.name in ("value_float", "value_floats") for a in node.attribute
                ):
                    raise ValueError(f"{node.name} holds a float outside a tensor")
                else:
                    for attribute in node.attribute:
                        if attribute.name == "value":
                            convert(attribute.t)
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)

    def rename(old: str, new: str) -> None:
        for node in graph.node:
            node.input[:] = [new if name == old else name for name in node.input]
            node.output[:] = [new if name == old else name for name in node.output]

    casts_in: list[onnx.NodeProto] = []
    for value in graph.input:
        if value.type.tensor_type.elem_type == TensorProto.FLOAT:
            name = f"{value.name}_float16"
            rename(value.name, name)
            casts_in.append(
                helper.make_node(
                    "Cast", [value.name], [name], name=name, to=TensorProto.FLOAT16
                )
            )
    casts_out: list[onnx.NodeProto] = []
    for value in graph.output:
        if value.type.tensor_type.elem_type == TensorProto.FLOAT:
            name = f"{value.name}_float16"
            rename(value.name, name)
            casts_out.append(
                helper.make_node(
                    "Cast", [name], [value.name], name=name, to=TensorProto.FLOAT
                )
            )
    nodes = [*casts_in, *graph.node, *casts_out]
    del graph.node[:]
    graph.node.extend(nodes)
    # The recorded types no longer hold; onnxruntime infers them on load.
    del graph.value_info[:]

    # Raises for an operator that ONNX does not define for float16.
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.save(model, output_file)
    logger.debug("converted to float16: {!r}", str(output_file))


def convert_int8(onnx_file: pathlib.Path, output_file: pathlib.Path) -> None:
    """Write a copy of the model whose linear layers compute in int8, for CPUs.

    Their weights become int8, and onnxruntime quantizes their inputs as it
    runs (dynamic quantization). Other ops stay float32, and so do the inputs
    and outputs.
    """
    model: onnx.ModelProto = onnx.load(onnx_file)
    graph: onnx.GraphProto = model.graph
    initializers: dict[str, onnx.TensorProto] = {t.name: t for t in graph.initializer}
    # Gemm -> MatMul + Add, as dynamic quantization takes MatMul and not Gemm,
    # which the linear layers are after postprocess.
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weight = initializers.get(node.input[1]) if node.op_type == "Gemm" else None
        if (
            weight is None
            or attributes.get("transA", 0)
            or attributes.get("alpha", 1.0) != 1.0
            or attributes.get("beta", 1.0) != 1.0
        ):
            nodes.append(node)
            continue
        (x, weight_name, *rest) = node.input
        if attributes.get("transB", 0):
            weight_name = f"{weight.name}_transposed"
            if weight_name not in initializers:
                transposed = numpy_helper.to_array(weight).T
                initializers[weight_name] = numpy_helper.from_array(
                    transposed, weight_name
                )
                graph.initializer.append(initializers[weight_name])
        (output,) = node.output
        bias: str = rest[0] if rest else ""
        matmul = f"{output}_matmul" if bias else output
        nodes.append(
            helper.make_node("MatMul", [x, weight_name], [matmul], name=matmul)
        )
        if bias:
            nodes.append(helper.make_node("Add", [matmul, bias], [output], name=output))
    del graph.node[:]
    graph.node.extend(nodes)
    # Drop the {N, K} weights that only the Gemms read.
    used: set[str] = {name for n in _all_nodes(graph) for name in n.input}
    used.update(o.name for o in graph.output)
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]
    # Only the MatMuls with a weight: attention's MatMuls of two activations
    # stay float32.
    quantize_dynamic(
        model, output_file, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8
    )
    logger.debug("converted to int8: {!r}", str(output_file))


def fuse_attention(onnx_file: pathlib.Path) -> None:
    """Replace each attention of the image encoder with onnxruntime's
    MultiHeadAttention, which runs it in one kernel on CUDA and WebGPU.

    It matches what torch exports for the windowed and global attention of
    sam3.model.vitdet.Attention, with q, k, and v in (batch, heads, tokens,
    channels) and k^T as Reshape(Transpose(Reshape(k))):

        Reshape(Transpose(Reshape(MatMul(Softmax(MatMul(q * s, k^T * s)), v))))

    and leaves any other attention as it is. CoreML lacks the op.
    """
    model: onnx.ModelProto = onnx.load(onnx_file)
    graph: onnx.GraphProto = model.graph
    shapes: dict[str, list[int]] = _static_shapes(model)
    producer = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def scale(mul: onnx.NodeProto) -> float:
        (tensor,) = [t for t in graph.initializer if t.name == mul.input[1]]
        return float(numpy_helper.to_array(tensor).item())

    def chain(name: str, op_types: list[str]) -> list[onnx.NodeProto] | None:
        """The nodes that produce name, back through op_types."""
        nodes = []
        for op_type in op_types:
            node = producer.get(name)
            if node is None or node.op_type != op_type:
                return None
            nodes.append(node)
            name = node.input[0]
        return nodes

    def chain_down(name: str, op_types: list[str]) -> list[onnx.NodeProto] | None:
        """The nodes that alone read name, on through op_types."""
        nodes = []
        for op_type in op_types:
            (node, *others) = consumers.get(name, [None])
            if node is None or others or node.op_type != op_type:
                return None
            nodes.append(node)
            name = node.output[0]
        return nodes

    replacements: dict[int, list[onnx.NodeProto]] = {}
    for i, softmax in enumerate(n for n in graph.node if n.op_type == "Softmax"):
        (qk,) = chain(softmax.input[0], ["MatMul"]) or [None]
        if qk is None:
            continue
        q_mul = chain(qk.input[0], ["Mul"])
        k_path = chain(qk.input[1], ["Mul", "Reshape", "Transpose", "Reshape"])
        out_path = chain_down(
            softmax.output[0], ["MatMul", "Reshape", "Transpose", "Reshape"]
        )
        if q_mul is None or k_path is None or out_path is None:
            continue
        (av, heads_out, _, reshape) = out_path
        (batch, heads, tokens, channels) = shapes[q_mul[0].input[0]]
        p = f"attention{i}_"
        # q and k in (batch, tokens, heads * channels), v in (batch, heads,
        # tokens, channels), which the op takes as it is.
        q_shape = numpy_helper.from_array(
            np.array([batch, tokens, heads * channels], np.int64), p + "q_shape"
        )
        k_shape = numpy_helper.from_array(
            np.array([batch, heads, tokens, channels], np.int64), p + "k_shape"
        )
        graph.initializer.extend([q_shape, k_shape])
        replacements[id(heads_out)] = [
            helper.make_node(
                "Transpose", [q_mul[0].input[0]], [p + "q_t"], perm=[0, 2, 1, 3]
            ),
            helper.make_node("Reshape", [p + "q_t", q_shape.name], [p + "q"]),
            helper.make_node("Reshape", [k_path[3].input[0], k_shape.name], [p + "k"]),
            helper.make_node(
                "MultiHeadAttention",
                [p + "q", p + "k", av.input[1]],
                [p + "out"],
                domain="com.microsoft",
                num_heads=heads,
                scale=scale(q_mul[0]) * scale(k_path[0]),
            ),
        ]
        # The op gives (batch, tokens, heads * channels), as the Transpose and
        # Reshape after it did.
        reshape.input[0] = p + "out"
    nodes = []
    for node in graph.node:
        nodes.extend(replacements.get(id(node), []))
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(_live_nodes(nodes, {o.name for o in graph.output}))
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(model, onnx_file)
    logger.debug("fused {} attentions: {!r}", len(replacements), str(onnx_file))


def _live_nodes(nodes: list[onnx.NodeProto], outputs: set[str]) -> list[onnx.NodeProto]:
    needed: set[str] = set(outputs)
    live: list[onnx.NodeProto] = []
    for node in reversed(nodes):
        if any(o in needed for o in node.output):
            live.append(node)
            needed.update(node.input)
    return live[::-1]


def _static_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    graph: onnx.GraphProto = onnx.shape_inference.infer_shapes(
        model, data_prop=True
    ).graph
    shapes: dict[str, list[int]] = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
        dims = value.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") for d in dims):
            shapes[value.name] = [d.dim_value for d in dims]
    return shapes


def _all_nodes(graph: onnx.GraphProto):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            for subgraph in [attribute.g, *attribute.graphs]:
                yield from _all_nodes(subgraph)


def main() -> None:
    parser = argparse.ArgumentParser(description=postprocess.__doc__)
    parser.add_argument("onnx_files", type=pathlib.Path, nargs="+")
    copy = parser.add_mutually_exclusive_group()
    copy.add_argument(
        "--fp16",
        action="store_true",
        help="Instead, write a float16 copy for GPUs next to each file, "
        "as <name>_fp16.onnx.",
    )
    copy.add_argument(
        "--int8",
        action="store_true",
        help="Instead, write an int8 copy for CPUs next to each file, "
        "as <name>_int8.onnx.",
    )
    args = parser.parse_args()
    for onnx_file in args.onnx_files:
        if not (args.fp16 or args.int8):
            postprocess(onnx_file)
        elif not onnx_file.stem.endswith(("_fp16", "_int8")):
            (suffix, convert) = (
                ("_fp16", convert_float16) if args.fp16 else ("_int8", convert_int8)
            )
            convert(onnx_file, onnx_file.with_name(f"{onnx_file.stem}{suffix}.onnx"))


if __name__ == "__main__":
    main()
