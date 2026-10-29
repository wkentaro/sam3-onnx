#!/usr/bin/env python3

import argparse
import pathlib

import numpy as np
import onnx
from loguru import logger
from onnx import helper
from onnx import numpy_helper


def postprocess(onnx_file: pathlib.Path) -> None:
    """Rewrite ops that onnxruntime's CoreML EP rejects or loads slowly, and inline
    the weights.

    Every rewrite is exact, so CPU and CUDA outputs stay the same. CoreML runs a
    graph it fully supports as one partition instead of tens, and its EP cannot
    read external data (it resolves the .data file against the .onnx file path).
    """
    model: onnx.ModelProto = onnx.load(onnx_file, load_external_data=False)
    data_files: set[str] = {
        entry.value
        for tensor in model.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
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
        # them for static shapes, but with W already {N, K} (transB=1). Otherwise
        # the CoreML EP writes the transposed W into the model as text, which
        # takes Core ML minutes to load.
        weight: onnx.TensorProto | None = (
            initializers.get(node.input[1]) if node.op_type == "MatMul" else None
        )
        (add, *others) = consumers.get(node.output[0], [None])
        if (
            weight is not None
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
            w_t = add_constant(
                f"{output}_weight", numpy_helper.to_array(weight).T.copy()
            )
            shape_2d = add_constant(f"{x_2d}_shape", np.array([-1, k]))
            shape_nd = add_constant(f"{output}_shape", np.array([*x_shape[:-1], n]))
            nodes += [
                helper.make_node(
                    "Reshape", [node.input[0], shape_2d], [x_2d], name=x_2d
                ),
                helper.make_node(
                    "Gemm", [x_2d, w_t, bias], [y_2d], name=y_2d, transB=1
                ),
                helper.make_node("Reshape", [y_2d, shape_nd], [output], name=output),
            ]
            fused.add(output)
            changed.add(node.output[0])
            continue
        nodes.append(node)

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
        perm = next(a for a in transpose.attribute if a.name == "perm")
        if perm.ints[0] != 0 or any(
            c.op_type != "Reshape"
            or c.input[1] not in constants
            or 0 in constants[c.input[1]]
            for c in consumers.get(transpose.output[0], [])
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
    # Drop the scalar indices, rank-6 shapes and {K, N} weights nothing reads.
    used: set[str] = {name for n in _all_nodes(graph) for name in n.input}
    used.update(o.name for o in graph.output)
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]

    # Raises for models over protobuf's 2 GB, which would need external data.
    onnx.save(model, onnx_file)
    for data_file in data_files:
        (onnx_file.parent / data_file).unlink()
    logger.debug("postprocessed onnx model: {!r}", str(onnx_file))


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
    for onnx_file in parser.parse_args().onnx_files:
        postprocess(onnx_file)


if __name__ == "__main__":
    main()
