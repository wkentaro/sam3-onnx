#!/usr/bin/env python3

import argparse
import pathlib

import numpy as np
import onnx
from loguru import logger
from onnx import helper
from onnx import numpy_helper


def postprocess(onnx_file: pathlib.Path) -> None:
    """Rewrite ops that onnxruntime's CoreML EP rejects, and inline the weights.

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
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def add_constant(name: str, value: np.ndarray) -> str:
        graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    changed: set[str] = set()
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
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
        if shape is None or len(shape) != 6 or shape[0] != 1:
            continue
        (transpose, *others) = consumers.get(node.output[0], [None])
        if others or transpose is None or transpose.op_type != "Transpose":
            continue
        perm = next(a for a in transpose.attribute if a.name == "perm")
        if perm.ints[0] != 0 or any(
            c.op_type != "Reshape" for c in consumers.get(transpose.output[0], [])
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
    # Drop the scalar indices and rank-6 shapes that nothing reads anymore.
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
