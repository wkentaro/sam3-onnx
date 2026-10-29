import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

from postprocess_onnx import postprocess


def test_postprocess() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        # x -> Gather(scalar) -> Reshape(rank 6) -> Transpose -> Reshape -> y,
        # as the windowed attention and rotary encoding of SAM3's image encoder.
        initializers = [
            numpy_helper.from_array(np.array(1), "index"),
            numpy_helper.from_array(np.array([1, 4, 3, 2, 2, 5]), "shape6"),
            numpy_helper.from_array(np.array([-1, 3, 2, 5]), "shape4"),
            numpy_helper.from_array(np.array([0, -1]), "shape_copy"),
            numpy_helper.from_array(np.ones(4096, np.float32), "weight"),
            numpy_helper.from_array(np.arange(20, dtype=np.float32).reshape(4, 5), "w"),
            numpy_helper.from_array(np.arange(5, dtype=np.float32), "b"),
        ]
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Gather", ["x", "index"], ["g"], name="gather", axis=1
                ),
                helper.make_node("Reshape", ["g", "shape6"], ["r"], name="reshape"),
                helper.make_node(
                    "Transpose", ["r"], ["t"], name="transpose", perm=[0, 1, 3, 2, 4, 5]
                ),
                helper.make_node("Reshape", ["t", "shape4"], ["y"], name="reshape2"),
                # The same, but the last Reshape copies the leading 1, so it stays.
                helper.make_node("Reshape", ["g", "shape6"], ["r0"], name="reshape0"),
                helper.make_node(
                    "Transpose",
                    ["r0"],
                    ["t0"],
                    name="transpose0",
                    perm=[0, 1, 3, 2, 4, 5],
                ),
                helper.make_node(
                    "Reshape", ["t0", "shape_copy"], ["y0"], name="reshape_copy"
                ),
                helper.make_node(
                    "ReduceSum", ["weight"], ["s"], name="sum", keepdims=0
                ),
                # z -> MatMul -> Add -> v, as a linear layer.
                helper.make_node("MatMul", ["z", "w"], ["m"], name="matmul"),
                helper.make_node("Add", ["m", "b"], ["v"], name="add"),
            ],
            "test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 2, 6, 10]),
                helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3, 4]),
            ],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("y0", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("s", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("v", TensorProto.FLOAT, None),
            ],
            initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        onnx_file = tmp_path / "model.onnx"
        onnx.save(
            model, onnx_file, save_as_external_data=True, location="model.onnx.data"
        )
        assert (tmp_path / "model.onnx.data").exists()

        rng = np.random.default_rng(0)
        feeds = {
            "x": rng.random((4, 2, 6, 10), dtype=np.float32),
            "z": rng.random((2, 3, 4), dtype=np.float32),
        }
        expected = onnxruntime.InferenceSession(onnx_file).run(None, feeds)

        postprocess(onnx_file)

        assert not (tmp_path / "model.onnx.data").exists()
        actual = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        processed = onnx.load(onnx_file)
        constants = {
            i.name: numpy_helper.to_array(i) for i in processed.graph.initializer
        }
        assert "MatMul" not in {node.op_type for node in processed.graph.node}
        for node in processed.graph.node:
            if node.op_type == "Gemm":
                assert node.attribute[0].name == "transB" and node.attribute[0].i == 1
            if node.op_type in ("Gather", "Reshape"):
                assert constants[node.input[1]].ndim == 1
            if node.op_type == "Reshape" and node.name != "reshape0":
                assert len(constants[node.input[1]]) <= 5


if __name__ == "__main__":
    test_postprocess()
    print("ok")
