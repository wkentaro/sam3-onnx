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
            numpy_helper.from_array(np.ones(4096, np.float32), "weight"),
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
                helper.make_node(
                    "ReduceSum", ["weight"], ["w"], name="sum", keepdims=0
                ),
            ],
            "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 2, 6, 10])],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, None),
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

        x = np.random.default_rng(0).random((4, 2, 6, 10), dtype=np.float32)
        expected = onnxruntime.InferenceSession(onnx_file).run(None, {"x": x})

        postprocess(onnx_file)

        assert not (tmp_path / "model.onnx.data").exists()
        actual = onnxruntime.InferenceSession(onnx_file).run(None, {"x": x})
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        processed = onnx.load(onnx_file)
        constants = {
            i.name: numpy_helper.to_array(i) for i in processed.graph.initializer
        }
        for node in processed.graph.node:
            if node.op_type in ("Gather", "Reshape"):
                assert constants[node.input[1]].ndim == 1
            if node.op_type == "Reshape":
                assert len(constants[node.input[1]]) <= 5


if __name__ == "__main__":
    test_postprocess()
    print("ok")
