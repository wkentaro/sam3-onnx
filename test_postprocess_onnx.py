import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

from infer_onnx import load_for_coreml
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
            numpy_helper.from_array(
                np.arange(20, dtype=np.float32).reshape(5, 4), "w_nk"
            ),
            numpy_helper.from_array(np.arange(20).reshape(4, 5), "w_int"),
            numpy_helper.from_array(np.arange(5), "b_int"),
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
                # The same, but Transpose reverses the axes, so it stays.
                helper.make_node("Reshape", ["g", "shape6"], ["r1"], name="reshape1"),
                helper.make_node("Transpose", ["r1"], ["t1"], name="transpose1"),
                helper.make_node("Reshape", ["t1", "shape4"], ["y1"], name="reshape3"),
                helper.make_node(
                    "ReduceSum", ["weight"], ["s"], name="sum", keepdims=0
                ),
                # z -> MatMul -> Add -> v, as a linear layer.
                helper.make_node("MatMul", ["z", "w"], ["m"], name="matmul"),
                helper.make_node("Add", ["m", "b"], ["v"], name="add"),
                # A linear layer as torch exports it, with W {N, K} (transB=1).
                helper.make_node(
                    "Gemm", ["z_2d", "w_nk", "b"], ["u"], name="gemm", transB=1
                ),
                # The same in int64, which onnxruntime's Gemm rejects, so it stays.
                helper.make_node(
                    "MatMul", ["z_int", "w_int"], ["m_int"], name="matmul_int"
                ),
                helper.make_node("Add", ["m_int", "b_int"], ["v_int"], name="add_int"),
            ],
            "test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 2, 6, 10]),
                helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("z_int", TensorProto.INT64, [2, 3, 4]),
                helper.make_tensor_value_info("z_2d", TensorProto.FLOAT, [3, 4]),
            ],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("y0", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("y1", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("s", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("v", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("u", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("v_int", TensorProto.INT64, None),
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
            "z_int": rng.integers(0, 10, (2, 3, 4)),
            "z_2d": rng.random((3, 4), dtype=np.float32),
        }
        expected = onnxruntime.InferenceSession(onnx_file).run(None, feeds)

        postprocess(onnx_file)

        # Kept for other models that may share it, and no longer read.
        (tmp_path / "model.onnx.data").unlink()
        actual = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        processed = onnx.load(onnx_file)
        constants = {
            i.name: numpy_helper.to_array(i) for i in processed.graph.initializer
        }
        matmuls = [n.name for n in processed.graph.node if n.op_type == "MatMul"]
        assert matmuls == ["matmul_int"]
        gemms = [n for n in processed.graph.node if n.op_type == "Gemm"]
        assert len(gemms) == 2
        # transB=0, which onnxruntime's WebGPU EP runs at twice the speed.
        assert not any(a.name == "transB" and a.i for n in gemms for a in n.attribute)
        # Back to transB=1 for the CoreML EP, with the same outputs.
        coreml_model = load_for_coreml(str(onnx_file))
        actual = onnxruntime.InferenceSession(coreml_model).run(None, feeds)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        coreml_gemms = [
            n
            for n in onnx.load_from_string(coreml_model).graph.node
            if n.op_type == "Gemm"
        ]
        assert all(a.i for n in coreml_gemms for a in n.attribute if a.name == "transB")
        assert all(any(a.name == "transB" for a in n.attribute) for n in coreml_gemms)
        for node in processed.graph.node:
            if node.op_type in ("Gather", "Reshape"):
                assert constants[node.input[1]].ndim == 1
            if node.op_type == "Reshape" and node.name not in ("reshape0", "reshape1"):
                assert len(constants[node.input[1]]) <= 5


if __name__ == "__main__":
    test_postprocess()
    print("ok")
