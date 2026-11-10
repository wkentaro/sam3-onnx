import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from infer_onnx import load_for_coreml
from postprocess_onnx import convert_float16
from postprocess_onnx import convert_int8
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


def test_convert_float16() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        rng = np.random.default_rng(0)
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["m"], name="matmul"),
                helper.make_node(
                    "Constant",
                    [],
                    ["b"],
                    name="constant",
                    value=numpy_helper.from_array(
                        np.array([0.5, -0.25, 1e9], np.float32)
                    ),
                ),
                helper.make_node("Min", ["m", "b"], ["clipped"], name="min"),
                helper.make_node(
                    "Cast", ["i"], ["i_float"], name="cast", to=TensorProto.FLOAT
                ),
                helper.make_node("Add", ["clipped", "i_float"], ["y"], name="add"),
                helper.make_node("ArgMax", ["y"], ["y_int"], name="argmax", axis=1),
                # The scales of Resize stay float32, and its data becomes float16.
                helper.make_node(
                    "Resize", ["image", "", "scales"], ["resized"], name="resize"
                ),
            ],
            "test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4]),
                helper.make_tensor_value_info("i", TensorProto.INT64, [2, 3]),
                helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 2, 2]),
            ],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("y_int", TensorProto.INT64, None),
                helper.make_tensor_value_info("resized", TensorProto.FLOAT, None),
            ],
            [
                numpy_helper.from_array(rng.random((4, 3), dtype=np.float32), "w"),
                numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales"),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        onnx_file = tmp_path / "model.onnx"
        fp16_file = tmp_path / "model_fp16.onnx"
        onnx.save(model, onnx_file)
        feeds = {
            "x": rng.random((2, 4), dtype=np.float32),
            "i": rng.integers(0, 10, (2, 3)),
            "image": rng.random((1, 1, 2, 2), dtype=np.float32),
        }
        expected = onnxruntime.InferenceSession(onnx_file).run(None, feeds)

        convert_float16(onnx_file, fp16_file)

        # The same inputs and outputs, in float32.
        session = onnxruntime.InferenceSession(fp16_file)
        assert [i.type for i in session.get_inputs()] == [
            "tensor(float)",
            "tensor(int64)",
            "tensor(float)",
        ]
        y, y_int, resized = (np.asarray(o) for o in session.run(None, feeds))
        assert y.dtype == np.float32 and resized.dtype == np.float32
        np.testing.assert_allclose(y, np.asarray(expected[0]), rtol=1e-3)
        np.testing.assert_array_equal(y_int, np.asarray(expected[1]))
        np.testing.assert_allclose(resized, np.asarray(expected[2]), rtol=1e-3)

        converted = onnx.load(fp16_file)
        types = {i.name: i.data_type for i in converted.graph.initializer}
        assert types == {"w": TensorProto.FLOAT16, "scales": TensorProto.FLOAT}
        (constant,) = [n for n in converted.graph.node if n.name == "constant"]
        value = numpy_helper.to_array(constant.attribute[0].t)
        # 1e9 is beyond float16, and stays finite.
        assert value.dtype == np.float16 and np.isfinite(value).all()


def test_convert_float16_keeps_counts_float32() -> None:
    # The mask stability score of SAM3's point decoder: the ratio of two pixel
    # counts, which pass 65504, the largest float16, on a 288x288 mask.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        def count(name: str, threshold: str) -> list[onnx.NodeProto]:
            return [
                helper.make_node("Greater", ["mask", threshold], [f"{name}_bool"]),
                helper.make_node(
                    "Cast", [f"{name}_bool"], [f"{name}_int"], to=TensorProto.INT64
                ),
                helper.make_node(
                    "ReduceSum", [f"{name}_int"], [f"{name}_sum"], keepdims=0
                ),
                helper.make_node("Cast", [f"{name}_sum"], [name], to=TensorProto.FLOAT),
            ]

        graph = helper.make_graph(
            [
                *count("intersection", "high"),
                *count("union", "low"),
                helper.make_node("Div", ["intersection", "union"], ["ratio"]),
                helper.make_node("Greater", ["ratio", "stable"], ["is_stable"]),
            ],
            "test",
            [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [288, 288])],
            [
                helper.make_tensor_value_info("ratio", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("is_stable", TensorProto.BOOL, []),
            ],
            [
                numpy_helper.from_array(np.array(0.05, np.float32), "high"),
                numpy_helper.from_array(np.array(-0.05, np.float32), "low"),
                numpy_helper.from_array(np.array(0.8, np.float32), "stable"),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        onnx_file = tmp_path / "model.onnx"
        fp16_file = tmp_path / "model_fp16.onnx"
        onnx.save(model, onnx_file)
        convert_float16(onnx_file, fp16_file)

        # 70000 pixels above 0.05 and 82944 above -0.05.
        mask = np.zeros(288 * 288, np.float32)
        mask[:70000] = 1
        # onnxruntime on CPU computes float16 nodes in float32, which hides an
        # overflow, so run the reference implementation.
        ratio, is_stable = ReferenceEvaluator(str(fp16_file)).run(
            None, {"mask": mask.reshape(288, 288)}
        )
        np.testing.assert_allclose(np.asarray(ratio), 70000 / 82944, rtol=1e-3)
        assert is_stable


def test_convert_int8() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        rng = np.random.default_rng(0)
        # A linear layer as postprocess leaves it, Gemm(transB=1) with a bias,
        # then a MatMul of two activations, as in attention.
        graph = helper.make_graph(
            [
                helper.make_node("Gemm", ["x", "w", "b"], ["h"], transB=1),
                helper.make_node("Transpose", ["h"], ["h_t"]),
                helper.make_node("MatMul", ["h", "h_t"], ["y"]),
            ],
            "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 64])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 8])],
            [
                numpy_helper.from_array(
                    rng.standard_normal((32, 64), dtype=np.float32), "w"
                ),
                numpy_helper.from_array(rng.standard_normal(32, dtype=np.float32), "b"),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        onnx_file = tmp_path / "model.onnx"
        int8_file = tmp_path / "model_int8.onnx"
        onnx.save(model, onnx_file)
        feeds = {"x": rng.standard_normal((8, 64), dtype=np.float32)}
        (expected,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)

        convert_int8(onnx_file, int8_file)

        (y,) = onnxruntime.InferenceSession(int8_file).run(None, feeds)
        y = np.asarray(y)
        assert y.dtype == np.float32
        expected = np.asarray(expected)
        assert np.linalg.norm(y - expected) < 0.01 * np.linalg.norm(expected)
        ops = [n.op_type for n in onnx.load(int8_file).graph.node]
        # The weight's MatMul is int8, the activations' stays float32.
        assert "Gemm" not in ops
        assert ops.count("MatMulInteger") == 1 and ops.count("MatMul") == 1


if __name__ == "__main__":
    test_postprocess()
    test_convert_float16()
    test_convert_float16_keeps_counts_float32()
    test_convert_int8()
    print("ok")
