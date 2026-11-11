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
from postprocess_onnx import fuse_attention
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
            # An initializer can be an output too.
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 8]),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, [32, 64]),
            ],
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
        (expected, _) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)

        convert_int8(onnx_file, int8_file)

        (y, _) = onnxruntime.InferenceSession(int8_file).run(None, feeds)
        y = np.asarray(y)
        assert y.dtype == np.float32
        expected = np.asarray(expected)
        assert np.linalg.norm(y - expected) < 0.01 * np.linalg.norm(expected)
        ops = [n.op_type for n in onnx.load(int8_file).graph.node]
        # The weight's MatMul is int8, the activations' stays float32.
        assert "Gemm" not in ops
        assert ops.count("MatMulInteger") == 1 and ops.count("MatMul") == 1


def test_fuse_attention() -> None:
    # The attention of sam3.model.vitdet.Attention over a 2x3 window, as torch
    # exports it: batch 2, heads 2, tokens 6, channels 4.
    (batch, heads, h, w, channels) = (2, 2, 2, 3, 4)
    tokens = h * w

    def shape(name: str, dims: list[int]) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(dims, np.int64), name)

    scale = numpy_helper.from_array(np.array([channels**-0.25], np.float32), "s")
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["q", "s"], ["q_s"]),
            helper.make_node("Reshape", ["k", "k_3d"], ["k_r"]),
            helper.make_node("Transpose", ["k_r"], ["k_t"], perm=[0, 2, 1]),
            helper.make_node("Reshape", ["k_t", "k_4d"], ["k_t4"]),
            helper.make_node("Mul", ["k_t4", "s"], ["k_s"]),
            helper.make_node("MatMul", ["q_s", "k_s"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["weights"], axis=-1),
            helper.make_node("MatMul", ["weights", "v"], ["heads"]),
            helper.make_node("Reshape", ["heads", "heads_5d"], ["heads_r"]),
            helper.make_node(
                "Transpose", ["heads_r"], ["heads_t"], perm=[0, 2, 3, 1, 4]
            ),
            helper.make_node("Reshape", ["heads_t", "y_shape"], ["y"]),
        ],
        "test",
        [
            helper.make_tensor_value_info(
                x, TensorProto.FLOAT, [batch, heads, tokens, channels]
            )
            for x in "qkv"
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [
            scale,
            shape("k_3d", [batch * heads, tokens, channels]),
            shape("k_4d", [batch, heads, channels, tokens]),
            shape("heads_5d", [batch, heads, h, w, channels]),
            shape("y_shape", [batch, h, w, heads * channels]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    rng = np.random.default_rng(0)
    feeds = {
        x: rng.normal(size=(batch, heads, tokens, channels)).astype(np.float32)
        for x in "qkv"
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = pathlib.Path(tmp_dir) / "model.onnx"
        onnx.save(model, onnx_file)
        (expected,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
        fuse_attention(onnx_file)
        ops = [n.op_type for n in onnx.load(onnx_file).graph.node]
        (actual,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
    assert ops.count("MultiHeadAttention") == 1 and "Softmax" not in ops
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_fuse_attention_one_head() -> None:
    # The attention of RoPEAttention in the memory attention, as torch exports
    # it: one head, and q, k, and v in (batch, 1, tokens, channels), with the
    # batch unknown as in all but its first layer.
    (q_tokens, kv_tokens, channels) = (6, 5, 4)
    scale = numpy_helper.from_array(np.array([channels**-0.25], np.float32), "s")
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["q", "s"], ["q_s"]),
            helper.make_node("Transpose", ["k"], ["k_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Mul", ["k_t", "s"], ["k_s"]),
            helper.make_node("MatMul", ["q_s", "k_s"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["weights"], axis=-1),
            helper.make_node("MatMul", ["weights", "v"], ["out"]),
            helper.make_node("Relu", ["out"], ["y"]),
        ],
        "test",
        [
            helper.make_tensor_value_info(
                x, TensorProto.FLOAT, ["batch", 1, n, channels]
            )
            for x, n in zip("qkv", (q_tokens, kv_tokens, kv_tokens))
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    rng = np.random.default_rng(0)
    feeds = {
        x: rng.normal(size=(1, 1, n, channels)).astype(np.float32)
        for x, n in zip("qkv", (q_tokens, kv_tokens, kv_tokens))
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = pathlib.Path(tmp_dir) / "model.onnx"
        onnx.save(model, onnx_file)
        (expected,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
        fuse_attention(onnx_file)
        ops = [n.op_type for n in onnx.load(onnx_file).graph.node]
        (actual,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
        # The float16 copy takes the queries in halves.
        fp16_file = pathlib.Path(tmp_dir) / "model_fp16.onnx"
        convert_float16(onnx_file, fp16_file)
        fp16_ops = [n.op_type for n in onnx.load(fp16_file).graph.node]
        (fp16_actual,) = onnxruntime.InferenceSession(fp16_file).run(None, feeds)
    assert ops.count("MultiHeadAttention") == 1 and "Softmax" not in ops
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)
    assert fp16_ops.count("MultiHeadAttention") == 2
    np.testing.assert_allclose(np.asarray(fp16_actual), np.asarray(expected), atol=1e-2)


def test_fuse_attention_sequence_first() -> None:
    # The attention of nn.MultiheadAttention in the decoder, as torch exports
    # it: q, k, and v in (tokens, batch 1, heads * channels), and the scale
    # that scaled_dot_product_attention computes from the shape.
    (heads, q_tokens, kv_tokens, channels) = (2, 3, 5, 4)
    dim = heads * channels

    def shape(name: str, dims: list[int]) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(dims, np.int64), name)

    def split(x: str, tokens: int) -> list[onnx.NodeProto]:
        return [
            helper.make_node("Reshape", [x, f"{x}_3d"], [f"{x}_r"]),
            helper.make_node("Transpose", [f"{x}_r"], [f"{x}_t"], perm=[1, 0, 2]),
            helper.make_node("Reshape", [f"{x}_t", f"{x}_4d"], [f"{x}_h"]),
        ]

    graph = helper.make_graph(
        [
            *split("q", q_tokens),
            *split("k", kv_tokens),
            *split("v", kv_tokens),
            helper.make_node("Sqrt", ["channels"], ["channels_sqrt"]),
            helper.make_node("Div", ["one", "channels_sqrt"], ["inverse"]),
            helper.make_node("Cast", ["inverse"], ["inverse_f"], to=TensorProto.FLOAT),
            helper.make_node("Sqrt", ["inverse_f"], ["s"]),
            helper.make_node("Mul", ["q_h", "s"], ["q_s"]),
            helper.make_node("Transpose", ["k_h"], ["k_t4"], perm=[0, 1, 3, 2]),
            helper.make_node("Mul", ["k_t4", "s"], ["k_s"]),
            helper.make_node("MatMul", ["q_s", "k_s"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["weights"], axis=-1),
            helper.make_node("MatMul", ["weights", "v_h"], ["heads"]),
            helper.make_node("Transpose", ["heads"], ["heads_t"], perm=[2, 0, 1, 3]),
            helper.make_node("Reshape", ["heads_t", "y_shape"], ["y"]),
        ],
        "test",
        [
            helper.make_tensor_value_info(x, TensorProto.FLOAT, [tokens, 1, dim])
            for x, tokens in zip("qkv", (q_tokens, kv_tokens, kv_tokens))
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [
            numpy_helper.from_array(np.array(channels, np.float32), "channels"),
            numpy_helper.from_array(np.array(1, np.float32), "one"),
            *[
                shape(f"{x}_{n}d", d)
                for x, tokens in zip("qkv", (q_tokens, kv_tokens, kv_tokens))
                for n, d in [
                    (3, [tokens, heads, channels]),
                    (4, [1, heads, tokens, channels]),
                ]
            ],
            shape("y_shape", [q_tokens, dim]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    rng = np.random.default_rng(0)
    feeds = {
        x: rng.normal(size=(tokens, 1, dim)).astype(np.float32)
        for x, tokens in zip("qkv", (q_tokens, kv_tokens, kv_tokens))
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = pathlib.Path(tmp_dir) / "model.onnx"
        onnx.save(model, onnx_file)
        (expected,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
        fuse_attention(onnx_file)
        ops = [n.op_type for n in onnx.load(onnx_file).graph.node]
        (actual,) = onnxruntime.InferenceSession(onnx_file).run(None, feeds)
    assert ops.count("MultiHeadAttention") == 1 and "Softmax" not in ops
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


if __name__ == "__main__":
    test_postprocess()
    test_convert_float16()
    test_convert_float16_keeps_counts_float32()
    test_convert_int8()
    test_fuse_attention()
    test_fuse_attention_one_head()
    test_fuse_attention_sequence_first()
    print("ok")
