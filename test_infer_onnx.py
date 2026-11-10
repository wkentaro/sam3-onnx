import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import onnxruntime

from infer_onnx import main
from infer_onnx import model_device
from infer_onnx import model_file
from infer_onnx import parse_args
from infer_onnx import run_on_device
from infer_onnx import run_point_decoder


def _parse(*argv: str):
    sys.argv = ["infer_onnx.py", "--image", "images/bus.jpg", *argv]
    return parse_args()


def test_language_encoder_selection() -> None:
    for variant in [None, "s0", "s1", "l"]:
        argv = ["infer_onnx.py", "--image", "images/bus.jpg", "--text-prompt", "person"]
        if variant:
            argv += ["--litetext", variant]
        with (
            patch.object(sys, "argv", argv),
            patch("infer_onnx.create_sessions", side_effect=InterruptedError) as create,
        ):
            try:
                main()
            except InterruptedError:
                pass
            language = (
                f"litetext_{variant}_language_encoder"
                if variant
                else "language_encoder"
            )
            create.assert_called_once_with(
                device="cpu",
                names=["image_encoder", language, "decoder"],
                fp16=False,
                int8=False,
            )


def test_model_file() -> None:
    assert model_file("decoder", "cuda", fp16=False) == "models/sam3_decoder.onnx"
    for device in ["cuda", "webgpu", "coreml"]:
        assert (
            model_file("decoder", device, fp16=True) == "models/sam3_decoder_fp16.onnx"
        )
    # A model that the CPU runs stays float32.
    assert model_file("decoder", "cpu", fp16=True) == "models/sam3_decoder.onnx"
    # The CPU runs the int8 image encoder alone.
    assert (
        model_file("image_encoder", "cpu", fp16=False, int8=True)
        == "models/sam3_image_encoder_int8.onnx"
    )
    assert model_file("decoder", "cpu", fp16=False, int8=True) == (
        "models/sam3_decoder.onnx"
    )


def test_model_device() -> None:
    for device in ["cpu", "cuda", "webgpu"]:
        for name in ["image_encoder", "decoder"]:
            assert model_device(name, device, has_webgpu=False) == device
    # CoreML runs the image encoder alone.
    for has_webgpu in [False, True]:
        assert model_device("image_encoder", "coreml", has_webgpu) == "coreml"
    assert model_device("memory_attention", "coreml", has_webgpu=True) == "webgpu"
    assert model_device("memory_attention", "coreml", has_webgpu=False) == "cpu"
    # The fixed-shape memory attention exists for CoreML.
    assert model_device("memory_attention_fixed", "coreml", True) == "coreml"


def test_parse_box_prompt() -> None:
    args = _parse(
        "--text-prompt", "person", "--box-prompt", "0.1,0.2,0.3,0.4",
        "--box-prompt", "0.5,0.6,0.7,0.8,0",
    )  # fmt: skip
    assert args.box_prompt == [[0.1, 0.2, 0.3, 0.4, 1], [0.5, 0.6, 0.7, 0.8, 0]]

    for argv in [
        ["--box-prompt", "0.1,0.2,0.3,0.4", "--box-prompt"],
        ["--box-prompt", "--box-prompt"],
        ["--box-prompt", "0.1,0.2,0.3,0.4,0.5"],
        ["--box-prompt", "0.1,0.2,0.3"],
        ["--box-prompt", "0.1,0.2,0.3,0.4", "--point-prompt", "0.1,0.2"],
        ["--text-prompt", "person", "--single-mask"],
    ]:
        try:
            _parse(*argv)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {argv}")


def test_single_mask() -> None:
    args = _parse("--point-prompt", "0.1,0.2", "--single-mask")
    assert args.single_mask

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], np.ndarray]] = []

        def run(self, output_names, inputs):
            self.calls.append((output_names, inputs["mask_input"].copy()))
            masks = np.full((1, 1, 288, 288), len(self.calls), np.float32)
            return [np.ones((1, 1), np.float32), masks]

    session = Session()
    _, masks = run_point_decoder(
        sess_point_decode=session,  # type: ignore[invalid-argument-type]
        sam2_features=[np.zeros(1)] * 3,
        point_prompt=[[0.1, 0.2, 1]],
        multimask_output=False,
    )
    single = ["single_iou_predictions", "single_low_res_masks"]
    assert [names for names, _ in session.calls] == [single, single]
    # The refinement pass gets the first pass's mask as its mask prompt.
    assert (session.calls[1][1] == 1).all()
    assert (masks == 2).all()


def test_run_on_device() -> None:
    image = np.zeros((3, 4, 4), np.uint8)
    features = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(2, np.float32))  # type: ignore[possibly-missing-attribute]
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.run.return_value = [np.ones(1)]
    assert run_on_device(session, ["a"], {"image": image}) == [np.ones(1)]
    session.io_binding.assert_not_called()

    # On a GPU, the outputs stay on it, and OrtValue inputs bind as they are.
    session = MagicMock()
    session.get_providers.return_value = ["CUDAExecutionProvider"]
    binding = session.io_binding.return_value
    outputs = run_on_device(session, ["a", "b"], {"image": image, "f": features})
    binding.bind_cpu_input.assert_called_once_with("image", image)
    binding.bind_ortvalue_input.assert_called_once_with("f", features)
    assert [c.args for c in binding.bind_output.call_args_list] == [
        ("a", "cuda"),
        ("b", "cuda"),
    ]
    session.run.assert_not_called()
    assert outputs is binding.get_outputs.return_value


if __name__ == "__main__":
    test_language_encoder_selection()
    test_model_file()
    test_model_device()
    test_parse_box_prompt()
    test_single_mask()
    test_run_on_device()
    print("ok")
