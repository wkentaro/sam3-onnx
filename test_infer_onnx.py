import sys
from unittest.mock import patch

import numpy as np

from infer_onnx import main
from infer_onnx import model_device
from infer_onnx import model_file
from infer_onnx import parse_args
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
                device="cpu", names=["image_encoder", language, "decoder"], fp16=False
            )


def test_model_file() -> None:
    assert model_file("decoder", "cuda", fp16=False) == "models/sam3_decoder.onnx"
    for device in ["cuda", "webgpu", "coreml"]:
        assert (
            model_file("decoder", device, fp16=True) == "models/sam3_decoder_fp16.onnx"
        )
    # A model that the CPU runs stays float32.
    assert model_file("decoder", "cpu", fp16=True) == "models/sam3_decoder.onnx"


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


if __name__ == "__main__":
    test_language_encoder_selection()
    test_model_file()
    test_model_device()
    test_parse_box_prompt()
    test_single_mask()
    print("ok")
