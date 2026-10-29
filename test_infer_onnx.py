import sys
from unittest.mock import patch

from infer_onnx import main
from infer_onnx import parse_args


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
                device="cpu", names=["image_encoder", language, "decoder"]
            )


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
    ]:
        try:
            _parse(*argv)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {argv}")


if __name__ == "__main__":
    test_language_encoder_selection()
    test_parse_box_prompt()
    print("ok")
