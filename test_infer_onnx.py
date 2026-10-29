import sys

from infer_onnx import parse_args


def _parse(*argv: str):
    sys.argv = ["infer_onnx.py", "--image", "images/bus.jpg", *argv]
    return parse_args()


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
    test_parse_box_prompt()
    print("ok")
