"""CUDA integration check of video_onnx.VideoInference against
build_sam3_video_model in float32, using pre-exported models."""

import argparse
import inspect
import pathlib
import tempfile
import textwrap
import typing

import numpy as np
import PIL.Image
import torch
from numpy.typing import NDArray
from sam3.model import sam3_image  # type: ignore[unresolved-import]
from sam3.model_builder import build_sam3_video_model  # type: ignore[unresolved-import]

from infer_onnx import create_sessions
from infer_onnx import preprocess_image
from test_video_parity import extract_frames
from test_video_parity import iou
from video_onnx import MODEL_NAMES
from video_onnx import VideoInference
from video_onnx import VideoOutput
from video_onnx import encode_frame

MIN_IOU: float = 0.99
MAX_PROB_DIFF: float = 0.01


def patch_float32(model: typing.Any) -> None:
    """Run the reference in float32, as the models are exported."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tracker = model.tracker
    tracker.bf16_context.__exit__(None, None, None)
    # Without autocast, the bfloat16 memories it stores meet float32 weights.
    prepare = tracker._prepare_memory_conditioned_features

    def prepare_float(*args, output_dict, **kwargs):
        output_dict = {
            key: {
                t: {**out, "maskmem_features": out["maskmem_features"].float()}
                if out.get("maskmem_features") is not None
                else out
                for t, out in outputs.items()
            }
            for key, outputs in output_dict.items()
        }
        return prepare(*args, output_dict=output_dict, **kwargs)

    tracker._prepare_memory_conditioned_features = prepare_float
    # The detector casts the tracker's features to bfloat16 to all-gather them
    # across GPUs, which one GPU doesn't need.
    source = textwrap.dedent(
        inspect.getsource(
            sam3_image.Sam3ImageOnVideoMultiGPU._build_multigpu_buffer_next_chunk
        )
    )
    cast = "[x.to(torch.bfloat16) for x in"
    assert cast in source
    namespace: dict = {}
    exec(source.replace(cast, "[x for x in"), vars(sam3_image), namespace)
    detector = model.detector
    detector._build_multigpu_buffer_next_chunk = namespace[
        "_build_multigpu_buffer_next_chunk"
    ].__get__(detector)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=pathlib.Path,
        default=pathlib.Path("sam3/assets/videos/bedroom.mp4"),
    )
    parser.add_argument("--num-frames", type=int, default=60)
    args = parser.parse_args()

    model = build_sam3_video_model()
    patch_float32(model)

    names = ["image_encoder", *MODEL_NAMES]
    sessions = dict(zip(names, create_sessions(device="cpu", names=names)))

    frames_dir = pathlib.Path(tempfile.mkdtemp())
    extract_frames(args.video, args.num_frames, frames_dir)
    paths = sorted(frames_dir.glob("*.jpg"))
    num_frames = len(paths)
    if num_frames <= 30:
        parser.error("the check prompts on frame 30, so it needs 31 frames or more")
    width, height = PIL.Image.open(paths[0]).size
    frames = [np.asarray(PIL.Image.open(p).convert("RGB")) for p in paths]
    features: dict[int, dict[str, NDArray]] = {}

    def get_features(frame_idx: int) -> dict[str, NDArray]:
        if frame_idx not in features:
            features[frame_idx] = encode_frame(
                sessions["image_encoder"], frames[frame_idx]
            )
        return features[frame_idx]

    state = model.init_state(resource_path=str(frames_dir))
    # Both read the same resized frames, in float32 rather than float16.
    pixels = np.stack([preprocess_image(PIL.Image.fromarray(f)) for f in frames])
    state["input_batch"].img_batch = (
        torch.from_numpy(pixels).cuda().float() / 255 - 0.5
    ) / 0.5
    video = VideoInference(sessions, get_features, num_frames, height, width)

    failures: list[str] = []

    def check(
        name: str, expected: list[VideoOutput], actual: list[VideoOutput]
    ) -> None:
        ious, prob_diffs = [], []
        for e, a in zip(expected, actual, strict=True):
            e_ids, a_ids = e["out_obj_ids"].tolist(), a["out_obj_ids"].tolist()
            if e_ids != a_ids:
                print(f"{name}: object ids {e_ids} != {a_ids}")
                failures.append(name)
                return
            ious += [
                iou(m1, m2)
                for m1, m2 in zip(e["out_binary_masks"], a["out_binary_masks"])
            ]
            prob_diffs += np.abs(e["out_probs"] - a["out_probs"]).tolist()
        num_objects = max((len(e["out_obj_ids"]) for e in expected), default=0)
        print(
            f"{name}: {len(expected)} frames, up to {num_objects} objects, "
            f"min IoU {min(ious, default=1):.4f}, "
            f"max prob diff {max(prob_diffs, default=0):.4f}"
        )
        if min(ious, default=1) < MIN_IOU or max(prob_diffs, default=0) > MAX_PROB_DIFF:
            failures.append(name)

    def prompt(name: str, frame_idx: int, **kwargs) -> None:
        _, expected = model.add_prompt(state, frame_idx, **kwargs)
        _, actual = video.add_prompt(frame_idx, **kwargs)
        check(name, [expected], [actual])

    def propagate(name: str, **kwargs) -> None:
        expected = dict(model.propagate_in_video(state, **kwargs))
        actual = dict(video.propagate_in_video(**kwargs))
        assert list(expected) == list(actual), (list(expected), list(actual))
        check(name, list(expected.values()), list(actual.values()))

    prompt("text", 0, text_str="person")
    propagate("text: forward")

    # The box that test_video_parity.py tracks.
    box_xywh = np.array([[300 / width, 0, 200 / width, 400 / height]])
    prompt("box", 0, boxes_xywh=box_xywh, box_labels=np.array([1]))
    propagate("box: forward")

    prompt("text on frame 30", 30, text_str="person")
    propagate("text on frame 30: forward")
    propagate("text on frame 30: reverse", start_frame_idx=30, reverse=True)

    assert not failures, f"mismatch: {failures}"


if __name__ == "__main__":
    main()
