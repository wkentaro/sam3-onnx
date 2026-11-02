"""CUDA integration check of track_onnx.Tracker against Sam3TrackerPredictor in
float32, using pre-exported models."""

import argparse
import pathlib
import tempfile
import typing

import cv2
import numpy as np
import PIL.Image
import torch
from numpy.typing import NDArray
from sam3.model_builder import build_sam3_video_model  # type: ignore[unresolved-import]

from infer_onnx import create_sessions
from track_onnx import MODEL_NAMES
from track_onnx import Tracker
from track_onnx import encode_frame

MIN_IOU: float = 0.99


def extract_frames(video: pathlib.Path, num_frames: int, out_dir: pathlib.Path):
    # Both trackers read the same JPEGs, so decoding can't differ.
    cap = cv2.VideoCapture(str(video))
    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame)
    cap.release()


def iou(a: NDArray, b: NDArray) -> float:
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


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

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor.bf16_context.__exit__(None, None, None)
    # Without autocast, the bfloat16 memories it stores meet float32 weights.
    prepare = predictor._prepare_memory_conditioned_features

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

    predictor._prepare_memory_conditioned_features = prepare_float

    names = ["image_encoder", *MODEL_NAMES]
    sessions = dict(zip(names, create_sessions(device="cpu", names=names)))

    frames_dir = pathlib.Path(tempfile.mkdtemp())
    extract_frames(args.video, args.num_frames, frames_dir)
    paths = sorted(frames_dir.glob("*.jpg"))
    if len(paths) <= 40:
        parser.error("the check refines on frame 40, so it needs 41 frames or more")
    width, height = PIL.Image.open(paths[0]).size
    features: dict[int, list[NDArray]] = {}

    def get_features(frame_idx: int) -> list[NDArray]:
        if frame_idx not in features:
            frame = np.asarray(PIL.Image.open(paths[frame_idx]).convert("RGB"))
            features[frame_idx] = encode_frame(sessions["image_encoder"], frame)
        return features[frame_idx]

    def new_pair() -> tuple[dict, Tracker]:
        state = predictor.init_state(video_path=str(frames_dir))
        return state, Tracker(sessions, get_features, len(paths), height, width)

    failures: list[str] = []

    def check(name: str, expected: list[NDArray], actual: list[NDArray]) -> None:
        ious = [iou(e, a) for e, a in zip(expected, actual, strict=True)]
        print(f"{name}: {len(ious)} masks, min IoU {min(ious):.4f}")
        if min(ious) < MIN_IOU:
            failures.append(name)

    def both(name: str, state: dict, tracker: Tracker, method: str, **kwargs) -> None:
        torch_kwargs = {
            k: torch.from_numpy(v) if k == "mask" else v for k, v in kwargs.items()
        }
        _, expected_ids, _, expected = getattr(predictor, method)(state, **torch_kwargs)
        _, actual_ids, actual = getattr(tracker, method)(**kwargs)
        assert list(expected_ids) == actual_ids, (expected_ids, actual_ids)
        check(name, list(expected.cpu().numpy()), list(actual))

    def propagate(
        name: str, state: dict, tracker: Tracker, **kwargs: typing.Any
    ) -> None:
        expected = {
            frame_idx: masks.cpu().numpy()
            for frame_idx, _, _, masks, _ in predictor.propagate_in_video(
                state,
                start_frame_idx=kwargs.get("start_frame_idx"),
                max_frame_num_to_track=None,
                reverse=kwargs.get("reverse", False),
                propagate_preflight=True,
                tqdm_disable=True,
            )
        }
        actual = {f: masks for f, _, masks, _ in tracker.propagate_in_video(**kwargs)}
        assert list(expected) == list(actual)
        check(name, list(expected.values()), list(actual.values()))

    def rel(points: list[list[float]]) -> NDArray[np.float32]:
        return np.array([[x / width, y / height] for x, y in points], np.float32)

    # Points from examples/sam3_for_sam2_video_task_example.ipynb.
    state, tracker = new_pair()
    both(
        "point",
        state,
        tracker,
        "add_new_points_or_box",
        frame_idx=0,
        obj_id=1,
        points=rel([[210, 350]]),
        labels=np.array([1], np.int32),
    )
    propagate("point: forward", state, tracker)
    both(
        "point: refine",
        state,
        tracker,
        "add_new_points_or_box",
        frame_idx=40,
        obj_id=1,
        points=rel([[82, 410]]),
        labels=np.array([0], np.int32),
        clear_old_points=False,
    )
    propagate("point: forward after refine", state, tracker)
    propagate(
        "point: reverse", state, tracker, start_frame_idx=len(paths) - 1, reverse=True
    )
    mask = (
        next(predictor.propagate_in_video(state, 0, 0, False, tqdm_disable=True))[3][
            0, 0
        ]
        .cpu()
        .numpy()
        > 0
    )

    state, tracker = new_pair()
    both(
        "box",
        state,
        tracker,
        "add_new_points_or_box",
        frame_idx=0,
        obj_id=4,
        points=rel([[460, 60]]),
        labels=np.array([1], np.int32),
        box=np.array([300 / width, 0, 500 / width, 400 / height], np.float32),
    )
    propagate("box: forward", state, tracker)

    state, tracker = new_pair()
    both("mask", state, tracker, "add_new_mask", frame_idx=0, obj_id=1, mask=mask)
    propagate("mask: forward", state, tracker)

    state, tracker = new_pair()
    both(
        "objects",
        state,
        tracker,
        "add_new_points_or_box",
        frame_idx=0,
        obj_id=2,
        points=rel([[200, 300], [275, 175]]),
        labels=np.array([1, 0], np.int32),
    )
    both(
        "objects",
        state,
        tracker,
        "add_new_points_or_box",
        frame_idx=0,
        obj_id=3,
        points=rel([[400, 150]]),
        labels=np.array([1], np.int32),
    )
    propagate("objects: forward", state, tracker)
    predictor.remove_object(state, 2)
    assert tracker.remove_object(2) == [3]
    propagate("objects: forward after remove", state, tracker)

    assert not failures, f"IoU below {MIN_IOU}: {failures}"


if __name__ == "__main__":
    main()
