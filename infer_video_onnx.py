#!/usr/bin/env python3

import argparse
import pathlib
import sys

import cv2
import imgviz
import numpy as np
import onnxruntime
from loguru import logger
from numpy.typing import NDArray

import video_onnx
from infer_onnx import DEVICE_HELP
from infer_onnx import DEVICES
from infer_onnx import FP16_HELP
from infer_onnx import create_sessions
from track_onnx import MODEL_NAMES
from track_onnx import Tracker
from track_onnx import encode_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", type=pathlib.Path, required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--output", type=pathlib.Path, required=True, help="Path to the output mp4."
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        help="Text prompt to find and track every matching object.",
    )
    parser.add_argument(
        "--point-prompt",
        type=str,
        action="append",
        help="Point prompt for the object in format: x,y[,label] (normalized, "
        "label 1 for foreground and 0 for background). Repeatable.",
    )
    parser.add_argument(
        "--box-prompt",
        type=str,
        help="Box prompt for the object in format: cx,cy,w,h (normalized).",
    )
    parser.add_argument(
        "--frame-idx", type=int, default=0, help="Frame to put the prompts on."
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help=DEVICE_HELP,
    )
    parser.add_argument("--fp16", action="store_true", help=FP16_HELP)
    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    if args.text_prompt is not None:
        if args.point_prompt or args.box_prompt:
            parser.error(
                "--text-prompt can't be used with --point-prompt or --box-prompt"
            )
    elif not (args.point_prompt or args.box_prompt):
        parser.error("one of --text-prompt, --point-prompt, --box-prompt is required")
    if args.point_prompt:
        points = [[float(x) for x in p.split(",")] for p in args.point_prompt]
        if any(len(p) not in (2, 3) for p in points):
            parser.error("--point-prompt must have 2 or 3 values: x, y[, label]")
        args.point_prompt = [p if len(p) == 3 else [*p, 1] for p in points]
    if args.box_prompt:
        box = [float(x) for x in args.box_prompt.split(",")]
        if len(box) != 4:
            parser.error("--box-prompt must have 4 values: cx, cy, w, h")
        cx, cy, w, h = box
        args.box_prompt = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return args


def read_video(path: pathlib.Path) -> tuple[list[NDArray[np.uint8]], float]:
    # ponytail: holds every frame in memory; stream them for long videos.
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1])
    cap.release()
    return frames, fps


def main() -> None:
    args = parse_args()

    model_names = MODEL_NAMES if args.text_prompt is None else video_onnx.MODEL_NAMES
    names = ["image_encoder", *model_names]
    sessions = dict(
        zip(names, create_sessions(device=args.device, names=names, fp16=args.fp16))
    )

    frames, fps = read_video(args.video)
    if not frames:
        logger.error("no frames in {}", args.video)
        sys.exit(1)
    if not 0 <= args.frame_idx < len(frames):
        logger.error("--frame-idx must be in [0, {})", len(frames))
        sys.exit(1)
    height, width = frames[0].shape[:2]
    logger.debug("video: {} frames of {}x{}", len(frames), width, height)

    if args.text_prompt is None:
        masks, obj_ids = track(args, sessions, frames)
    else:
        masks, obj_ids = detect_and_track(args, sessions, frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    for frame_idx, frame in enumerate(frames):
        viz = draw(frame, masks[frame_idx], obj_ids[frame_idx])
        writer.write(np.ascontiguousarray(viz[:, :, ::-1]))
    writer.release()
    logger.info("saved: {}", args.output)


def draw(
    frame: NDArray[np.uint8], masks: NDArray[np.bool_], obj_ids: list[int]
) -> NDArray[np.uint8]:
    """Overlay each mask in a color picked by its object id, so an object
    keeps its color as others come and go."""
    label = np.zeros(frame.shape[:2], np.int32)
    for mask, obj_id in zip(masks, obj_ids):
        # The colormap has 256 colors, and label 0 is the background.
        label[mask] = obj_id % 255 + 1
    return imgviz.label2rgb(label, image=frame, alpha=0.5)


def track(
    args: argparse.Namespace,
    sessions: dict[str, onnxruntime.InferenceSession],
    frames: list[NDArray[np.uint8]],
) -> tuple[dict[int, NDArray[np.bool_]], dict[int, list[int]]]:
    """Track the object at the point or box prompt."""
    height, width = frames[0].shape[:2]
    tracker = Tracker(
        sessions,
        lambda i: encode_frame(sessions["image_encoder"], frames[i]),
        len(frames),
        height,
        width,
    )
    points = labels = None
    if args.point_prompt:
        points = np.array([p[:2] for p in args.point_prompt], np.float32)
        labels = np.array([p[2] for p in args.point_prompt], np.int32)
    tracker.add_new_points_or_box(
        frame_idx=args.frame_idx,
        obj_id=1,
        points=points,
        labels=labels,
        box=None if args.box_prompt is None else np.array(args.box_prompt),
    )

    masks: dict[int, NDArray[np.bool_]] = {}
    for reverse in [False, True]:
        for frame_idx, _, video_res_masks, _ in tracker.propagate_in_video(
            start_frame_idx=args.frame_idx, reverse=reverse
        ):
            masks[frame_idx] = video_res_masks[:, 0] > 0
            logger.debug("tracked frame {}", frame_idx)
    return masks, {frame_idx: [0] for frame_idx in masks}


def detect_and_track(
    args: argparse.Namespace,
    sessions: dict[str, onnxruntime.InferenceSession],
    frames: list[NDArray[np.uint8]],
) -> tuple[dict[int, NDArray[np.bool_]], dict[int, list[int]]]:
    """Find and track every object of the text prompt, as Sam3VideoPredictor
    does with propagation_direction="both"."""
    height, width = frames[0].shape[:2]
    video = video_onnx.VideoInference(
        sessions,
        lambda i: video_onnx.encode_frame(sessions["image_encoder"], frames[i]),
        len(frames),
        height,
        width,
    )
    video.add_prompt(args.frame_idx, text_str=args.text_prompt)
    outputs: dict[int, video_onnx.VideoOutput] = {}
    for reverse in [False, True]:
        for frame_idx, out in video.propagate_in_video(
            start_frame_idx=args.frame_idx, reverse=reverse
        ):
            outputs[frame_idx] = out
            logger.debug("frame {}: objects {}", frame_idx, out["out_obj_ids"].tolist())
    masks = {i: out["out_binary_masks"] for i, out in outputs.items()}
    obj_ids = {i: out["out_obj_ids"].tolist() for i, out in outputs.items()}
    return masks, obj_ids


if __name__ == "__main__":
    main()
