#!/usr/bin/env python3

import argparse
import pathlib
import sys

import cv2
import imgviz
import numpy as np
from loguru import logger
from numpy.typing import NDArray

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
        choices=["cpu", "cuda", "coreml"],
        default="cpu",
        help="Device to run the models on. cuda needs the gpu dependency group, "
        "and coreml needs macOS.",
    )
    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    if not (args.point_prompt or args.box_prompt):
        parser.error("one of --point-prompt, --box-prompt is required")
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

    names = ["image_encoder", *MODEL_NAMES]
    sessions = dict(zip(names, create_sessions(device=args.device, names=names)))

    frames, fps = read_video(args.video)
    if not frames:
        logger.error("no frames in {}", args.video)
        sys.exit(1)
    if not 0 <= args.frame_idx < len(frames):
        logger.error("--frame-idx must be in [0, {})", len(frames))
        sys.exit(1)
    height, width = frames[0].shape[:2]
    logger.debug("video: {} frames of {}x{}", len(frames), width, height)

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    for frame_idx, frame in enumerate(frames):
        viz = imgviz.instances2rgb(
            image=frame, masks=masks[frame_idx], labels=[1], alpha=0.5
        )
        writer.write(np.ascontiguousarray(viz[:, :, ::-1]))
    writer.release()
    logger.info("saved: {}", args.output)


if __name__ == "__main__":
    main()
