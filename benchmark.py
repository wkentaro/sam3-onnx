#!/usr/bin/env python3
"""Time the ONNX models and upstream's PyTorch models on the same work.

It runs every kind of prompt: text, box, and point on an image, then point,
box, and text on a video. It prints the mean time in ms of each model and of
the whole, per image or per tracked frame. --runtime torch runs upstream as
it ships: bfloat16 autocast on CUDA.

An image's time runs from the decoded image and the prompt to masks at the
image size. A frame's time runs from the JPEG files of the video and the
prompt to the masks of its last frame, over the frame count. So it holds the
same work in both runtimes: decoding and resizing the frames, which upstream
does in init_state with normalizing them and moving them to the GPU, and
which the ONNX image encoder does in part inside its run.

Each case runs once untimed with the same prompts, then again from a fresh
state.
"""

import argparse
import collections
import gc
import json
import pathlib
import tempfile
import time
import typing

import cv2
import numpy as np
import PIL.Image

from infer_onnx import DEVICES
from infer_onnx import FP16_HELP

TEXT: str = "person"
# On images/bus.jpg, normalized: a box (cx, cy, w, h) and a point on a person.
IMAGE_BOX: list[float] = [0.184, 0.601, 0.24, 0.466]
IMAGE_POINT: list[float] = [0.18, 0.55]
# On bedroom.mp4, in pixels of its 960x540 frames, from
# examples/sam3_for_sam2_video_task_example.ipynb: a point per object, and a
# box (x1, y1, x2, y2).
VIDEO_POINTS: list[tuple[int, int]] = [(210, 350), (400, 150), (460, 60)]
VIDEO_BOX: tuple[int, int, int, int] = (300, 0, 500, 400)

Timings = dict[str, list[float]]
Sync = typing.Callable[[], None]


def _timed(timings: Timings, name: str, fn: typing.Callable, sync: Sync):
    def wrapper(*args, **kwargs):
        sync()
        start = time.perf_counter()
        _nested.append(0.0)
        out = fn(*args, **kwargs)
        sync()
        elapsed = (time.perf_counter() - start) * 1000
        # Without the timed calls inside it, as upstream's detector runs the
        # image encoder within forward_grounding on video.
        timings[name].append(elapsed - _nested.pop())
        if _nested:
            _nested[-1] += elapsed
        return out

    return wrapper


# The time of the timed calls inside each timed call that is running.
_nested: list[float] = []


def _measure(
    make_steps: typing.Callable[[], typing.Iterable], timings: Timings, sync: Sync
) -> dict[str, float]:
    """The mean time per item of a whole iteration, and of each model in it.

    Over the whole and not between items, as a predictor can buffer frames
    before it yields them, and then yield several without running a model.
    The whole starts before make_steps, so it holds what a predictor prepares
    before its first item, like loading the frames of a video.
    """
    timings.clear()
    sync()
    start = time.perf_counter()
    count = sum(1 for _ in make_steps())
    sync()
    total = (time.perf_counter() - start) * 1000
    report = {name: sum(times) / count for name, times in sorted(timings.items())}
    return {
        name: round(ms, 1) for name, ms in {**report, "total": total / count}.items()
    }


def _measure_warm(
    make_steps: typing.Callable[[], typing.Iterable], timings: Timings, sync: Sync
) -> dict[str, float]:
    """_measure of a fresh run, after an untimed one of the same work, so every
    shape it feeds a model has run before."""
    for _ in make_steps():
        pass
    return _measure(make_steps, timings, sync)


def _repeat(run: typing.Callable[[], object], times: int) -> typing.Iterator[None]:
    for _ in range(times):
        run()
        yield


def _extract_frames(video: pathlib.Path, num_frames: int) -> list[pathlib.Path]:
    # Both runtimes read the same JPEGs.
    out_dir = pathlib.Path(tempfile.mkdtemp())
    cap = cv2.VideoCapture(str(video))
    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame)
    cap.release()
    return sorted(out_dir.glob("*.jpg"))


def run_onnx(args: argparse.Namespace, paths: list[pathlib.Path]) -> dict:
    from osam._models.yoloworld.clip import tokenize

    import track_onnx
    import video_onnx
    from infer_onnx import DECODER_FEATURES
    from infer_onnx import POINT_DECODER_FEATURES
    from infer_onnx import create_sessions
    from infer_onnx import make_box_inputs
    from infer_onnx import optional_names
    from infer_onnx import postprocess_decoder_output
    from infer_onnx import postprocess_point_decoder_output
    from infer_onnx import preprocess_image
    from infer_onnx import run_point_decoder

    names = [
        "image_encoder",
        *video_onnx.MODEL_NAMES,
        *optional_names(args.device, args.fp16),
    ]
    start = time.perf_counter()
    sessions = dict(
        zip(names, create_sessions(device=args.device, names=names, fp16=args.fp16))
    )
    result: dict[str, typing.Any] = {"load_s": round(time.perf_counter() - start, 1)}
    timings: Timings = collections.defaultdict(list)
    for name, session in sessions.items():
        session.run = _timed(timings, name, session.run, lambda: None)  # type: ignore[method-assign]

    def measure(make_steps: typing.Callable[[], typing.Iterable]) -> dict[str, float]:
        return _measure_warm(make_steps, timings, lambda: None)

    # Each run covers what Sam3Processor does: from the image and the prompt
    # to masks at the image size.
    image = PIL.Image.open(args.image).convert("RGB")
    size = {"image_width": image.width, "image_height": image.height}

    def detect(text: str, boxes: list[list[float]]) -> None:
        image_out = sessions["image_encoder"].run(
            DECODER_FEATURES, {"image": preprocess_image(image)}
        )
        text_out = sessions["language_encoder"].run(
            None, {"tokens": tokenize(texts=[text], context_length=32)}
        )
        boxes_out, _, masks = sessions["decoder"].run(
            ["boxes", "scores", "masks"],
            {
                **dict(zip(DECODER_FEATURES, image_out)),
                "language_mask": text_out[0],
                "language_features": text_out[1],
                **make_box_inputs(sessions["decoder"], boxes),
            },
        )
        postprocess_decoder_output(boxes=boxes_out, masks=masks, **size)

    def segment_point() -> None:
        features = sessions["image_encoder"].run(
            POINT_DECODER_FEATURES,
            {"image": preprocess_image(image)},
        )
        iou_predictions, low_res_masks = run_point_decoder(
            sess_point_decode=sessions["point_decoder"],
            sam2_features=features,
            point_prompt=[[*IMAGE_POINT, 1]],
        )
        postprocess_point_decoder_output(
            iou_predictions=iou_predictions, low_res_masks=low_res_masks, **size
        )

    image_cases: dict[str, typing.Callable[[], None]] = {
        "image: text": lambda: detect(TEXT, []),
        # A box alone goes with the text "visual", as in infer_onnx.py.
        "image: box": lambda: detect("visual", [[*IMAGE_BOX, 1]]),
        "image: point": segment_point,
    }
    for name, run in image_cases.items():
        result[name] = measure(lambda: _repeat(run, args.image_runs))

    width, height = PIL.Image.open(paths[0]).size

    def load_frames() -> list[np.ndarray]:
        # Inside the timing, as upstream's init_state is. The predictors
        # resize each frame with their own filter as they encode it.
        return [np.asarray(PIL.Image.open(path).convert("RGB")) for path in paths]

    def track(**prompts: typing.Any) -> typing.Iterator:
        frames = load_frames()
        tracker = track_onnx.Tracker(
            sessions,
            lambda i: track_onnx.encode_frame(sessions["image_encoder"], frames[i]),
            len(frames),
            height,
            width,
        )
        for obj_id, prompt in enumerate(_tracker_prompts(width, height, **prompts)):
            tracker.add_new_points_or_box(frame_idx=0, obj_id=obj_id, **prompt)
        return tracker.propagate_in_video()

    def track_text() -> typing.Iterator:
        frames = load_frames()
        video = video_onnx.VideoInference(
            sessions,
            lambda i: video_onnx.encode_frame(sessions["image_encoder"], frames[i]),
            len(frames),
            height,
            width,
        )
        video.add_prompt(0, text_str=TEXT)
        return video.propagate_in_video()

    for num_objects in (1, len(VIDEO_POINTS)):
        result[f"video: point, {num_objects} objects"] = measure(
            lambda: track(num_points=num_objects)
        )
    result["video: box"] = measure(lambda: track(box=True))
    result["video: text"] = measure(track_text)
    return result


def _tracker_prompts(
    width: int, height: int, num_points: int = 0, box: bool = False
) -> list[dict[str, typing.Any]]:
    """The arguments of add_new_points_or_box for each object, which both
    runtimes take alike."""
    prompts: list[dict[str, typing.Any]] = [
        {
            "points": np.array([[x / width, y / height]], np.float32),
            "labels": np.array([1], np.int32),
        }
        for x, y in VIDEO_POINTS[:num_points]
    ]
    if box:
        scale = [width, height, width, height]
        prompts.append({"box": np.array(VIDEO_BOX, np.float32) / scale})
    return prompts


def run_torch(args: argparse.Namespace, paths: list[pathlib.Path]) -> dict:
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[unresolved-import]
    from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]
    from sam3.model_builder import build_sam3_video_model  # type: ignore[unresolved-import]

    if not torch.cuda.is_available():
        raise SystemExit("--runtime torch needs CUDA, as upstream's predictors do")
    sync = torch.cuda.synchronize
    timings: Timings = collections.defaultdict(list)
    result: dict[str, typing.Any] = {}

    def wrap(obj: typing.Any, attr: str, name: str) -> None:
        setattr(obj, attr, _timed(timings, name, getattr(obj, attr), sync))

    def wrap_tracker(tracker: typing.Any) -> None:
        wrap(tracker.transformer.encoder, "forward", "memory_attention")
        wrap(tracker.sam_mask_decoder, "forward", "point_decoder")
        wrap(tracker.maskmem_backbone, "forward", "memory_encoder")

    def wrap_detector(detector: typing.Any) -> None:
        wrap(detector.backbone, "forward_image", "image_encoder")
        wrap(detector.backbone, "forward_text", "language_encoder")
        wrap(detector, "forward_grounding", "decoder")

    def measure(make_steps: typing.Callable[[], typing.Iterable]) -> dict[str, float]:
        return _measure_warm(make_steps, timings, sync)

    def run_image() -> float:
        start = time.perf_counter()
        image_model = build_sam3_image_model(enable_inst_interactivity=True)
        load_s = time.perf_counter() - start
        wrap_detector(image_model)
        wrap_tracker(image_model.inst_interactive_predictor.model)
        processor = Sam3Processor(image_model)
        image = PIL.Image.open(args.image).convert("RGB")

        def segment_point() -> None:
            state = processor.set_image(image)
            kwargs = dict(
                point_coords=np.array([IMAGE_POINT]) * [image.width, image.height],
                point_labels=np.array([1]),
                multimask_output=True,
            )
            _, scores, low_res_masks = image_model.predict_inst(state, **kwargs)
            # Refine with the best mask, as infer_torch.py does.
            image_model.predict_inst(
                state, mask_input=low_res_masks[[scores.argmax()]], **kwargs
            )

        image_cases: dict[str, typing.Callable[[], object]] = {
            "image: text": lambda: processor.set_text_prompt(
                prompt=TEXT, state=processor.set_image(image)
            ),
            "image: box": lambda: processor.add_geometric_prompt(
                box=IMAGE_BOX, label=True, state=processor.set_image(image)
            ),
            "image: point": segment_point,
        }
        for name, run in image_cases.items():
            result[name] = measure(lambda: _repeat(run, args.image_runs))
        return load_s

    with torch.inference_mode():
        load_s = run_image()
        # One model at a time, to fit a 12 GB GPU.
        gc.collect()
        torch.cuda.empty_cache()

        start = time.perf_counter()
        video_model = build_sam3_video_model()
        result = {"load_s": round(load_s + time.perf_counter() - start, 1), **result}
        wrap_detector(video_model.detector)
        wrap_tracker(video_model.tracker)

        def track_text() -> typing.Iterator:
            state = video_model.init_state(resource_path=str(paths[0].parent))
            video_model.add_prompt(state, 0, text_str=TEXT)
            return video_model.propagate_in_video(state)

        # Before the tracker below takes the detector's backbone.
        video_text = measure(track_text)

        predictor = video_model.tracker
        predictor.backbone = video_model.detector.backbone
        width, height = PIL.Image.open(paths[0]).size

        def track(**prompts: typing.Any) -> typing.Iterator:
            state = predictor.init_state(video_path=str(paths[0].parent))
            for obj_id, prompt in enumerate(_tracker_prompts(width, height, **prompts)):
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id, **prompt
                )
            return predictor.propagate_in_video(
                state,
                start_frame_idx=None,
                max_frame_num_to_track=None,
                reverse=False,
                propagate_preflight=True,
                tqdm_disable=True,
            )

        for num_objects in (1, len(VIDEO_POINTS)):
            result[f"video: point, {num_objects} objects"] = measure(
                lambda: track(num_points=num_objects)
            )
        result["video: box"] = measure(lambda: track(box=True))
        result["video: text"] = video_text
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runtime", choices=["onnx", "torch"], default="onnx")
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help="Device of --runtime onnx, as infer_onnx.py takes. --runtime torch "
        "runs on CUDA.",
    )
    parser.add_argument("--fp16", action="store_true", help=FP16_HELP)
    parser.add_argument("--image", type=pathlib.Path, default="images/bus.jpg")
    parser.add_argument("--image-runs", type=int, default=5)
    parser.add_argument(
        "--video",
        type=pathlib.Path,
        default=pathlib.Path("sam3/assets/videos/bedroom.mp4"),
    )
    parser.add_argument("--num-frames", type=int, default=40)
    args = parser.parse_args()

    paths = _extract_frames(args.video, args.num_frames)
    run = run_torch if args.runtime == "torch" else run_onnx
    result: dict[str, typing.Any] = {"runtime": args.runtime}
    if args.runtime == "onnx":
        result.update(device=args.device, fp16=args.fp16)
    elif args.fp16:
        parser.error("--fp16 is for --runtime onnx")
    result.update(run(args, paths))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
