#!/usr/bin/env python3

import argparse
import pathlib
import sys
import typing

import cv2
import imgviz
import numpy as np
import onnxruntime
import PIL.Image
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from osam.types._model import _setup_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        help="Path to the input image.",
        required=True,
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        help="Text prompt for segmentation.",
    )
    parser.add_argument(
        "--box-prompt",
        type=str,
        action="append",
        nargs="?",
        const="0,0,0,0",
        help="Box prompt of an example object in format: cx,cy,w,h[,label] "
        "(normalized, label 1 for positive and 0 for negative). Repeatable, and "
        "combinable with --text-prompt.",
    )
    parser.add_argument(
        "--point-prompt",
        type=str,
        action="append",
        help="Point prompt for segmenting one object in format: x,y[,label] "
        "(normalized, label 1 for foreground and 0 for background). Repeatable.",
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

    if args.point_prompt and (args.text_prompt or args.box_prompt):
        parser.error("--point-prompt can't be combined with other prompts")
    if not (args.text_prompt or args.box_prompt or args.point_prompt):
        parser.error("one of --text-prompt, --box-prompt, --point-prompt is required")

    if args.box_prompt and len(args.box_prompt) > 1 and "0,0,0,0" in args.box_prompt:
        parser.error("--box-prompt without a value can't be repeated")

    if args.box_prompt:
        boxes = [[float(x) for x in b.split(",")] for b in args.box_prompt]
        if any(len(b) not in (4, 5) for b in boxes):
            logger.error("box_prompt must have 4 or 5 values: cx, cy, w, h[, label]")
            sys.exit(1)
        if any(len(b) == 5 and b[4] not in (0, 1) for b in boxes):
            logger.error("box_prompt label must be 0 or 1")
            sys.exit(1)
        args.box_prompt = [b if len(b) == 5 else [*b, 1] for b in boxes]

    if args.point_prompt:
        points = [[float(x) for x in p.split(",")] for p in args.point_prompt]
        if any(len(p) not in (2, 3) for p in points):
            logger.error("point_prompt must have 2 or 3 values: x, y[, label]")
            sys.exit(1)
        args.point_prompt = [p if len(p) == 3 else [*p, 1] for p in points]

    if args.box_prompt == [[0, 0, 0, 0, 1]]:
        image: NDArray[np.uint8] = imgviz.asrgb(imgviz.io.imread(args.image))

        logger.info("please select box prompt in the image window")
        x, y, w, h = cv2.selectROI(
            "Select box prompt and press ENTER or SPACE",
            image[:, :, ::-1],
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()
        if [x, y, w, h] == [0, 0, 0, 0]:
            logger.warning("no box prompt selected, exiting")
            sys.exit(1)

        box = [
            (x + w / 2) / image.shape[1],
            (y + h / 2) / image.shape[0],
            w / image.shape[1],
            h / image.shape[0],
        ]
        logger.debug("box_prompt: {!r}", ",".join(f"{x:.3f}" for x in box))
        args.box_prompt = [[*box, 1]]

    return args


def postprocess_decoder_output(
    boxes: NDArray,
    masks: NDArray,
    image_width: int,
    image_height: int,
) -> tuple[NDArray, NDArray]:
    boxes = boxes * np.array(
        [image_width, image_height, image_width, image_height], dtype=np.float32
    )
    if len(masks) == 0:
        return boxes, np.empty((0, image_height, image_width), dtype=bool)
    masks = np.array(
        [
            cv2.resize(
                mask[0],
                dsize=(image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )
            > 0.5
            for mask in masks
        ]
    )
    return boxes, masks


def postprocess_point_decoder_output(
    iou_predictions: NDArray,
    low_res_masks: NDArray,
    image_width: int,
    image_height: int,
) -> tuple[NDArray, float]:
    best = iou_predictions[0].argmax()
    mask_logits: NDArray[np.float32] = low_res_masks[0, best]
    # Fill background holes up to 256 px, as SAM2Transforms.postprocess_masks.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_logits <= 0).astype(np.uint8), connectivity=8
    )
    is_hole = (labels > 0) & (stats[labels, cv2.CC_STAT_AREA] <= 256)
    mask_logits = np.where(is_hole, np.float32(10), mask_logits)
    mask = (
        cv2.resize(
            mask_logits,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        > 0
    )
    return mask, float(iou_predictions[0, best])


def run_point_decoder(
    sess_point_decode: onnxruntime.InferenceSession,
    sam2_features: list[NDArray],
    point_prompt: list[list[float]],
) -> tuple[NDArray, NDArray]:
    points = np.array(point_prompt, dtype=np.float32)
    inputs: dict[str, NDArray] = {
        "image_embeddings": sam2_features[0],
        "high_res_features1": sam2_features[1],
        "high_res_features2": sam2_features[2],
        "point_coords": points[None, :, :2] * 1008,
        "point_labels": points[None, :, 2],
        "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
    }
    iou_predictions, low_res_masks = typing.cast(
        list[NDArray], sess_point_decode.run(None, inputs)
    )
    # Refine with the best mask from the first pass as the mask prompt.
    inputs["mask_input"] = low_res_masks[:, [iou_predictions[0].argmax()]]
    inputs["has_mask_input"] = np.array([1], dtype=np.float32)
    iou_predictions, low_res_masks = typing.cast(
        list[NDArray], sess_point_decode.run(None, inputs)
    )
    return iou_predictions, low_res_masks


def create_sessions(
    device: str, names: list[str]
) -> list[onnxruntime.InferenceSession]:
    providers: list[str] = ["CPUExecutionProvider"]
    coreml_options: dict[str, str] | None = None
    sess_options = onnxruntime.SessionOptions()  # type: ignore[possibly-missing-attribute]
    if device == "cuda":
        # Loads the CUDA libraries of osam[gpu] and shares one CUDA arena across
        # sessions, without which the three sessions OOM on a 12 GB GPU.
        if not _setup_cuda():
            logger.error(
                "CUDA is unavailable, run with: UV_PROJECT_ENVIRONMENT=.venv-gpu "
                "uv run --no-dev --no-group cpu --group gpu infer_onnx.py ..."
            )
            sys.exit(1)
        providers.insert(0, "CUDAExecutionProvider")
        sess_options.add_session_config_entry("session.use_env_allocators", "1")
    elif device == "coreml":
        available = onnxruntime.get_available_providers()  # type: ignore[possibly-missing-attribute]
        if "CoreMLExecutionProvider" not in available:
            logger.error("CoreML is unavailable, it needs onnxruntime on macOS")
            sys.exit(1)
        # onnxruntime's CoreML EP can't read external data.
        if pathlib.Path("models/sam3_image_encoder.onnx.data").exists():
            logger.error(
                "CoreML needs models with inline weights, run: "
                "uv run postprocess_onnx.py models/*.onnx && rm models/*.onnx.data"
            )
            sys.exit(1)
        # MLProgram supports more operators than the default NeuralNetwork, and
        # leaving out the Neural Engine keeps the GPU's accuracy and speed.
        # ponytail: no ModelCacheDirectory, which saves 3 s of the compile but
        # stores another 3.5 GB keyed by model path; add it for long sessions.
        coreml_options = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}
    sessions: list[onnxruntime.InferenceSession] = []
    for name in names:
        # Only the image encoder: the decoders take longer to compile for CoreML
        # than it saves them.
        session_providers: list[str | tuple[str, dict[str, str]]] = [*providers]
        if coreml_options is not None and name == "image_encoder":
            session_providers.insert(0, ("CoreMLExecutionProvider", coreml_options))
        session = onnxruntime.InferenceSession(
            f"models/sam3_{name}.onnx", sess_options, providers=session_providers
        )
        # onnxruntime silently falls back to CPU for a provider it can't use.
        (provider, *_) = session_providers
        provider = provider if isinstance(provider, str) else provider[0]
        if provider not in session.get_providers():
            logger.error("{} is not running on {}", name, provider)
            sys.exit(1)
        sessions.append(session)
    return sessions


def main():
    args = parse_args()

    decoder_names: list[str] = (
        ["point_decoder"] if args.point_prompt else ["language_encoder", "decoder"]
    )
    sess_image, *sess_decoders = create_sessions(
        device=args.device, names=["image_encoder", *decoder_names]
    )
    logger.debug("providers: {}", sess_image.get_providers())

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")
    logger.debug("original image size: {}", image.size)

    logger.debug("running image encoder...")
    output_names: list[str] = (
        ["image_embeddings", "high_res_features1", "high_res_features2"]
        if args.point_prompt
        else [f"vision_pos_enc_{i}" for i in range(3)]
        + [f"backbone_fpn_{i}" for i in range(3)]
    )
    output = sess_image.run(
        output_names,
        {"image": np.asarray(image.resize((1008, 1008))).transpose(2, 0, 1)},
    )
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("finished running image encoder")

    if args.point_prompt:
        logger.debug("running point decoder...")
        iou_predictions, low_res_masks = run_point_decoder(
            sess_point_decode=sess_decoders[0],
            sam2_features=output,
            point_prompt=args.point_prompt,
        )
        mask, score = postprocess_point_decoder_output(
            iou_predictions=iou_predictions,
            low_res_masks=low_res_masks,
            image_width=image.width,
            image_height=image.height,
        )
        logger.debug("output: {}", {"mask": mask.shape, "score": score})
        viz = imgviz.instances2rgb(
            image=np.asarray(image),
            masks=mask[None],
            labels=[1],
            captions=[f"point: {score:.0%}"],
            font_size=max(1, min(image.size) // 40),
        )
        imgviz.io.imshow(viz)
        return

    sess_language, sess_decode = sess_decoders
    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:]

    logger.debug("running language encoder...")
    text_prompt: str = args.text_prompt if args.text_prompt else "visual"
    output = sess_language.run(
        None, {"tokens": tokenize(texts=[text_prompt], context_length=32)}
    )
    assert len(output) == 3
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    language_mask: NDArray = output[0]
    language_features: NDArray = output[1]
    # language_embeds: NDArray = output[2]
    logger.debug("finished running language encoder")

    logger.debug("running decoder...")
    # The decoder takes a fixed number of boxes, and masks out the unused ones.
    box_prompt: list[list[float]] = args.box_prompt or []
    (box_input,) = [i for i in sess_decode.get_inputs() if i.name == "box_coords"]
    max_boxes: int = box_input.shape[0]
    if len(box_prompt) > max_boxes:
        logger.error("box_prompt must have at most {} boxes", max_boxes)
        sys.exit(1)
    box_coords: NDArray[np.float32] = np.zeros((max_boxes, 1, 4), dtype=np.float32)
    box_labels: NDArray[np.int64] = np.ones((max_boxes, 1), dtype=np.int64)
    box_masks: NDArray[np.bool_] = np.ones((1, max_boxes), dtype=np.bool_)
    for i, box in enumerate(box_prompt):
        box_coords[i, 0] = box[:4]
        box_labels[i, 0] = box[4]
        box_masks[0, i] = False
    output = sess_decode.run(
        None,
        {
            "backbone_fpn_0": backbone_fpn[0],
            "backbone_fpn_1": backbone_fpn[1],
            "backbone_fpn_2": backbone_fpn[2],
            "vision_pos_enc_2": vision_pos_enc[2],
            "language_mask": language_mask,
            "language_features": language_features,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        },
    )
    assert len(output) == 3
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    boxes: NDArray = output[0]
    scores: NDArray = output[1]
    masks: NDArray = output[2]
    logger.debug("finished running decoder")

    boxes, masks = postprocess_decoder_output(
        boxes=boxes,
        masks=masks,
        image_width=image.width,
        image_height=image.height,
    )

    logger.debug(
        "output: {}",
        {"masks": masks.shape, "boxes": boxes.shape, "scores": scores.shape},
    )

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks,
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
        font_size=max(1, min(image.size) // 40),
    )
    imgviz.io.imshow(viz)


if __name__ == "__main__":
    main()
