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
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--text-prompt",
        type=str,
        help="Text prompt for segmentation.",
    )
    prompt_group.add_argument(
        "--box-prompt",
        type=str,
        nargs="?",
        const="0,0,0,0",
        help="Box prompt for segmentation in format: cx cy w h (normalized).",
    )
    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    if args.box_prompt:
        args.box_prompt = [float(x) for x in args.box_prompt.split(",")]
        if len(args.box_prompt) != 4:
            logger.error("box_prompt must have 4 values: cx, cy, w, h")
            sys.exit(1)

    if args.box_prompt == [0, 0, 0, 0]:
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

        args.box_prompt = [
            (x + w / 2) / image.shape[1],
            (y + h / 2) / image.shape[0],
            w / image.shape[1],
            h / image.shape[0],
        ]
        logger.debug("box_prompt: {!r}", ",".join(f"{x:.3f}" for x in args.box_prompt))

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


def main():
    args = parse_args()

    sess_image = onnxruntime.InferenceSession("models/sam3_image_encoder.onnx")
    sess_language = onnxruntime.InferenceSession("models/sam3_language_encoder.onnx")
    sess_decode = onnxruntime.InferenceSession("models/sam3_decoder.onnx")

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")
    logger.debug("original image size: {}", image.size)

    logger.debug("running image encoder...")
    output = sess_image.run(
        None, {"image": np.asarray(image.resize((1008, 1008))).transpose(2, 0, 1)}
    )
    assert len(output) == 6
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:]
    logger.debug("finished running image encoder")

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
    box_coords: NDArray[np.float32] = np.array(
        args.box_prompt if args.box_prompt else [0, 0, 0, 0], dtype=np.float32
    ).reshape(1, 1, 4)
    box_labels: NDArray[np.int64] = np.array([[1]], dtype=np.int64)
    box_masks: NDArray[np.bool_] = np.array(
        [False] if args.box_prompt else [True], dtype=np.bool_
    ).reshape(1, 1)
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
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
