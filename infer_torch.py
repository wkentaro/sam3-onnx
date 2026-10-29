#!/usr/bin/env python3

import imgviz
import numpy as np
import PIL.Image
import torch
from loguru import logger

from infer_onnx import parse_args


def get_replace_freqs_cis(module):
    if hasattr(module, "freqs_cis"):
        freqs_cos = module.freqs_cis.real.float()
        freqs_sin = module.freqs_cis.imag.float()
        # Replace the buffer
        module.register_buffer("freqs_cos", freqs_cos)
        module.register_buffer("freqs_sin", freqs_sin)
        del module.freqs_cis  # Remove complex version
    for child in module.children():
        get_replace_freqs_cis(child)


def main() -> None:
    args = parse_args()

    # XXX: those imports has to be after cv2.selectROI to avoid segfault
    from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
    from sam3.model.sam3_image_processor import (  # type: ignore[unresolved-import]
        Sam3Processor,
    )
    from sam3.model_builder import (  # type: ignore[unresolved-import]
        build_sam3_image_model,
    )

    model: Sam3Image = build_sam3_image_model(
        enable_inst_interactivity=bool(args.point_prompt)
    )

    if 0:
        with torch.no_grad():
            get_replace_freqs_cis(model)

    processor: Sam3Processor = Sam3Processor(model)

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")
    state = processor.set_image(image)

    if args.point_prompt:
        points = np.array(args.point_prompt)
        kwargs = dict(
            point_coords=points[:, :2] * [image.width, image.height],
            point_labels=points[:, 2],
            multimask_output=True,
        )
        _, scores, low_res_masks = model.predict_inst(state, **kwargs)
        # Refine with the best mask from the first pass as the mask prompt.
        masks, scores, _ = model.predict_inst(
            state, mask_input=low_res_masks[[scores.argmax()]], **kwargs
        )
        best = scores.argmax()
        viz = imgviz.instances2rgb(
            image=np.asarray(image),
            masks=masks[[best]].astype(bool),
            labels=[1],
            captions=[f"point: {scores[best]:.0%}"],
        )
        imgviz.io.imshow(viz)
        return

    if args.text_prompt:
        output = processor.set_text_prompt(prompt=args.text_prompt, state=state)
    for box in args.box_prompt or []:
        output = processor.add_geometric_prompt(
            box=box[:4],
            label=bool(box[4]),
            state=state,
        )

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    logger.debug(
        "output: {}",
        {"masks": masks.shape, "boxes": boxes.shape, "scores": scores.shape},
    )

    text_prompt: str = args.text_prompt if args.text_prompt else "visual"
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks.cpu().numpy()[:, 0, :, :],
        bboxes=boxes.cpu().numpy()[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
    )
    imgviz.io.imshow(viz)


if __name__ == "__main__":
    main()
