#!/usr/bin/env python3

import pathlib
import typing

import imgviz
import numpy as np
import onnxruntime
import PIL.Image
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from torchvision.transforms import v2

from infer_onnx import postprocess_decoder_output
from infer_torch import get_replace_freqs_cis
from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import (  # type: ignore[unresolved-import]
    Sam3Processor,
)
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]


class _ImageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor: Sam3Processor = processor
        self._transform = v2.Compose(
            [
                # NOTE: Resize in .transform has difference between pytorch and onnx
                # v2.ToDtype(torch.uint8, scale=True),
                # v2.Resize(size=(resolution := 1008, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        image = self._transform(image).unsqueeze(0)

        backbone_out = self._processor.model.backbone._forward_image_no_act_ckpt(image)
        del backbone_out["vision_features"]
        del backbone_out["sam2_backbone_out"]

        assert len(backbone_out["vision_pos_enc"]) == 3
        assert len(backbone_out["backbone_fpn"]) == 3
        return *backbone_out["vision_pos_enc"], *backbone_out["backbone_fpn"]


def _export_image_encoder(
    processor: Sam3Processor, image: PIL.Image.Image
) -> tuple[list[NDArray], list[NDArray]]:
    image = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)

    onnx_file: pathlib.Path = pathlib.Path("models/sam3_image_encoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        encoder: _ImageEncoder = _ImageEncoder(processor=processor)
        input_image: torch.Tensor = v2.functional.to_image(image).to("cuda")

        # with torch.no_grad():
        #     output = image_backbone(input_image)

        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            encoder,
            args=(input_image,),
            f=onnx_file,
            input_names=["image"],
            output_names=[
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
            ],
            opset_version=21,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(onnx_file)
    output = session.run(None, {"image": np.asarray(image).transpose(2, 0, 1)})
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("finished onnx runtime inference")

    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:]
    return vision_pos_enc, backbone_fpn


class _LanguageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor: Sam3Processor = processor

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model: Sam3Image = self._processor.model

        # VETextEncoder.forward
        text_attention_mask = (tokens != 0).bool()
        #
        inputs_embeds = model.backbone.language_backbone.encoder.token_embedding(tokens)
        _, text_memory = model.backbone.language_backbone.encoder(tokens)
        #
        assert text_memory.shape[1] == inputs_embeds.shape[1]
        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = model.backbone.language_backbone.resizer(text_memory)
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)


def _export_language_encoder(processor: Sam3Processor) -> list[NDArray]:
    tokens = tokenize(texts=["person"], context_length=32)

    onnx_file: pathlib.Path = pathlib.Path("models/sam3_language_encoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        encoder: _LanguageEncoder = _LanguageEncoder(processor=processor)
        tokens_input: torch.Tensor = torch.from_numpy(tokens).to("cuda")

        # with torch.no_grad():
        #     output = encoder(tokens=tokens_input)

        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            encoder,
            args=(tokens_input,),
            f=onnx_file,
            input_names=["tokens"],
            output_names=["text_attention_mask", "text_memory", "text_embeds"],
            opset_version=21,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(onnx_file)
    output = session.run(None, {"tokens": tokens})
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("finished onnx runtime inference")

    return output


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model: Sam3Image = build_sam3_image_model()
        self._processor: Sam3Processor = Sam3Processor(self._model)

    def forward(
        self,
        vision_pos_enc_0: torch.Tensor,
        vision_pos_enc_1: torch.Tensor,
        vision_pos_enc_2: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        language_mask: torch.Tensor,
        language_features: torch.Tensor,
        language_embeds: torch.Tensor,
        box_coords: torch.Tensor,
        box_labels: torch.Tensor,
        box_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geometric_prompt = self._processor.model._get_dummy_prompt()
        geometric_prompt.box_embeddings = box_coords
        geometric_prompt.box_labels = box_labels
        geometric_prompt.box_mask = box_masks
        backbone_out = {
            "vision_pos_enc": [
                vision_pos_enc_0,
                vision_pos_enc_1,
                vision_pos_enc_2,
            ],
            "backbone_fpn": [
                backbone_fpn_0,
                backbone_fpn_1,
                backbone_fpn_2,
            ],
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
        }

        outputs = self._model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self._processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        out_bbox = out_bbox[out_probs > self._processor.confidence_threshold]
        out_masks = out_masks[out_probs > self._processor.confidence_threshold]
        out_probs = out_probs[out_probs > self._processor.confidence_threshold]

        # normalized xyxy boxes
        x_c, y_c, w, h = out_bbox.unbind(-1)
        boxes = torch.stack(
            [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1
        )

        # masks at model native resolution (no interpolation to original image size)
        masks = out_masks.unsqueeze(1).sigmoid()

        return boxes, out_probs, masks


def _export_decoder(
    vision_pos_enc_0: NDArray,
    vision_pos_enc_1: NDArray,
    vision_pos_enc_2: NDArray,
    backbone_fpn_0: NDArray,
    backbone_fpn_1: NDArray,
    backbone_fpn_2: NDArray,
    language_mask: NDArray,
    language_features: NDArray,
    language_embeds: NDArray,
    box_coords: NDArray,
    box_labels: NDArray,
    box_masks: NDArray,
) -> list[NDArray]:
    onnx_file: pathlib.Path = pathlib.Path("models/sam3_decoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        decoder: _Decoder = _Decoder()

        torch.onnx.export(
            decoder,
            args=(
                torch.tensor(vision_pos_enc_0).to("cuda"),
                torch.tensor(vision_pos_enc_1).to("cuda"),
                torch.tensor(vision_pos_enc_2).to("cuda"),
                torch.tensor(backbone_fpn_0).to("cuda"),
                torch.tensor(backbone_fpn_1).to("cuda"),
                torch.tensor(backbone_fpn_2).to("cuda"),
                torch.tensor(language_mask).to("cuda"),
                torch.tensor(language_features).to("cuda"),
                torch.tensor(language_embeds).to("cuda"),
                torch.tensor(box_coords).to("cuda"),
                torch.tensor(box_labels).to("cuda"),
                torch.tensor(box_masks).to("cuda"),
            ),
            f=onnx_file,
            input_names=[
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "language_mask",
                "language_features",
                "language_embeds",
                "box_coords",
                "box_labels",
                "box_masks",
            ],
            output_names=["boxes", "scores", "masks"],
            opset_version=21,
            dynamo=False,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))

    session = onnxruntime.InferenceSession(onnx_file)
    output = session.run(
        None,
        {
            "vision_pos_enc_2": vision_pos_enc_2,
            "backbone_fpn_0": backbone_fpn_0,
            "backbone_fpn_1": backbone_fpn_1,
            "backbone_fpn_2": backbone_fpn_2,
            "language_mask": language_mask,
            "language_features": language_features,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        },
    )
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("decoder onnx runtime inference done")

    return output


def main():
    model: Sam3Image = build_sam3_image_model()
    get_replace_freqs_cis(model)
    processor: Sam3Processor = Sam3Processor(model)

    image: PIL.Image.Image = PIL.Image.open("images/bus.jpg")
    # image: PIL.Image.Image = PIL.Image.open("2011_000006.jpg")

    # image_encoder {{
    # state = processor.set_image(image)

    vision_pos_enc, backbone_fpn = _export_image_encoder(processor, image)
    # }}

    # language_encoder {{
    # result = processor.set_text_prompt(prompt="person", state=state)

    language_mask, language_features, language_embeds = _export_language_encoder(
        processor=processor
    )
    # }}

    # decoder {{
    # result = processor._forward_grounding(state)
    # boxes, scores, masks = result["boxes"], result["scores"], result["masks"]

    box_coords = np.array([[[0.1620, 0.4010, 0.0640, 0.0180]]], dtype=np.float32)
    box_labels = np.array([[1]], dtype=np.int64)
    box_masks = np.array([[True]], dtype=np.bool_)

    boxes, scores, masks = _export_decoder(
        vision_pos_enc_0=vision_pos_enc[0],
        vision_pos_enc_1=vision_pos_enc[1],
        vision_pos_enc_2=vision_pos_enc[2],
        backbone_fpn_0=backbone_fpn[0],
        backbone_fpn_1=backbone_fpn[1],
        backbone_fpn_2=backbone_fpn[2],
        language_mask=language_mask,
        language_features=language_features,
        language_embeds=language_embeds,
        box_coords=box_coords,
        box_labels=box_labels,
        box_masks=box_masks,
    )
    # }}

    boxes, masks = postprocess_decoder_output(
        boxes=boxes,
        masks=masks,
        image_width=image.width,
        image_height=image.height,
    )

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks,
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(boxes)) + 1,
        captions=[f"{s:.2f}" for s in scores],
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
