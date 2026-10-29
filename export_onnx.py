#!/usr/bin/env python3

import argparse
import copy
import pathlib
import typing

import imgviz
import numpy as np
import onnx
import onnxruntime
import PIL.Image
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[unresolved-import]
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]
from torchvision.transforms import v2

from infer_onnx import postprocess_decoder_output
from infer_onnx import postprocess_point_decoder_output
from infer_onnx import run_point_decoder
from infer_torch import get_replace_freqs_cis
from postprocess_onnx import postprocess

MAX_BOXES: int = 16


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

        # Features for the point and mask decoder, as in Sam3Processor.set_image
        # and Sam3Image.predict_inst.
        sam2_fpn = backbone_out.pop("sam2_backbone_out")["backbone_fpn"]
        tracker = self._processor.model.inst_interactive_predictor.model
        high_res_features1 = tracker.sam_mask_decoder.conv_s0(sam2_fpn[0])
        high_res_features2 = tracker.sam_mask_decoder.conv_s1(sam2_fpn[1])
        image_embeddings = sam2_fpn[2] + tracker.no_mem_embed.view(1, -1, 1, 1)

        assert len(backbone_out["vision_pos_enc"]) == 3
        assert len(backbone_out["backbone_fpn"]) == 3
        return (
            *backbone_out["vision_pos_enc"],
            *backbone_out["backbone_fpn"],
            image_embeddings,
            high_res_features1,
            high_res_features2,
        )


def _postprocess(onnx_file: pathlib.Path) -> None:
    postprocess(onnx_file)
    # torch writes this file for this model alone, and nothing reads it now.
    (onnx_file.parent / f"{onnx_file.name}.data").unlink(missing_ok=True)


def _export_image_encoder(
    processor: Sam3Processor, image: PIL.Image.Image
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    image = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)

    onnx_file: pathlib.Path = pathlib.Path("models/sam3_image_encoder.onnx")
    # v0.3.0 and earlier encoders lack the outputs of the point decoder.
    if onnx_file.exists() and "image_embeddings" in [
        o.name for o in onnx.load(onnx_file, load_external_data=False).graph.output
    ]:
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
                "image_embeddings",
                "high_res_features1",
                "high_res_features2",
            ],
            opset_version=21,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))
        _postprocess(onnx_file)

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(onnx_file)
    output = session.run(None, {"image": np.asarray(image).transpose(2, 0, 1)})
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("finished onnx runtime inference")

    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:6]
    sam2_features: list[NDArray] = output[6:]
    return vision_pos_enc, backbone_fpn, sam2_features


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
        _postprocess(onnx_file)

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
    # v0.4.0 and earlier decoders take one box.
    if onnx_file.exists() and [
        i.type.tensor_type.shape.dim[0].dim_value
        for i in onnx.load(onnx_file, load_external_data=False).graph.input
        if i.name == "box_coords"
    ] == [len(box_coords)]:
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
        _postprocess(onnx_file)

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


class _PointDecoder(torch.nn.Module):
    def __init__(self, model: Sam3Image) -> None:
        super().__init__()
        tracker = model.inst_interactive_predictor.model
        # On CUDA, tracing torch.repeat_interleave in the mask decoder puts its
        # index on the CPU, so export this small decoder on the CPU.
        self._prompt_encoder = copy.deepcopy(tracker.sam_prompt_encoder).cpu()
        self._mask_decoder = copy.deepcopy(tracker.sam_mask_decoder).cpu()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_features1: torch.Tensor,
        high_res_features2: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # SAM3InteractiveImagePredictor._predict with multimask_output=True
        prompt_encoder = self._prompt_encoder
        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        dense_embeddings = (
            has_mask_input * prompt_encoder._embed_masks(mask_input)
            + (1 - has_mask_input) * dense_embeddings
        )
        low_res_masks, iou_predictions, _, _ = self._mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        # As SAM3InteractiveImagePredictor._predict, so they can be fed back.
        return iou_predictions, low_res_masks.clamp(-32.0, 32.0)


def _export_point_decoder(
    model: Sam3Image, sam2_features: list[NDArray]
) -> onnxruntime.InferenceSession:
    onnx_file: pathlib.Path = pathlib.Path("models/sam3_point_decoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        inputs: dict[str, NDArray] = {
            "image_embeddings": sam2_features[0],
            "high_res_features1": sam2_features[1],
            "high_res_features2": sam2_features[2],
            "point_coords": np.full((1, 1, 2), 504, dtype=np.float32),
            "point_labels": np.ones((1, 1), dtype=np.float32),
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        }
        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            _PointDecoder(model=model),
            args=tuple(torch.tensor(v) for v in inputs.values()),
            f=onnx_file,
            input_names=list(inputs),
            output_names=["iou_predictions", "low_res_masks"],
            dynamic_axes={
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            },
            opset_version=21,
            dynamo=False,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))
        _postprocess(onnx_file)

    return onnxruntime.InferenceSession(onnx_file)


def _export_litetext_language_encoder(
    source: pathlib.Path, variant: str, checkpoint: pathlib.Path | None
) -> None:
    from litetext import load_litetext_encoder

    onnx_file = pathlib.Path(f"models/sam3_litetext_{variant}_language_encoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
        return
    encoder = load_litetext_encoder(source, variant, checkpoint)
    tokens = torch.from_numpy(tokenize(texts=["person"], context_length=16))
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        args=(tokens,),
        f=onnx_file,
        input_names=["tokens"],
        output_names=["text_attention_mask", "text_memory", "text_embeds"],
        opset_version=21,
        verify=True,
    )
    _postprocess(onnx_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--litetext", choices=["s0", "s1", "l"])
    parser.add_argument("--litetext-source", type=pathlib.Path)
    parser.add_argument("--litetext-checkpoint", type=pathlib.Path)
    args = parser.parse_args()
    if args.litetext:
        if args.litetext_source is None:
            parser.error("--litetext requires --litetext-source")
        _export_litetext_language_encoder(
            args.litetext_source, args.litetext, args.litetext_checkpoint
        )
        return
    if args.litetext_source or args.litetext_checkpoint:
        parser.error("--litetext-source and --litetext-checkpoint require --litetext")

    model: Sam3Image = build_sam3_image_model(enable_inst_interactivity=True)
    # The tracker enters bfloat16 autocast for the whole process on construction,
    # which puts bfloat16 Convs into the exported model.
    model.inst_interactive_predictor.model.bf16_context.__exit__(None, None, None)
    get_replace_freqs_cis(model)
    processor: Sam3Processor = Sam3Processor(model)

    image: PIL.Image.Image = PIL.Image.open("images/bus.jpg")
    # image: PIL.Image.Image = PIL.Image.open("2011_000006.jpg")

    # image_encoder {{
    # state = processor.set_image(image)

    vision_pos_enc, backbone_fpn, sam2_features = _export_image_encoder(
        processor, image
    )
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

    # A fixed number of box slots, as tracing bakes the box count into the model.
    # Unused slots are masked out as padding, which SAM3 batches the same way.
    box_coords = np.zeros((MAX_BOXES, 1, 4), dtype=np.float32)
    box_labels = np.ones((MAX_BOXES, 1), dtype=np.int64)
    box_masks = np.ones((1, MAX_BOXES), dtype=np.bool_)

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

    # point_decoder {{
    # state = processor.set_image(image)
    # model.predict_inst(state, point_coords=..., point_labels=..., mask_input=...)

    best_box: NDArray = boxes[scores.argmax()]
    point: NDArray = (best_box[:2] + best_box[2:]) / 2
    iou_predictions, low_res_masks = run_point_decoder(
        sess_point_decode=_export_point_decoder(model, sam2_features),
        sam2_features=sam2_features,
        point_prompt=[[point[0] / image.width, point[1] / image.height, 1]],
    )
    # }}

    state = processor.set_image(image)
    kwargs = dict(
        point_coords=point[None], point_labels=np.array([1]), multimask_output=True
    )
    _, iou_torch, low_res_torch = model.predict_inst(state, **kwargs)
    _, iou_torch, low_res_torch = model.predict_inst(
        state, mask_input=low_res_torch[[iou_torch.argmax()]], **kwargs
    )
    logger.debug(
        "point decoder vs pytorch: iou_predictions {} vs {}, mask agreement {:.3f}",
        iou_predictions[0].round(3),
        iou_torch.round(3),
        ((low_res_masks[0] > 0) == (low_res_torch > 0)).mean(),
    )

    mask, score = postprocess_point_decoder_output(
        iou_predictions=iou_predictions,
        low_res_masks=low_res_masks,
        image_width=image.width,
        image_height=image.height,
    )
    viz_point = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=mask[None],
        labels=[1],
        captions=[f"{score:.2f}"],
    )
    imgviz.io.imshow(imgviz.tile([viz, viz_point], row=1, col=2))


if __name__ == "__main__":
    main()
