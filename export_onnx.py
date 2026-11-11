#!/usr/bin/env python3

import argparse
import copy
import pathlib
import shutil
import typing

import imgviz
import numpy as np
import onnx
import onnxruntime
import PIL.Image
import sam3.model.vitdet  # type: ignore[unresolved-import]
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[unresolved-import]
from sam3.model.sam3_tracker_utils import get_1d_sine_pe  # type: ignore[unresolved-import]
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]
from torchvision.transforms import v2

from infer_onnx import postprocess_decoder_output
from infer_onnx import postprocess_point_decoder_output
from infer_onnx import preprocess_image
from infer_onnx import run_point_decoder
from infer_torch import get_replace_freqs_cis
from postprocess_onnx import fuse_attention
from postprocess_onnx import postprocess
from track_onnx import FIXED_NUM_MEMORIES
from track_onnx import FIXED_NUM_OBJ_PTRS

MAX_BOXES: int = 16


def _rope_image_encoder(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """sam3.model.vitdet.apply_rotary_enc2 without its split into real and
    imaginary halves, which takes onnxruntime longer than the rotation.

    A channel pair (a, b) rotates to (a cos - b sin, b cos + a sin), which is
    x * (cos, cos) + (b, a) * (-sin, sin): one Gather swaps each pair.
    """
    assert not repeat_freqs_k
    cos = freqs_cos.repeat_interleave(2, dim=-1)
    sin = torch.stack([-freqs_sin, freqs_sin], dim=-1).flatten(-2)
    swap = torch.arange(xq.shape[-1], device=xq.device) ^ 1
    xq, xk = (x * cos + torch.index_select(x, -1, swap) * sin for x in (xq, xk))
    return xq, xk


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
    processor: Sam3Processor, image: PIL.Image.Image, fused: bool = False
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """With fused, the copy that CUDA and WebGPU run: its RoPE and attention
    take them half the time, but CoreML 6 minutes to compile."""
    pixels = preprocess_image(image)

    onnx_file = pathlib.Path(
        f"models/sam3_image_encoder{'_fused' if fused else ''}.onnx"
    )
    # v0.3.0 and earlier encoders lack the outputs of the point decoder.
    if onnx_file.exists() and "image_embeddings" in [
        o.name for o in onnx.load(onnx_file, load_external_data=False).graph.output
    ]:
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        encoder: _ImageEncoder = _ImageEncoder(processor=processor)
        input_image: torch.Tensor = torch.from_numpy(pixels.copy()).to("cuda")

        # with torch.no_grad():
        #     output = image_backbone(input_image)

        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        apply_rotary_enc2 = sam3.model.vitdet.apply_rotary_enc2
        if fused:
            sam3.model.vitdet.apply_rotary_enc2 = _rope_image_encoder
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
        sam3.model.vitdet.apply_rotary_enc2 = apply_rotary_enc2
        logger.debug("exported onnx model: {!r}", str(onnx_file))
        _postprocess(onnx_file)

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(onnx_file)
    output = session.run(None, {"image": pixels})
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        mask_logits = out_masks.unsqueeze(1)

        # The video predictor resizes and fills holes in the logits.
        return boxes, out_probs, mask_logits.sigmoid(), mask_logits


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
    graph = (
        onnx.load(onnx_file, load_external_data=False).graph
        if onnx_file.exists()
        else None
    )
    # v0.4.0 and earlier decoders take one box, and v0.6.0 and earlier lack
    # the mask logits for video.
    if (
        graph is not None
        and [
            i.type.tensor_type.shape.dim[0].dim_value
            for i in graph.input
            if i.name == "box_coords"
        ]
        == [len(box_coords)]
        and "mask_logits" in [o.name for o in graph.output]
    ):
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
            output_names=["boxes", "scores", "masks", "mask_logits"],
            opset_version=21,
            dynamo=False,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))
        _postprocess(onnx_file)

    session = onnxruntime.InferenceSession(onnx_file)
    output = session.run(
        ["boxes", "scores", "masks"],
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
        self._obj_ptr_proj = copy.deepcopy(tracker.obj_ptr_proj).cpu()
        self.register_buffer("_no_obj_ptr", tracker.no_obj_ptr.detach().cpu())

    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_features1: torch.Tensor,
        high_res_features2: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # SAM3InteractiveImagePredictor._predict with both values of multimask_output
        prompt_encoder = self._prompt_encoder
        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        dense_embeddings = (
            has_mask_input * prompt_encoder._embed_masks(mask_input)
            + (1 - has_mask_input) * dense_embeddings
        )
        # All 4 mask tokens, which MaskDecoder.forward splits by multimask_output.
        all_masks, all_iou, mask_tokens, object_score_logits = (
            self._mask_decoder.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                repeat_image=False,
                high_res_features=[high_res_features1, high_res_features2],
            )
        )
        # Sam3TrackerBase._forward_sam_heads, for every mask token: the tracker
        # takes the single mask's pointer, or the best of the multimask ones.
        is_obj_appearing = (object_score_logits > 0).float()[:, :, None]
        obj_ptrs = (
            is_obj_appearing * self._obj_ptr_proj(mask_tokens)
            + (1 - is_obj_appearing) * self._no_obj_ptr
        )
        # As SAM3InteractiveImagePredictor._predict, so they can be fed back.
        all_masks = all_masks.clamp(-32.0, 32.0)
        assert self._mask_decoder.dynamic_multimask_via_stability
        single_masks, single_iou = self._mask_decoder._dynamic_multimask_via_stability(
            all_masks, all_iou
        )
        return (
            all_iou[:, 1:],
            all_masks[:, 1:],
            single_iou,
            single_masks,
            object_score_logits,
            obj_ptrs[:, 1:],
            obj_ptrs[:, 0],
        )


def _export_point_decoder(
    model: Sam3Image, sam2_features: list[NDArray]
) -> onnxruntime.InferenceSession:
    onnx_file: pathlib.Path = pathlib.Path("models/sam3_point_decoder.onnx")
    graph = (
        onnx.load(onnx_file, load_external_data=False).graph
        if onnx_file.exists()
        else None
    )
    # Earlier decoders lack the outputs for tracking, or take one object.
    if (
        graph is not None
        and "single_obj_ptr" in [o.name for o in graph.output]
        and not [
            i.type.tensor_type.shape.dim[0].dim_value
            for i in graph.input
            if i.name == "image_embeddings"
        ][0]
    ):
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        # Two objects, which the tracker batches on one frame: their own
        # image embeddings with memory, and that frame's high-res features.
        inputs: dict[str, NDArray] = {
            "image_embeddings": np.repeat(sam2_features[0], 2, axis=0),
            "high_res_features1": sam2_features[1],
            "high_res_features2": sam2_features[2],
            "point_coords": np.full((2, 1, 2), 504, dtype=np.float32),
            "point_labels": np.ones((2, 1), dtype=np.float32),
            "mask_input": np.zeros((2, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        }
        output_names: list[str] = [
            "iou_predictions",
            "low_res_masks",
            "single_iou_predictions",
            "single_low_res_masks",
            "object_score_logits",
            "obj_ptrs",
            "single_obj_ptr",
        ]
        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            _PointDecoder(model=model),
            args=tuple(torch.tensor(v) for v in inputs.values()),
            f=onnx_file,
            input_names=list(inputs),
            output_names=output_names,
            dynamic_axes={
                "image_embeddings": {0: "num_objects"},
                "point_coords": {0: "num_objects", 1: "num_points"},
                "point_labels": {0: "num_objects", 1: "num_points"},
                "mask_input": {0: "num_objects"},
                **{name: {0: "num_objects"} for name in output_names},
            },
            opset_version=21,
            dynamo=False,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))
        _postprocess(onnx_file)

    return onnxruntime.InferenceSession(onnx_file)


class _MemoryEncoder(torch.nn.Module):
    """Sam3TrackerBase._encode_new_memory for one frame of a batch of objects."""

    _maskmem_backbone: typing.Any
    _no_mem_embed: torch.Tensor
    _no_obj_embed_spatial: torch.Tensor

    def __init__(self, model: Sam3Image) -> None:
        super().__init__()
        tracker = model.inst_interactive_predictor.model
        self._maskmem_backbone = copy.deepcopy(tracker.maskmem_backbone).cpu()
        self.register_buffer("_no_mem_embed", tracker.no_mem_embed.detach().cpu())
        self.register_buffer(
            "_no_obj_embed_spatial", tracker.no_obj_embed_spatial.detach().cpu()
        )
        self._sigmoid_scale: float = tracker.sigmoid_scale_for_mem_enc
        self._sigmoid_bias: float = tracker.sigmoid_bias_for_mem_enc

    def forward(
        self,
        image_embeddings: torch.Tensor,
        mask_logits: torch.Tensor,
        is_mask_from_pts: torch.Tensor,
        object_score_logits: torch.Tensor,
        non_overlapping: torch.Tensor,
    ) -> torch.Tensor:
        # The image encoder adds no_mem_embed for the point decoder.
        pix_feat = image_embeddings - self._no_mem_embed.view(1, -1, 1, 1)
        # mask_logits is pred_masks_high_res, at 1008 as _forward_sam_heads
        # upsamples it, or at 1152 as Sam3VideoBase._tracker_update_memories does.
        # Sam3TrackerBase._apply_non_overlapping_constraints, which
        # Sam3TrackerPredictor applies to the frames with prompts.
        is_max = mask_logits.argmax(dim=0, keepdim=True) == torch.arange(
            mask_logits.shape[0]
        ).view(-1, 1, 1, 1)
        mask_logits = torch.where(
            (non_overlapping > 0) & ~is_max, mask_logits.clamp(max=-10.0), mask_logits
        )
        mask_for_mem = torch.where(
            is_mask_from_pts > 0, (mask_logits > 0).float(), mask_logits.sigmoid()
        )
        mask_for_mem = mask_for_mem * self._sigmoid_scale + self._sigmoid_bias
        # SimpleMaskEncoder.forward with skip_mask_sigmoid. Its downsampler
        # upsamples with antialias, which ONNX lacks and which upsampling ignores,
        # to 1152, where resizing does nothing.
        backbone = self._maskmem_backbone
        mask_for_mem = torch.nn.functional.interpolate(
            mask_for_mem,
            size=backbone.mask_downsampler.interpol_size,
            mode="bilinear",
            align_corners=False,
        )
        x = backbone.pix_feat_proj(pix_feat)
        x = x + backbone.mask_downsampler.encoder(mask_for_mem)
        maskmem_features = backbone.out_proj(backbone.fuser(x))
        is_obj_appearing = (object_score_logits > 0).float()[..., None, None]
        return (
            maskmem_features
            + (1 - is_obj_appearing) * self._no_obj_embed_spatial[..., None, None]
        )


class _MaskDownsampler(torch.nn.Module):
    """The mask prompt of Sam3TrackerBase._use_mask_as_output."""

    _no_mem_embed: torch.Tensor
    _no_obj_ptr: torch.Tensor

    def __init__(self, model: Sam3Image) -> None:
        super().__init__()
        tracker = model.inst_interactive_predictor.model
        self._mask_downsample = copy.deepcopy(tracker.mask_downsample).cpu()
        self.register_buffer("_no_mem_embed", tracker.no_mem_embed.detach().cpu())
        self.register_buffer("_no_obj_ptr", tracker.no_obj_ptr.detach().cpu())

    def forward(
        self, image_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The point decoder takes the features without no_mem_embed here.
        image_features = image_embeddings - self._no_mem_embed.view(1, -1, 1, 1)
        # The pointer of an empty mask, as _use_mask_as_output gives.
        return image_features, self._mask_downsample(mask), self._no_obj_ptr


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """sam3.sam.rope.apply_rotary_enc in real numbers, repeated along x's tokens."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1, cos.shape[0], cos.shape[1], 2)
    x_real, x_imag = x[..., 0], x[..., 1]
    x = torch.stack([x_real * cos - x_imag * sin, x_real * sin + x_imag * cos], -1)
    return x.reshape(shape)


class _MemoryAttention(torch.nn.Module):
    """Sam3TrackerBase._prepare_memory_conditioned_features after the memory
    selection, which stays outside the model.

    It takes n spatial memories and p object pointers, whose counts vary per
    frame. RoPE applies to the spatial memories only, as num_obj_ptr_tokens
    does in TransformerEncoderCrossAttention.
    """

    _encoder: typing.Any
    _obj_ptr_tpos_proj: typing.Any
    _no_mem_embed: torch.Tensor
    _maskmem_tpos_enc: torch.Tensor
    _src_pos: torch.Tensor
    _maskmem_pos: torch.Tensor
    _cos: torch.Tensor
    _sin: torch.Tensor

    def __init__(self, model: Sam3Image, vision_pos_enc: NDArray) -> None:
        super().__init__()
        tracker = model.inst_interactive_predictor.model
        self._encoder = copy.deepcopy(tracker.transformer.encoder).cpu()
        self.register_buffer("_no_mem_embed", tracker.no_mem_embed.detach().cpu())
        self.register_buffer(
            "_maskmem_tpos_enc", tracker.maskmem_tpos_enc.detach().cpu()
        )
        self._obj_ptr_tpos_proj = copy.deepcopy(tracker.obj_ptr_tpos_proj).cpu()
        self._mem_dim: int = tracker.mem_dim
        # Position encodings depend on the feature size alone. The tracker's
        # image features share their position encoding with the detector's.
        self.register_buffer(
            "_src_pos", torch.from_numpy(vision_pos_enc).flatten(2).transpose(1, 2)
        )
        maskmem_pos = tracker.maskmem_backbone.position_encoding(
            torch.zeros(1, self._mem_dim, *vision_pos_enc.shape[-2:])
        )
        self.register_buffer("_maskmem_pos", maskmem_pos.flatten(2).transpose(1, 2))
        # As get_replace_freqs_cis replaced RoPEAttention.freqs_cis.
        self_attn = self._encoder.layers[0].self_attn
        self.register_buffer("_cos", self_attn.freqs_cos.cpu())
        self.register_buffer("_sin", self_attn.freqs_sin.cpu())

    def _attention(
        self,
        attention: typing.Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_k_rope: int | torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """RoPEAttention.forward for one head, with num_k_rope traced."""
        q = _rope(attention.q_proj(q)[:, None], self._cos, self._sin)
        k = attention.k_proj(k)[:, None]
        k = torch.cat(
            [_rope(k[:, :, :num_k_rope], self._cos, self._sin), k[:, :, num_k_rope:]],
            dim=2,
        )
        v = attention.v_proj(v)[:, None]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        return attention.out_proj(out[:, 0])

    def forward(
        self,
        image_embeddings: torch.Tensor,
        memory: torch.Tensor,
        memory_tpos: torch.Tensor,
        obj_ptrs: torch.Tensor,
        obj_ptr_tpos: torch.Tensor,
        memory_valid: torch.Tensor | None = None,
        obj_ptr_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, height, width = image_embeddings.shape
        pix_feat = image_embeddings - self._no_mem_embed.view(1, -1, 1, 1)
        src = pix_feat.flatten(2).transpose(1, 2)

        # Each spatial memory with its temporal position encoding.
        num_k_rope = memory.shape[0] * memory.shape[2] * memory.shape[3]
        prompt_memory = memory.flatten(2).transpose(1, 2).reshape(1, -1, self._mem_dim)
        prompt_memory_pos = (
            self._maskmem_pos
            + self._maskmem_tpos_enc[memory_tpos].view(-1, 1, self._mem_dim)
        ).reshape(1, -1, self._mem_dim)

        # Each object pointer splits into channels // mem_dim tokens.
        split = channels // self._mem_dim
        prompt_ptrs = obj_ptrs.reshape(1, -1, self._mem_dim)
        ptr_pos = self._obj_ptr_tpos_proj(get_1d_sine_pe(obj_ptr_tpos, dim=channels))
        prompt_ptrs_pos = ptr_pos.repeat_interleave(split, dim=0)[None]

        prompt = torch.cat([prompt_memory, prompt_ptrs], dim=1)
        prompt_pos = torch.cat([prompt_memory_pos, prompt_ptrs_pos], dim=1)

        # The fixed-shape model takes padded memories and pointers, and gives
        # the padding no weight.
        attn_mask: torch.Tensor | None = None
        if memory_valid is not None and obj_ptr_valid is not None:
            attn_mask = torch.cat(
                [
                    ((memory_valid - 1) * 1e4).repeat_interleave(height * width),
                    ((obj_ptr_valid - 1) * 1e4).repeat_interleave(split),
                ]
            ).view(1, 1, 1, -1)

        # TransformerEncoderCrossAttention.forward with TransformerDecoderLayerv2
        # in pre-norm, self-attention first.
        output = src + 0.1 * self._src_pos
        for layer in self._encoder.layers:
            tgt = layer.norm1(output)
            output = output + self._attention(
                layer.self_attn, tgt, tgt, tgt, num_k_rope=tgt.shape[1]
            )
            tgt = layer.norm2(output)
            output = output + self._attention(
                layer.cross_attn_image,
                tgt,
                prompt + prompt_pos,
                prompt,
                num_k_rope=num_k_rope,
                attn_mask=attn_mask,
            )
            tgt = layer.norm3(output)
            output = output + layer.linear2(layer.activation(layer.linear1(tgt)))
        output = self._encoder.norm(output)
        return output.transpose(1, 2).reshape(batch, channels, height, width)


def _export_tracker_models(
    model: Sam3Image, vision_pos_enc: NDArray, sam2_features: list[NDArray]
) -> None:
    image_embeddings: NDArray = sam2_features[0]
    rng = np.random.default_rng(0)

    def export(
        name: str,
        module: typing.Callable[[], torch.nn.Module],
        inputs: dict[str, NDArray],
        output_names: list[str],
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> None:
        onnx_file = pathlib.Path(f"models/sam3_{name}.onnx")
        # Re-export a model that lacks a dynamic axis, like the memory encoder
        # before video, which took low-res masks of a fixed size.
        if onnx_file.exists() and not any(
            dims[axis].dim_value
            for i in onnx.load(onnx_file, load_external_data=False).graph.input
            for dims in [i.type.tensor_type.shape.dim]
            for axis in (dynamic_axes or {}).get(i.name, {})
        ):
            logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
            return
        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            module(),
            args=tuple(torch.tensor(v) for v in inputs.values()),
            f=onnx_file,
            input_names=list(inputs),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=21,
            dynamo=False,
            verify=True,
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))
        _postprocess(onnx_file)

    export(
        "memory_encoder",
        lambda: _MemoryEncoder(model=model),
        inputs={
            "image_embeddings": image_embeddings,
            "mask_logits": rng.normal(size=(2, 1, 1008, 1008)).astype(np.float32),
            "is_mask_from_pts": np.array([1], dtype=np.float32),
            "object_score_logits": np.array([[1], [-1]], dtype=np.float32),
            "non_overlapping": np.array([1], dtype=np.float32),
        },
        output_names=["maskmem_features"],
        dynamic_axes={
            "mask_logits": {0: "num_objects", 2: "height", 3: "width"},
            "object_score_logits": {0: "num_objects"},
            "maskmem_features": {0: "num_objects"},
        },
    )
    export(
        "memory_attention",
        lambda: _MemoryAttention(model=model, vision_pos_enc=vision_pos_enc),
        inputs={
            "image_embeddings": image_embeddings,
            "memory": rng.normal(size=(2, 64, 72, 72)).astype(np.float32),
            "memory_tpos": np.array([6, 5], dtype=np.int64),
            "obj_ptrs": rng.normal(size=(3, 256)).astype(np.float32),
            "obj_ptr_tpos": np.array([0, 1 / 15, 2 / 15], dtype=np.float32),
        },
        output_names=["pix_feat_with_mem"],
        dynamic_axes={
            "memory": {0: "num_memories"},
            "memory_tpos": {0: "num_memories"},
            "obj_ptrs": {0: "num_ptrs"},
            "obj_ptr_tpos": {0: "num_ptrs"},
        },
    )
    # The logits of its cross-attention grow with each frame, and CUDA's shared
    # arena keeps a new region for them per frame until it runs out of memory.
    # CUDA runs this copy, whose fused attention never holds them.
    fused_file = pathlib.Path("models/sam3_memory_attention_fused.onnx")
    if not fused_file.exists():
        shutil.copy("models/sam3_memory_attention.onnx", fused_file)
        fuse_attention(fused_file)
    # CoreML can't compile an input whose shape changes, so it takes this copy,
    # padded to the memories and pointers of one conditioning frame. Other
    # providers run the model above, which is faster for them.
    num_memories, num_ptrs = FIXED_NUM_MEMORIES, FIXED_NUM_OBJ_PTRS
    export(
        "memory_attention_fixed",
        lambda: _MemoryAttention(model=model, vision_pos_enc=vision_pos_enc),
        inputs={
            "image_embeddings": image_embeddings,
            "memory": rng.normal(size=(num_memories, 64, 72, 72)).astype(np.float32),
            "memory_tpos": np.arange(num_memories)[::-1].copy(),
            "obj_ptrs": rng.normal(size=(num_ptrs, 256)).astype(np.float32),
            "obj_ptr_tpos": (np.arange(num_ptrs) / 15).astype(np.float32),
            "memory_valid": np.ones(num_memories, dtype=np.float32),
            "obj_ptr_valid": np.ones(num_ptrs, dtype=np.float32),
        },
        output_names=["pix_feat_with_mem"],
    )
    export(
        "mask_downsampler",
        lambda: _MaskDownsampler(model=model),
        inputs={
            "image_embeddings": image_embeddings,
            "mask": (rng.random((1, 1, 1152, 1152)) > 0.5).astype(np.float32),
        },
        output_names=["image_features", "mask_input", "no_obj_ptr"],
    )


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
    fused_outputs = _export_image_encoder(processor, image, fused=True)
    for output, fused_output in zip(
        [*vision_pos_enc, *backbone_fpn, *sam2_features], sum(fused_outputs, [])
    ):
        np.testing.assert_allclose(fused_output, output, atol=1e-2)
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
    sess_point_decode = _export_point_decoder(model, sam2_features)
    state = processor.set_image(image)
    for multimask_output in [True, False]:
        iou_predictions, low_res_masks = run_point_decoder(
            sess_point_decode=sess_point_decode,
            sam2_features=sam2_features,
            point_prompt=[[point[0] / image.width, point[1] / image.height, 1]],
            multimask_output=multimask_output,
        )
        kwargs = dict(
            point_coords=point[None],
            point_labels=np.array([1]),
            multimask_output=multimask_output,
        )
        _, iou_torch, low_res_torch = model.predict_inst(state, **kwargs)
        _, iou_torch, low_res_torch = model.predict_inst(
            state, mask_input=low_res_torch[[iou_torch.argmax()]], **kwargs
        )
        logger.debug(
            "point decoder vs pytorch (multimask_output={}): "
            "iou_predictions {} vs {}, mask agreement {:.3f}",
            multimask_output,
            iou_predictions[0].round(3),
            iou_torch.round(3),
            ((low_res_masks[0] > 0) == (low_res_torch > 0)).mean(),
        )
    # }}

    # memory_encoder, memory_attention, mask_downsampler {{
    # Sam3TrackerBase.track_step, with the point decoder as its SAM heads
    _export_tracker_models(model, vision_pos_enc[2], sam2_features)
    # }}

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
