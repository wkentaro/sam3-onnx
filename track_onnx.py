"""SAM3's video tracker (Sam3TrackerPredictor) with ONNX models and numpy.

It ports sam3/model/sam3_tracking_predictor.py and the tracking parts of
sam3/model/sam3_tracker_base.py for the configuration that build_sam3_video_model
uses, keeping their names so the two read side by side. The point decoder and
memory attention run one object at a time; the per-frame state holds every
object along the first axis.
"""

import collections
import functools
import typing

import cv2
import ml_dtypes  # installed with onnx
import numpy as np
import onnxruntime
import PIL.Image
from numpy.typing import NDArray

# Sam3TrackerBase and build_tracker
IMAGE_SIZE: int = 1008
LOW_RES_MASK_SIZE: int = 288
INPUT_MASK_SIZE: int = 1152
NUM_MASKMEM: int = 7
MAX_COND_FRAMES_IN_ATTN: int = 4
MAX_OBJ_PTRS_IN_ENCODER: int = 16
MF_THRESHOLD: float = 0.01
NO_OBJ_SCORE: float = -1024.0
MAX_POINT_NUM_IN_PROMPT_ENC: int = 16

# The sessions Tracker runs, as models/sam3_{name}.onnx.
MODEL_NAMES: tuple[str, ...] = (
    "point_decoder",
    "memory_encoder",
    "memory_attention",
    "mask_downsampler",
)

FrameOutput = dict[str, typing.Any]
OutputDict = dict[str, dict[int, FrameOutput]]


def _new_output_dict() -> OutputDict:
    return {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}


def _to_bfloat16(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Round to bfloat16 and back, as the tracker stores its memories."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def _resize(x: NDArray[np.float32], height: int, width: int) -> NDArray[np.float32]:
    """F.interpolate(mode="bilinear", align_corners=False) of (N, 1, H, W)."""
    if x.shape[-2:] == (height, width):
        return x
    return np.stack(
        [
            cv2.resize(m[0], (width, height), interpolation=cv2.INTER_LINEAR)[None]
            for m in x
        ]
    )


def _resize_antialias(
    x: NDArray[np.float32], height: int, width: int
) -> NDArray[np.float32]:
    """F.interpolate(mode="bilinear", antialias=True) of (N, 1, H, W), which
    follows PIL's bilinear filter."""
    if x.shape[-2:] == (height, width):
        return x
    return np.stack(
        [
            np.asarray(
                PIL.Image.fromarray(m[0], mode="F").resize(
                    (width, height), resample=PIL.Image.Resampling.BILINEAR
                )
            )[None]
            for m in x
        ]
    )


def encode_frame(
    session: onnxruntime.InferenceSession, frame: NDArray[np.uint8]
) -> list[NDArray]:
    """Run the image encoder on an RGB frame (H, W, 3) for Tracker."""
    # Sam3TrackerPredictor resizes video frames with PIL's default filter.
    image = PIL.Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE))
    outputs = session.run(
        ["image_embeddings", "high_res_features1", "high_res_features2"],
        {"image": np.asarray(image).transpose(2, 0, 1)},
    )
    return typing.cast(list[NDArray], outputs)


def _select_closest_cond_frames(
    frame_idx: int, cond_frame_outputs: dict[int, FrameOutput], max_cond_frame_num: int
) -> tuple[dict[int, FrameOutput], dict[int, FrameOutput]]:
    """sam3_tracker_utils.select_closest_cond_frames, without keep_first_cond_frame."""
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}
    selected: dict[int, FrameOutput] = {}
    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected[idx_before] = cond_frame_outputs[idx_before]
    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected[idx_after] = cond_frame_outputs[idx_after]
    num_remain = max_cond_frame_num - len(selected)
    inds_remain = sorted(
        (t for t in cond_frame_outputs if t not in selected),
        key=lambda x: abs(x - frame_idx),
    )[:num_remain]
    selected.update((t, cond_frame_outputs[t]) for t in inds_remain)
    unselected = {t: v for t, v in cond_frame_outputs.items() if t not in selected}
    return selected, unselected


class Tracker:
    """Track objects through a video from point, box, and mask prompts.

    get_features(frame_idx) returns the image encoder's image_embeddings,
    high_res_features1, and high_res_features2 for that frame.
    """

    def __init__(
        self,
        sessions: dict[str, onnxruntime.InferenceSession],
        get_features: typing.Callable[[int], list[NDArray]],
        num_frames: int,
        video_height: int,
        video_width: int,
    ) -> None:
        missing = set(MODEL_NAMES) - set(sessions)
        if missing:
            raise ValueError(f"missing sessions: {sorted(missing)}")
        if "single_obj_ptr" not in [
            o.name for o in sessions["point_decoder"].get_outputs()
        ]:
            raise ValueError(
                "the point decoder predates video tracking, "
                "re-export it with: uv run export_onnx.py"
            )
        self._sessions = sessions
        self._get_features = get_features
        self._features_cache: tuple[int, list[NDArray]] | None = None
        self.num_frames = num_frames
        self.video_height = video_height
        self.video_width = video_width

        self.obj_id_to_idx: dict[int, int] = collections.OrderedDict()
        self.obj_idx_to_id: dict[int, int] = collections.OrderedDict()
        self.obj_ids: list[int] = []
        self.point_inputs_per_obj: dict[int, dict[int, dict[str, NDArray]]] = {}
        self.mask_inputs_per_obj: dict[int, dict[int, NDArray]] = {}
        self.output_dict: OutputDict = _new_output_dict()
        self.output_dict_per_obj: dict[int, OutputDict] = {}
        # With add_all_frames_to_correct_as_cond, every prompted frame is a
        # conditioning frame, so the temporary outputs and the consolidated
        # frames only hold "cond_frame_outputs".
        self.temp_output_dict_per_obj: dict[int, dict[int, FrameOutput]] = {}
        self.consolidated_frame_inds: set[int] = set()
        self.tracking_has_started: bool = False
        self.frames_already_tracked: dict[int, dict[str, bool]] = {}

    # Models {{

    def _features(self, frame_idx: int) -> list[NDArray]:
        if self._features_cache is None or self._features_cache[0] != frame_idx:
            self._features_cache = (frame_idx, self._get_features(frame_idx))
        return self._features_cache[1]

    def _run(self, name: str, **inputs: NDArray) -> dict[str, NDArray]:
        session = self._sessions[name]
        outputs = session.run(None, inputs)
        return dict(zip([o.name for o in session.get_outputs()], outputs))

    def _forward_sam_heads(
        self,
        frame_idx: int,
        backbone_features: NDArray,
        point_inputs: dict[str, NDArray] | None,
        mask_inputs: NDArray | None,
        multimask_output: bool,
    ) -> dict[str, NDArray]:
        """Sam3TrackerBase._forward_sam_heads for one object."""
        _, high_res_features1, high_res_features2 = self._features(frame_idx)
        if point_inputs is None:
            point_coords = np.zeros((1, 1, 2), dtype=np.float32)
            point_labels = -np.ones((1, 1), dtype=np.float32)
        else:
            point_coords = point_inputs["point_coords"]
            point_labels = point_inputs["point_labels"].astype(np.float32)
        out = self._run(
            "point_decoder",
            image_embeddings=backbone_features,
            high_res_features1=high_res_features1,
            high_res_features2=high_res_features2,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=(
                np.zeros((1, 1, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE), np.float32)
                if mask_inputs is None
                else mask_inputs
            ),
            has_mask_input=np.array([mask_inputs is not None], dtype=np.float32),
        )
        object_score_logits = out["object_score_logits"]
        if multimask_output:
            ious = out["iou_predictions"]
            best = int(ious[0].argmax())
            low_res_masks = out["low_res_masks"][:, best : best + 1]
            obj_ptr = out["obj_ptrs"][:, best]
        else:
            ious = out["single_iou_predictions"]
            low_res_masks = out["single_low_res_masks"]
            obj_ptr = out["single_obj_ptr"]
        if not object_score_logits[0, 0] > 0:
            low_res_masks = np.full_like(low_res_masks, NO_OBJ_SCORE)
        return {
            "ious": ious,
            "low_res_masks": low_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }

    def _use_mask_as_output(
        self, frame_idx: int, mask_inputs: NDArray
    ) -> dict[str, NDArray]:
        """Sam3TrackerBase._use_mask_as_output for one object."""
        out_scale, out_bias = 20.0, -10.0
        high_res_masks = mask_inputs * out_scale + out_bias
        low_res_masks = _resize_antialias(
            high_res_masks, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE
        )
        downsampled = self._run(
            "mask_downsampler",
            image_embeddings=self._features(frame_idx)[0],
            mask=mask_inputs,
        )
        obj_ptr = self._forward_sam_heads(
            frame_idx,
            backbone_features=downsampled["image_features"],
            point_inputs=None,
            mask_inputs=downsampled["mask_input"],
            multimask_output=False,
        )["obj_ptr"]
        is_obj_appearing = bool((mask_inputs > 0).any())
        return {
            "ious": np.ones((1, 1), dtype=np.float32),
            "low_res_masks": low_res_masks,
            "obj_ptr": obj_ptr if is_obj_appearing else self._no_obj_ptr[None],
            "object_score_logits": np.full(
                (1, 1), out_scale * is_obj_appearing + out_bias, dtype=np.float32
            ),
        }

    def _encode_new_memory(
        self,
        frame_idx: int,
        high_res_masks: NDArray,
        object_score_logits: NDArray,
        is_mask_from_pts: bool,
        non_overlapping: bool = False,
    ) -> NDArray:
        """Sam3TrackerBase._encode_new_memory, stored in bfloat16 as
        Sam3TrackerPredictor does."""
        maskmem_features = self._run(
            "memory_encoder",
            image_embeddings=self._features(frame_idx)[0],
            mask_logits=high_res_masks,
            is_mask_from_pts=np.array([is_mask_from_pts], dtype=np.float32),
            object_score_logits=object_score_logits,
            non_overlapping=np.array([non_overlapping], dtype=np.float32),
        )["maskmem_features"]
        return _to_bfloat16(maskmem_features)

    # }}

    # Sam3TrackerBase {{

    @staticmethod
    def _cal_mem_score(object_score_logits: NDArray, iou_score: NDArray) -> float:
        sigmoid = 1 / (1 + np.exp(-object_score_logits))
        object_score_norm = np.where(object_score_logits > 0, sigmoid * 2 - 1, 0)
        return float((object_score_norm * iou_score).mean())

    def _frame_filter(
        self, output_dict: OutputDict, track_in_reverse: bool, frame_idx: int
    ) -> list[int]:
        num_frames = self.num_frames
        if (frame_idx == 0 and not track_in_reverse) or (
            frame_idx == num_frames - 1 and track_in_reverse
        ):
            return []
        max_num = min(num_frames, MAX_OBJ_PTRS_IN_ENCODER)
        if not track_in_reverse:
            start, end, step, must_include = frame_idx - 1, 0, -1, frame_idx - 1
        else:
            start, end, step, must_include = frame_idx + 1, num_frames, 1, frame_idx + 1
        valid_indices: list[int] = []
        for i in range(start, end, step):
            out = output_dict["non_cond_frame_outputs"].get(i)
            if out is None or "eff_iou_score" not in out:
                continue
            if out["eff_iou_score"] > MF_THRESHOLD:
                valid_indices.insert(0, i)
            if len(valid_indices) >= max_num - 1:
                break
        if must_include not in valid_indices:
            valid_indices.append(must_include)
        return valid_indices

    def _prepare_memory_conditioned_features(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        output_dict: OutputDict,
        track_in_reverse: bool,
        use_prev_mem_frame: bool,
        batch_size: int,
    ) -> list[NDArray]:
        """Sam3TrackerBase._prepare_memory_conditioned_features with
        use_memory_selection, returning each object's features."""
        image_embeddings = self._features(frame_idx)[0]
        if is_init_cond_frame or not use_prev_mem_frame:
            # The image encoder adds no_mem_embed already.
            return [image_embeddings] * batch_size

        tpos_sign_mul = -1 if track_in_reverse else 1
        selected_cond_outputs, unselected_cond_outputs = _select_closest_cond_frames(
            frame_idx, output_dict["cond_frame_outputs"], MAX_COND_FRAMES_IN_ATTN
        )
        t_pos_and_prevs: list[tuple[int, FrameOutput | None, bool]] = [
            ((frame_idx - t) * tpos_sign_mul, out, True)
            for t, out in selected_cond_outputs.items()
        ]
        valid_indices = self._frame_filter(output_dict, track_in_reverse, frame_idx)
        for t_pos in range(1, NUM_MASKMEM):
            t_rel = NUM_MASKMEM - t_pos
            if t_rel > len(valid_indices):
                continue
            prev_frame_idx = valid_indices[-t_rel]
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx)
            if out is None:
                out = unselected_cond_outputs.get(prev_frame_idx)
            t_pos_and_prevs.append((t_pos, out, False))

        memories: list[NDArray] = []
        memory_tpos: list[int] = []
        for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
            if prev is None:
                continue
            memories.append(prev["maskmem_features"])
            t = t_pos if not is_selected_cond_frame else 0
            memory_tpos.append(NUM_MASKMEM - t - 1)

        max_obj_ptrs_in_encoder = min(self.num_frames, MAX_OBJ_PTRS_IN_ENCODER)
        pos_and_ptrs: list[tuple[int, NDArray]] = [
            ((frame_idx - t) * tpos_sign_mul, out["obj_ptr"])
            for t, out in selected_cond_outputs.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        ]
        for t_diff in range(1, max_obj_ptrs_in_encoder):
            if -t_diff <= -len(valid_indices):
                break
            t = valid_indices[-t_diff]
            out = output_dict["non_cond_frame_outputs"].get(
                t, unselected_cond_outputs.get(t)
            )
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))

        memory = np.stack(memories, axis=1)  # (batch_size, n, 64, 72, 72)
        # Reverse tracking from the last frame can have no pointers at all.
        obj_ptrs = (
            np.stack([ptr for _, ptr in pos_and_ptrs], axis=1)
            if pos_and_ptrs
            else np.zeros((batch_size, 0, 256), np.float32)
        )
        obj_ptr_tpos = np.array(
            [pos for pos, _ in pos_and_ptrs], dtype=np.float32
        ) / np.float32(max_obj_ptrs_in_encoder - 1)
        return [
            self._run(
                "memory_attention",
                image_embeddings=image_embeddings,
                memory=memory[i],
                memory_tpos=np.array(memory_tpos, dtype=np.int64),
                obj_ptrs=obj_ptrs[i],
                obj_ptr_tpos=obj_ptr_tpos,
            )["pix_feat_with_mem"]
            for i in range(batch_size)
        ]

    def _track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        point_inputs: dict[str, NDArray] | None,
        mask_inputs: NDArray | None,
        output_dict: OutputDict,
        track_in_reverse: bool,
        run_mem_encoder: bool,
        batch_size: int,
        prev_sam_mask_logits: NDArray | None = None,
        use_prev_mem_frame: bool = True,
    ) -> FrameOutput:
        """Sam3TrackerBase.track_step, as _run_single_frame_inference keeps it."""
        if mask_inputs is not None:
            assert batch_size == 1
            sam_outputs = [self._use_mask_as_output(frame_idx, mask_inputs)]
        else:
            pix_feats_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                output_dict=output_dict,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
                batch_size=batch_size,
            )
            # Sam3TrackerBase._use_multimask
            num_pts = (
                0 if point_inputs is None else point_inputs["point_labels"].shape[1]
            )
            multimask_output = 0 <= num_pts <= 1
            sam_outputs = [
                self._forward_sam_heads(
                    frame_idx,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=prev_sam_mask_logits,
                    multimask_output=multimask_output,
                )
                for pix_feat_with_mem in pix_feats_with_mem
            ]
        pred_masks = np.concatenate([o["low_res_masks"] for o in sam_outputs])
        object_score_logits = np.concatenate(
            [o["object_score_logits"] for o in sam_outputs]
        )
        iou_score = np.concatenate(
            [o["ious"].max(-1, keepdims=True) for o in sam_outputs]
        )
        current_out: FrameOutput = {
            "pred_masks": pred_masks,
            "obj_ptr": np.concatenate([o["obj_ptr"] for o in sam_outputs]),
            "object_score_logits": object_score_logits,
            "iou_score": iou_score,
            "eff_iou_score": self._cal_mem_score(object_score_logits, iou_score),
            "maskmem_features": None,
        }
        if run_mem_encoder:
            current_out["maskmem_features"] = self._encode_new_memory(
                frame_idx,
                high_res_masks=_resize(pred_masks, IMAGE_SIZE, IMAGE_SIZE),
                object_score_logits=object_score_logits,
                is_mask_from_pts=point_inputs is not None,
            )
        return current_out

    # }}

    # Sam3TrackerPredictor {{

    def _obj_id_to_idx(self, obj_id: int) -> int:
        obj_idx = self.obj_id_to_idx.get(obj_id)
        if obj_idx is not None:
            return obj_idx
        if self.tracking_has_started:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {self.obj_ids}."
            )
        obj_idx = len(self.obj_id_to_idx)
        self.obj_id_to_idx[obj_id] = obj_idx
        self.obj_idx_to_id[obj_idx] = obj_id
        self.obj_ids = list(self.obj_id_to_idx)
        self.point_inputs_per_obj[obj_idx] = {}
        self.mask_inputs_per_obj[obj_idx] = {}
        self.output_dict_per_obj[obj_idx] = _new_output_dict()
        self.temp_output_dict_per_obj[obj_idx] = {}
        return obj_idx

    def _get_obj_num(self) -> int:
        return len(self.obj_idx_to_id)

    def add_new_points_or_box(
        self,
        frame_idx: int,
        obj_id: int,
        points: NDArray | None = None,
        labels: NDArray | None = None,
        clear_old_points: bool = True,
        box: NDArray | None = None,
    ) -> tuple[int, list[int], NDArray]:
        """Add points (N, 2) with labels (N,), and a box (x1, y1, x2, y2), all
        normalized by the video size, for one object on one frame.

        Returns the frame index, the object ids, and the mask logits of every
        object on that frame at the video resolution.
        """
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.point_inputs_per_obj[obj_idx]
        mask_inputs_per_frame = self.mask_inputs_per_obj[obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")
        points = (
            np.zeros((0, 2), np.float32) if points is None else np.asarray(points)
        ).astype(np.float32).reshape(1, -1, 2) * IMAGE_SIZE
        labels = (
            (np.zeros(0, np.int32) if labels is None else np.asarray(labels))
            .astype(np.int32)
            .reshape(1, -1)
        )
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            box_coords = np.asarray(box, np.float32).reshape(1, 2, 2) * IMAGE_SIZE
            points = np.concatenate([box_coords, points], axis=1)
            labels = np.concatenate([np.array([[2, 3]], np.int32), labels], axis=1)

        point_inputs = (
            None if clear_old_points else point_inputs_per_frame.get(frame_idx)
        )
        if point_inputs is None:
            point_inputs = {"point_coords": points, "point_labels": labels}
        else:
            point_inputs = {
                "point_coords": np.concatenate(
                    [point_inputs["point_coords"], points], axis=1
                ),
                "point_labels": np.concatenate(
                    [point_inputs["point_labels"], labels], axis=1
                ),
            }
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in self.frames_already_tracked
        reverse = (
            False
            if is_init_cond_frame
            else self.frames_already_tracked[frame_idx]["reverse"]
        )
        obj_output_dict = self.output_dict_per_obj[obj_idx]
        obj_temp_output_dict = self.temp_output_dict_per_obj[obj_idx]

        num_points = point_inputs["point_coords"].shape[1]
        if num_points > MAX_POINT_NUM_IN_PROMPT_ENC:
            num_first = MAX_POINT_NUM_IN_PROMPT_ENC // 2
            num_last = MAX_POINT_NUM_IN_PROMPT_ENC - num_first
            point_inputs = {
                k: np.concatenate([v[:, :num_first], v[:, -num_last:]], axis=1)
                for k, v in point_inputs.items()
            }

        # iter_use_prev_mask_pred
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict.get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = np.clip(prev_out["pred_masks"], -32.0, 32.0)

        current_out = self._track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=obj_output_dict,
            track_in_reverse=reverse,
            run_mem_encoder=False,
            batch_size=1,
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=False,
        )
        obj_temp_output_dict[frame_idx] = current_out
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx, run_mem_encoder=False, consolidate_at_video_res=True
        )
        return (
            frame_idx,
            self.obj_ids,
            self._get_orig_video_res_output(consolidated_out["pred_masks_video_res"]),
        )

    def add_new_mask(
        self, frame_idx: int, obj_id: int, mask: NDArray
    ) -> tuple[int, list[int], NDArray]:
        """Add a mask (H, W) for one object on one frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.point_inputs_per_obj[obj_idx]
        mask_inputs_per_frame = self.mask_inputs_per_obj[obj_idx]

        assert mask.ndim == 2
        mask_inputs_orig = np.asarray(mask, dtype=np.float32)[None, None]
        mask_inputs = _resize_antialias(
            mask_inputs_orig, INPUT_MASK_SIZE, INPUT_MASK_SIZE
        )
        mask_inputs_video_res = (
            _resize_antialias(mask_inputs_orig, self.video_height, self.video_width)
            > 0.5
        )
        mask_inputs_per_frame[frame_idx] = mask_inputs_video_res
        point_inputs_per_frame.pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in self.frames_already_tracked
        reverse = (
            False
            if is_init_cond_frame
            else self.frames_already_tracked[frame_idx]["reverse"]
        )
        current_out = self._track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict=self.output_dict_per_obj[obj_idx],
            track_in_reverse=reverse,
            run_mem_encoder=False,
            batch_size=1,
        )
        current_out["pred_masks"] = None
        current_out["pred_masks_video_res"] = np.where(
            mask_inputs_video_res, -NO_OBJ_SCORE, NO_OBJ_SCORE
        ).astype(np.float32)
        self.temp_output_dict_per_obj[obj_idx][frame_idx] = current_out
        for obj_idx2, obj_temp_output_dict2 in self.temp_output_dict_per_obj.items():
            if obj_idx2 == obj_idx:
                continue
            current_out2 = obj_temp_output_dict2.get(frame_idx)
            if current_out2 is not None and "pred_masks_video_res" in current_out2:
                current_out2["pred_masks_video_res"] = np.where(
                    mask_inputs_video_res,
                    NO_OBJ_SCORE,
                    current_out2["pred_masks_video_res"],
                ).astype(np.float32)
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx, run_mem_encoder=False, consolidate_at_video_res=True
        )
        return (
            frame_idx,
            self.obj_ids,
            self._get_orig_video_res_output(consolidated_out["pred_masks_video_res"]),
        )

    def _get_orig_video_res_output(self, any_res_masks: NDArray) -> NDArray:
        return _resize(any_res_masks, self.video_height, self.video_width)

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx: int,
        run_mem_encoder: bool,
        consolidate_at_video_res: bool = False,
    ) -> FrameOutput:
        batch_size = self._get_obj_num()
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_size = (self.video_height, self.video_width)
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_size = (LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE)
            consolidated_mask_key = "pred_masks"
        consolidated_out: FrameOutput = {
            "maskmem_features": None,
            consolidated_mask_key: np.full(
                (batch_size, 1, *consolidated_size), NO_OBJ_SCORE, dtype=np.float32
            ),
            "obj_ptr": np.full((batch_size, 256), NO_OBJ_SCORE, dtype=np.float32),
            "object_score_logits": np.full((batch_size, 1), 10.0, dtype=np.float32),
            "iou_score": np.zeros((batch_size, 1), dtype=np.float32),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = self.temp_output_dict_per_obj[obj_idx]
            obj_output_dict = self.output_dict_per_obj[obj_idx]
            out = obj_temp_output_dict.get(frame_idx)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            if out is None:
                if run_mem_encoder:
                    # _get_empty_mask_ptr: an empty mask gives no_obj_ptr.
                    consolidated_out["obj_ptr"][obj_idx] = self._no_obj_ptr
                continue
            if out.get("pred_masks_video_res") is not None:
                obj_mask = _resize_antialias(
                    out["pred_masks_video_res"], *consolidated_size
                )
            else:
                obj_mask = _resize(out["pred_masks"], *consolidated_size)
            consolidated_out[consolidated_mask_key][obj_idx] = obj_mask[0]
            consolidated_out["obj_ptr"][obj_idx] = out["obj_ptr"][0]
            consolidated_out["object_score_logits"][obj_idx] = out[
                "object_score_logits"
            ][0]
            consolidated_out["iou_score"][obj_idx] = out["iou_score"][0]

        if run_mem_encoder:
            consolidated_out["maskmem_features"] = self._encode_new_memory(
                frame_idx,
                high_res_masks=_resize(
                    consolidated_out["pred_masks"], IMAGE_SIZE, IMAGE_SIZE
                ),
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,
                non_overlapping=True,
            )
        return consolidated_out

    @functools.cached_property
    def _no_obj_ptr(self) -> NDArray:
        # A constant of the model, which ignores the inputs for it.
        return self._run(
            "mask_downsampler",
            image_embeddings=np.zeros((1, 256, 72, 72), np.float32),
            mask=np.zeros((1, 1, INPUT_MASK_SIZE, INPUT_MASK_SIZE), np.float32),
        )["no_obj_ptr"][0]

    def propagate_in_video_preflight(self) -> None:
        self.tracking_has_started = True
        batch_size = self._get_obj_num()
        storage_key = "cond_frame_outputs"
        temp_frame_inds: set[int] = set()
        for obj_temp_output_dict in self.temp_output_dict_per_obj.values():
            temp_frame_inds.update(obj_temp_output_dict.keys())
        self.consolidated_frame_inds.update(temp_frame_inds)
        for frame_idx in temp_frame_inds:
            consolidated_out = self._consolidate_temp_output_across_obj(
                frame_idx, run_mem_encoder=True
            )
            self.output_dict[storage_key][frame_idx] = consolidated_out
            self._add_output_per_object(frame_idx, consolidated_out, storage_key)
            # clear_non_cond_mem_around_input, for a single object only
            if batch_size <= 1:
                self._clear_non_cond_mem_around_input(frame_idx)
        for obj_temp_output_dict in self.temp_output_dict_per_obj.values():
            obj_temp_output_dict.clear()

        for frame_idx in self.output_dict["cond_frame_outputs"]:
            self.output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in self.output_dict_per_obj.values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in self.consolidated_frame_inds:
            assert frame_idx in self.output_dict["cond_frame_outputs"]

        input_frames_inds: set[int] = set()
        for point_inputs_per_frame in self.point_inputs_per_obj.values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in self.mask_inputs_per_obj.values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert self.consolidated_frame_inds == input_frames_inds

    def _get_processing_order(
        self,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> typing.Iterable[int]:
        num_frames = self.num_frames
        if start_frame_idx is None:
            start_frame_idx = min(self.output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                return range(start_frame_idx, end_frame_idx - 1, -1)
            return [0]
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        return range(start_frame_idx, end_frame_idx + 1)

    def propagate_in_video(
        self,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> typing.Iterator[tuple[int, list[int], NDArray, NDArray]]:
        """Yield the frame index, the object ids, the mask logits at the video
        resolution, and the object score logits of each tracked frame."""
        self.propagate_in_video_preflight()
        if len(self.output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        for frame_idx in self._get_processing_order(
            start_frame_idx, max_frame_num_to_track, reverse
        ):
            current_out = self.propagate_one_frame(frame_idx, reverse)
            yield (
                frame_idx,
                self.obj_ids,
                self._get_orig_video_res_output(current_out["pred_masks"]),
                current_out["object_score_logits"],
            )

    def propagate_one_frame(
        self, frame_idx: int, reverse: bool, run_mem_encoder: bool = True
    ) -> FrameOutput:
        """Track every object on one frame after propagate_in_video_preflight, as
        a step of propagate_in_video. The output has the low-res "pred_masks"."""
        output_dict = self.output_dict
        if frame_idx in self.consolidated_frame_inds:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            if self._get_obj_num() <= 1:
                self._clear_non_cond_mem_around_input(frame_idx)
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict,
                track_in_reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                batch_size=self._get_obj_num(),
            )
            output_dict[storage_key][frame_idx] = current_out
        self._add_output_per_object(frame_idx, current_out, storage_key)
        self.frames_already_tracked[frame_idx] = {"reverse": reverse}
        return current_out

    def update_memories(
        self, frame_idx: int, high_res_masks: NDArray, object_score_logits: NDArray
    ) -> None:
        """Encode the memory of one frame from masks that the caller resolved
        across trackers, as Sam3VideoBase._tracker_update_memories does."""
        maskmem_features = self._encode_new_memory(
            frame_idx,
            high_res_masks=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )
        for storage_key, outputs in self.output_dict.items():
            if frame_idx in outputs:
                outputs[frame_idx]["maskmem_features"] = maskmem_features
                self._add_output_per_object(frame_idx, outputs[frame_idx], storage_key)

    def _add_output_per_object(
        self, frame_idx: int, current_out: FrameOutput, storage_key: str
    ) -> None:
        maskmem_features = current_out["maskmem_features"]
        for obj_idx, obj_output_dict in self.output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_output_dict[storage_key][frame_idx] = {
                "maskmem_features": None
                if maskmem_features is None
                else maskmem_features[obj_slice],
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
                "iou_score": current_out["iou_score"][obj_slice],
            }

    def clear_all_points_in_frame(self, frame_idx: int, obj_id: int) -> None:
        obj_idx = self._obj_id_to_idx(obj_id)
        self.point_inputs_per_obj[obj_idx].pop(frame_idx, None)
        self.mask_inputs_per_obj[obj_idx].pop(frame_idx, None)
        self.temp_output_dict_per_obj[obj_idx].pop(frame_idx, None)

        batch_size = self._get_obj_num()
        frame_has_input = any(
            frame_idx in self.point_inputs_per_obj[obj_idx2]
            or frame_idx in self.mask_inputs_per_obj[obj_idx2]
            for obj_idx2 in range(batch_size)
        )
        if not frame_has_input:
            output_dict = self.output_dict
            self.consolidated_frame_inds.discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                self.frames_already_tracked.pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                obj_output_dict = self.output_dict_per_obj[obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            # Unlike upstream, keep the prompts of other objects that have not
            # been propagated yet, which aren't in "cond_frame_outputs".
            if len(output_dict["cond_frame_outputs"]) == 0 and not any(
                [
                    *self.point_inputs_per_obj.values(),
                    *self.mask_inputs_per_obj.values(),
                ]
            ):
                self._reset_tracking_results()

    def clear_all_points_in_video(self) -> None:
        self._reset_tracking_results()
        self.obj_id_to_idx.clear()
        self.obj_idx_to_id.clear()
        self.obj_ids.clear()
        self.point_inputs_per_obj.clear()
        self.mask_inputs_per_obj.clear()
        self.output_dict_per_obj.clear()
        self.temp_output_dict_per_obj.clear()

    def _reset_tracking_results(self) -> None:
        for v in self.point_inputs_per_obj.values():
            v.clear()
        for v in self.mask_inputs_per_obj.values():
            v.clear()
        for d in [self.output_dict, *self.output_dict_per_obj.values()]:
            d["cond_frame_outputs"].clear()
            d["non_cond_frame_outputs"].clear()
        for v in self.temp_output_dict_per_obj.values():
            v.clear()
        self.consolidated_frame_inds.clear()
        self.tracking_has_started = False
        self.frames_already_tracked.clear()

    def remove_object(self, obj_id: int) -> list[int]:
        """Remove an object, and return the remaining object ids."""
        old_obj_idx_to_rm = self.obj_id_to_idx.get(obj_id)
        if old_obj_idx_to_rm is None:
            return self.obj_ids
        if len(self.obj_id_to_idx) == 1:
            self.clear_all_points_in_video()
            return self.obj_ids

        obj_input_frames_inds = set(self.point_inputs_per_obj[old_obj_idx_to_rm])
        obj_input_frames_inds.update(self.mask_inputs_per_obj[old_obj_idx_to_rm])
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(frame_idx, obj_id)

        old_obj_ids = self.obj_ids
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = [i for i in old_obj_inds if i != old_obj_idx_to_rm]
        new_obj_ids = [old_obj_ids[i] for i in remain_old_obj_inds]
        old_idx_to_new_idx = {old: new for new, old in enumerate(remain_old_obj_inds)}
        self.obj_id_to_idx = collections.OrderedDict(
            (obj_id, i) for i, obj_id in enumerate(new_obj_ids)
        )
        self.obj_idx_to_id = collections.OrderedDict(enumerate(new_obj_ids))
        self.obj_ids = new_obj_ids

        def remap(container: dict[int, typing.Any]) -> dict[int, typing.Any]:
            return {
                old_idx_to_new_idx[k]: v
                for k, v in container.items()
                if k in old_idx_to_new_idx
            }

        self.point_inputs_per_obj = remap(self.point_inputs_per_obj)
        self.mask_inputs_per_obj = remap(self.mask_inputs_per_obj)
        self.output_dict_per_obj = remap(self.output_dict_per_obj)
        self.temp_output_dict_per_obj = remap(self.temp_output_dict_per_obj)

        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            for frame_idx, out in self.output_dict[storage_key].items():
                for key in [
                    "maskmem_features",
                    "pred_masks",
                    "obj_ptr",
                    "object_score_logits",
                    "iou_score",
                ]:
                    out[key] = out[key][remain_old_obj_inds]
                out["eff_iou_score"] = self._cal_mem_score(
                    out["object_score_logits"], out["iou_score"]
                )
                self._add_output_per_object(frame_idx, out, storage_key)
        return self.obj_ids

    def _clear_non_cond_mem_around_input(self, frame_idx: int) -> None:
        for obj_output_dict in self.output_dict_per_obj.values():
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx - NUM_MASKMEM, frame_idx + NUM_MASKMEM + 1):
                non_cond_frame_outputs.pop(t, None)

    # }}
