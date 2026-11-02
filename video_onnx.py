"""SAM3's video predictor (Sam3VideoInferenceWithInstanceInteractivity) with
ONNX models and numpy.

It ports sam3/model/sam3_video_inference.py and sam3/model/sam3_video_base.py
for the configuration that build_sam3_video_model uses on one GPU, keeping their
names so the files read side by side. The detector finds the objects of a text
or box prompt on every frame, and a track_onnx.Tracker tracks the objects that
first appear on the same frame.

The configuration turns off several heuristics, which the port leaves out:
masklet confirmation, keep-alive suppression, boundary suppression, bounding
box reconditioning, and the object limit.
"""

import collections
import typing

import cv2
import numpy as np
import onnxruntime
import PIL.Image
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize

from infer_onnx import make_box_inputs
from infer_onnx import preprocess_image
from track_onnx import INPUT_MASK_SIZE
from track_onnx import LOW_RES_MASK_SIZE
from track_onnx import MODEL_NAMES as TRACKER_MODEL_NAMES
from track_onnx import Tracker
from track_onnx import _resize

# build_sam3_video_model with apply_temporal_disambiguation
SCORE_THRESHOLD_DETECTION: float = 0.5
DET_NMS_THRESH: float = 0.1
ASSOC_IOU_THRESH: float = 0.1
TRK_ASSOC_IOU_THRESH: float = 0.5
NEW_DET_THRESH: float = 0.7
HOTSTART_DELAY: int = 15
HOTSTART_UNMATCH_THRESH: int = 8
HOTSTART_DUP_THRESH: int = 8
SUPPRESS_OVERLAPPING_BASED_ON_RECENT_OCCLUSION_THRESHOLD: float = 0.7
FILL_HOLE_AREA: int = 16
RECONDITION_EVERY_NTH_FRAME: int = 16
# Sam3VideoBase._associate_det_trk and _recondition_masklets
HIGH_CONF_THRESH: float = 0.8
HIGH_IOU_THRESH: float = 0.8

# The sessions VideoInference runs, as models/sam3_{name}.onnx. The caller runs
# the image encoder, with encode_frame.
MODEL_NAMES: tuple[str, ...] = ("language_encoder", "decoder", *TRACKER_MODEL_NAMES)

VideoOutput = dict[str, NDArray]


def encode_frame(
    session: onnxruntime.InferenceSession, frame: NDArray[np.uint8]
) -> dict[str, NDArray]:
    """Run the image encoder on an RGB frame (H, W, 3) for VideoInference."""
    outputs = session.run(None, {"image": preprocess_image(PIL.Image.fromarray(frame))})
    return dict(
        zip(
            [o.name for o in session.get_outputs()], typing.cast(list[NDArray], outputs)
        )
    )


def _mask_iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> NDArray[np.float32]:
    """perflib.masks_ops.mask_iou of (N, H, W) and (M, H, W)."""
    a = a.reshape(len(a), -1).astype(np.float32)
    b = b.reshape(len(b), -1).astype(np.float32)
    intersection = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None] - intersection
    return intersection / np.maximum(union, 1)


def _nms_masks(probs: NDArray, masks: NDArray, iou_threshold: float) -> NDArray:
    """perflib.nms.nms_masks with generic_nms_cpu, for detections above the score
    threshold, returning the indices it keeps in order."""
    binary = masks > 0
    ious = _mask_iou(binary, binary)
    order = probs.argsort()[::-1]
    kept: list[int] = []
    while order.size > 0:
        i, order = order[0], order[1:]
        kept.append(int(i))
        order = order[ious[i, order] <= iou_threshold]
    return np.sort(np.array(kept, np.int64))


def _component_areas(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """The area of each pixel's 8-connected component, as perflib does."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return stats[labels, cv2.CC_STAT_AREA]


def _fill_holes_in_mask_scores(masks: NDArray) -> NDArray:
    """sam3_tracker_utils.fill_holes_in_mask_scores of (N, H, W)."""
    masks = masks.copy()
    for mask in masks:
        mask_bg = mask <= 0
        mask[mask_bg & (_component_areas(mask_bg) <= FILL_HOLE_AREA)] = 0.1
        mask_fg = mask > 0
        fg_area_thresh = min(int(mask_fg.sum()) // 2, FILL_HOLE_AREA)
        mask[mask_fg & (_component_areas(mask_fg) <= fg_area_thresh)] = -0.1
    return masks


def _apply_non_overlapping_constraints(pred_masks: NDArray) -> NDArray:
    """Sam3TrackerBase._apply_non_overlapping_constraints of (N, 1, H, W)."""
    if len(pred_masks) == 1:
        return pred_masks
    keep = pred_masks.argmax(0)[None] == np.arange(len(pred_masks))[:, None, None, None]
    return np.where(keep, pred_masks, np.minimum(pred_masks, -10.0))


def _suppress_object_pw_area_shrinkage(pred_masks: NDArray) -> NDArray:
    """Sam3TrackerPredictor._suppress_object_pw_area_shrinkage of (N, 1, H, W)."""
    non_overlapping = _apply_non_overlapping_constraints(pred_masks)
    area_before = np.maximum((pred_masks > 0).sum((-1, -2)), 1)
    area_after = (non_overlapping > 0).sum((-1, -2))
    keep = area_after / area_before >= 0.3
    return np.where(keep[..., None, None], pred_masks, np.minimum(pred_masks, -10.0))


def _apply_object_wise_non_overlapping_constraints(
    masks: NDArray[np.bool_], obj_scores: NDArray
) -> NDArray[np.bool_]:
    """Sam3TrackerPredictor._apply_object_wise_non_overlapping_constraints of
    binary masks (N, H, W) with background_value=0: the highest scoring object
    claims each pixel."""
    scores = np.where(masks, obj_scores[:, None, None], 0)
    is_max = scores.argmax(0)[None] == np.arange(len(masks))[:, None, None]
    return masks & is_max & (scores > 0)


def _masks_to_boxes(masks: NDArray[np.bool_]) -> NDArray[np.float32]:
    """torchvision.ops.masks_to_boxes of non-empty (N, H, W), in xyxy."""
    boxes = []
    for mask in masks:
        ys = np.flatnonzero(mask.any(1))
        xs = np.flatnonzero(mask.any(0))
        boxes.append([xs[0], ys[0], xs[-1], ys[-1]])
    return np.array(boxes, np.float32).reshape(-1, 4)


class VideoInference:
    """Detect, segment, and track the objects of a text or box prompt in a
    video.

    get_features(frame_idx) returns the image encoder's outputs by name for
    that frame, as encode_frame gives.
    """

    def __init__(
        self,
        sessions: dict[str, onnxruntime.InferenceSession],
        get_features: typing.Callable[[int], dict[str, NDArray]],
        num_frames: int,
        video_height: int,
        video_width: int,
    ) -> None:
        missing = set(MODEL_NAMES) - set(sessions)
        if missing:
            raise ValueError(f"missing sessions: {sorted(missing)}")
        if "mask_logits" not in [o.name for o in sessions["decoder"].get_outputs()]:
            raise ValueError(
                "the decoder predates video tracking, "
                "re-export it with: uv run export_onnx.py"
            )
        self._sessions = sessions
        self._get_features = get_features
        self._features_cache: tuple[int, dict[str, NDArray]] | None = None
        self.num_frames = num_frames
        self.video_height = video_height
        self.video_width = video_width
        self.reset_state()

    def reset_state(self) -> None:
        """Revert to the state right after initialization."""
        self.text_prompt: str | None = None
        self._language_features: dict[str, NDArray] = {}
        # frame_idx -> the box (cx, cy, w, h) and its label
        self.per_frame_geometric_prompt: dict[int, tuple[NDArray, int]] = {}
        self.frames_with_outputs: set[int] = set()
        self.trackers: list[Tracker] = []
        # Sam3VideoBase._initialize_metadata, on one GPU
        self.obj_ids: NDArray[np.int64] = np.array([], np.int64)
        self.max_obj_id: int = -1
        self.obj_id_to_score: dict[int, float] = {}
        self.obj_id_to_tracker_score_frame_wise: dict[int, dict[int, float]] = (
            collections.defaultdict(dict)
        )
        self.obj_id_to_last_occluded: dict[int, int] = {}
        self.obj_first_frame_idx: dict[int, int] = {}
        self.unmatched_frame_inds: dict[int, list[int]] = collections.defaultdict(list)
        self.overlap_pair_to_frame_inds: dict[tuple[int, int], list[int]] = (
            collections.defaultdict(list)
        )
        self.removed_obj_ids: set[int] = set()

    # Models {{

    def _features(self, frame_idx: int) -> dict[str, NDArray]:
        if self._features_cache is None or self._features_cache[0] != frame_idx:
            self._features_cache = (frame_idx, self._get_features(frame_idx))
        return self._features_cache[1]

    def _tracker_features(self, frame_idx: int) -> list[NDArray]:
        features = self._features(frame_idx)
        return [
            features[name]
            for name in ["image_embeddings", "high_res_features1", "high_res_features2"]
        ]

    def _new_tracker(self) -> Tracker:
        return Tracker(
            self._sessions,
            self._tracker_features,
            self.num_frames,
            self.video_height,
            self.video_width,
        )

    def _encode_prompt(self, text: str) -> None:
        session = self._sessions["language_encoder"]
        language_mask, language_features, _ = session.run(
            None, {"tokens": tokenize(texts=[text], context_length=32)}
        )
        self._language_features = {
            "language_mask": typing.cast(NDArray, language_mask),
            "language_features": typing.cast(NDArray, language_features),
        }

    def _run_detector(self, frame_idx: int) -> tuple[NDArray, NDArray]:
        """The detections above the score threshold, as mask logits (N, H, W) and
        scores (N,)."""
        session = self._sessions["decoder"]
        boxes = []
        if frame_idx in self.per_frame_geometric_prompt:
            box, label = self.per_frame_geometric_prompt[frame_idx]
            boxes.append([*box, label])
        features = self._features(frame_idx)
        probs, mask_logits = session.run(
            ["scores", "mask_logits"],
            {
                **{
                    f"backbone_fpn_{i}": features[f"backbone_fpn_{i}"] for i in range(3)
                },
                "vision_pos_enc_2": features["vision_pos_enc_2"],
                **self._language_features,
                **make_box_inputs(session, boxes),
            },
        )
        probs = typing.cast(NDArray, probs)
        # The video detector folds the presence score into its logits, with
        # inverse_sigmoid's eps of 1e-3.
        x1 = np.maximum(probs, 1e-3)
        scores = x1 / (x1 + np.maximum(1 - probs, 1e-3))
        masks = typing.cast(NDArray, mask_logits)[:, 0]
        keep = _nms_masks(scores, masks, DET_NMS_THRESH)
        return masks[keep], scores[keep]

    # }}

    # Sam3VideoInference {{

    def add_prompt(
        self,
        frame_idx: int,
        text_str: str | None = None,
        boxes_xywh: NDArray | None = None,
        box_labels: NDArray | None = None,
    ) -> tuple[int, VideoOutput]:
        """Prompt with a text, a box (x, y, w, h) normalized by the video size,
        or both, on one frame, and return the outputs on that frame.

        The prompt applies to every frame, and starts over the tracking.
        """
        if text_str is None and boxes_xywh is None:
            raise ValueError("at least one of text_str or boxes_xywh must be given")
        if not 0 <= frame_idx < self.num_frames:
            raise ValueError(f"{frame_idx=} is out of range of {self.num_frames}")
        self.reset_state()
        if text_str is not None and text_str != "visual":
            self.text_prompt = text_str
        self._encode_prompt(self.text_prompt or "visual")
        if boxes_xywh is not None:
            if box_labels is None:
                raise ValueError("boxes_xywh and box_labels must be given together")
            boxes_xywh = np.asarray(boxes_xywh, np.float32).reshape(-1, 4)
            if len(boxes_xywh) != 1:
                raise ValueError("a box as the initial prompt must be one box")
            x, y, w, h = boxes_xywh[0]
            box_cxcywh = np.array([x + w / 2, y + h / 2, w, h], np.float32)
            if not ((boxes_xywh >= 0).all() and (box_cxcywh <= 1).all()):
                raise ValueError("boxes_xywh must be normalized to 0~1")
            label = int(np.asarray(box_labels).reshape(-1)[0])
            self.per_frame_geometric_prompt[frame_idx] = (box_cxcywh, label)
        out = self._run_single_frame_inference(frame_idx, reverse=False)
        return frame_idx, self._postprocess_output(out)

    def _get_processing_order(
        self,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> tuple[range, int]:
        if start_frame_idx is None:
            if not self.frames_with_outputs:
                raise RuntimeError(
                    "No prompts are received on any frames. "
                    "Please add prompt on at least one frame before propagation."
                )
            start_frame_idx = min(self.frames_with_outputs)
        if max_frame_num_to_track is None:
            max_frame_num_to_track = self.num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            return range(start_frame_idx - 1, end_frame_idx - 1, -1), end_frame_idx
        end_frame_idx = min(
            start_frame_idx + max_frame_num_to_track, self.num_frames - 1
        )
        return range(start_frame_idx, end_frame_idx + 1), end_frame_idx

    def propagate_in_video(
        self,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> typing.Iterator[tuple[int, VideoOutput]]:
        """Yield the outputs of each frame. Frames come out HOTSTART_DELAY frames
        late, so objects that hotstart removes never show up."""
        processing_order, end_frame_idx = self._get_processing_order(
            start_frame_idx, max_frame_num_to_track, reverse
        )
        hotstart_buffer: list[tuple[int, dict[str, typing.Any]]] = []
        hotstart_removed_obj_ids: set[int] = set()
        for frame_idx in processing_order:
            out = self._run_single_frame_inference(frame_idx, reverse)
            hotstart_buffer.append((frame_idx, out))
            hotstart_removed_obj_ids.update(out["removed_obj_ids"])
            if frame_idx == end_frame_idx:
                yield_list, hotstart_buffer = hotstart_buffer, []
            elif len(hotstart_buffer) >= HOTSTART_DELAY:
                yield_list, hotstart_buffer = hotstart_buffer[:1], hotstart_buffer[1:]
            else:
                continue
            for yield_frame_idx, yield_out in yield_list:
                yield (
                    yield_frame_idx,
                    self._postprocess_output(yield_out, hotstart_removed_obj_ids),
                )

    def _run_single_frame_inference(
        self, frame_idx: int, reverse: bool
    ) -> dict[str, typing.Any]:
        obj_id_to_mask = self._det_track_one_frame(
            frame_idx,
            reverse,
            allow_new_detections=self.text_prompt is not None
            or frame_idx in self.per_frame_geometric_prompt,
        )
        self.frames_with_outputs.add(frame_idx)
        # Snapshots, as upstream copies the metadata on every frame.
        return {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": dict(self.obj_id_to_score),
            "obj_id_to_tracker_score": dict(
                self.obj_id_to_tracker_score_frame_wise[frame_idx]
            ),
            "removed_obj_ids": set(self.removed_obj_ids),
        }

    def _postprocess_output(
        self, out: dict[str, typing.Any], removed_obj_ids: typing.Collection[int] = ()
    ) -> VideoOutput:
        """The object ids, detection scores, boxes (x, y, w, h) normalized by the
        video size, and binary masks at the video resolution."""
        obj_id_to_mask = out["obj_id_to_mask"]
        obj_ids = np.array(sorted(obj_id_to_mask), np.int64)
        masks = np.zeros((0, self.video_height, self.video_width), np.bool_)
        if len(obj_ids) > 0:
            masks = np.stack([obj_id_to_mask[obj_id] for obj_id in obj_ids])
        probs = np.array(
            [out["obj_id_to_score"][obj_id] for obj_id in obj_ids], np.float32
        )
        tracker_probs = np.array(
            [out["obj_id_to_tracker_score"].get(obj_id, 0.0) for obj_id in obj_ids],
            np.float32,
        )
        keep = masks.any((1, 2)) & ~np.isin(obj_ids, list(removed_obj_ids))
        obj_ids, probs, tracker_probs, masks = (
            obj_ids[keep],
            probs[keep],
            tracker_probs[keep],
            masks[keep],
        )
        boxes_xywh = _masks_to_boxes(masks)
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        boxes_xywh /= np.array([self.video_width, self.video_height] * 2, np.float32)
        if len(masks) > 1:
            masks = _apply_object_wise_non_overlapping_constraints(masks, tracker_probs)
        return {
            "out_obj_ids": obj_ids,
            "out_probs": probs,
            "out_boxes_xywh": boxes_xywh,
            "out_binary_masks": masks,
        }

    # }}

    # Sam3VideoBase {{

    def _det_track_one_frame(
        self, frame_idx: int, reverse: bool, allow_new_detections: bool
    ) -> dict[int, NDArray[np.bool_]]:
        # Step 1: run the detector, which upstream also runs on frames without
        # prompts and then drops every detection.
        det_masks, det_scores = (
            self._run_detector(frame_idx)
            if allow_new_detections
            else (
                np.zeros((0, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE), np.float32),
                np.zeros(0, np.float32),
            )
        )

        # Step 2: propagate the tracked objects without encoding memories.
        obj_ids_prev = self.obj_ids
        trk_masks, trk_scores = self._run_tracker_propagation(frame_idx, reverse)

        # Step 3: run_tracker_update_planning_phase
        (
            new_det_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_det,
        ) = self._associate_det_trk(det_masks, det_scores, trk_masks, obj_ids_prev)
        new_det_obj_ids = self.max_obj_id + 1 + np.arange(len(new_det_inds))
        obj_ids_newly_removed = self._process_hotstart(
            frame_idx,
            reverse,
            det_to_matched_trk_obj_ids,
            new_det_obj_ids,
            unmatched_trk_obj_ids,
        )
        if (
            RECONDITION_EVERY_NTH_FRAME > 0
            and frame_idx % RECONDITION_EVERY_NTH_FRAME == 0
            and trk_id_to_det
        ):
            self._recondition_masklets(
                frame_idx, det_masks, trk_id_to_det, obj_ids_prev, trk_scores
            )
        last_occluded: dict[int, int] = {}
        if len(trk_masks) > 0:
            last_occluded = self._suppress_overlapping_based_on_recent_occlusion(
                frame_idx, trk_masks, obj_ids_prev, obj_ids_newly_removed, reverse
            )
            self._tracker_update_memories(frame_idx, trk_masks)

        # the new metadata
        obj_ids = np.concatenate([obj_ids_prev, new_det_obj_ids])
        self.obj_ids = obj_ids[~np.isin(obj_ids, list(obj_ids_newly_removed))]
        self.obj_id_to_last_occluded = last_occluded
        tracker_scores = self.obj_id_to_tracker_score_frame_wise[frame_idx]
        if len(new_det_obj_ids) > 0:
            new_det_scores = dict(
                zip(new_det_obj_ids.tolist(), det_scores[new_det_inds].tolist())
            )
            self.obj_id_to_score.update(new_det_scores)
            # Tracker scores aren't available for new objects.
            tracker_scores.update(new_det_scores)
            self.max_obj_id = int(new_det_obj_ids.max())
        for obj_id in obj_ids_newly_removed:
            self.obj_id_to_score[obj_id] = -1e4
            tracker_scores[obj_id] = -1e4
            self.obj_id_to_last_occluded.pop(obj_id, None)

        # Step 4: run_tracker_update_execution_phase
        if len(new_det_inds) > 0:
            self._tracker_add_new_objects(
                frame_idx, new_det_obj_ids, det_masks[new_det_inds]
            )
        for obj_id in obj_ids_newly_removed:
            self._tracker_remove_object(obj_id)

        # Step 5: build_outputs
        obj_id_to_mask: dict[int, NDArray[np.bool_]] = {}
        for ids, masks in [
            (obj_ids_prev, trk_masks),
            (new_det_obj_ids, _fill_holes_in_mask_scores(det_masks[new_det_inds])),
        ]:
            if len(masks) > 0:
                video_res_masks = _resize(
                    masks[:, None], self.video_height, self.video_width
                )[:, 0]
                obj_id_to_mask.update(zip(ids.tolist(), video_res_masks > 0))

        # This overwrites the -1e4 of the objects removed on this frame, as
        # upstream does.
        tracker_scores.update(
            zip(obj_ids_prev.tolist(), (1 / (1 + np.exp(-trk_scores))).tolist())
        )
        return obj_id_to_mask

    def _run_tracker_propagation(
        self, frame_idx: int, reverse: bool
    ) -> tuple[NDArray, NDArray]:
        """_propogate_tracker_one_frame_local_gpu: the low-res mask logits
        (N, H, W) and the object score logits (N,) of every tracked object."""
        masks, scores = [], []
        for tracker in self.trackers:
            out = tracker.propagate_one_frame(frame_idx, reverse, run_mem_encoder=False)
            masks.append(out["pred_masks"][:, 0])
            scores.append(out["object_score_logits"][:, 0])
        assert [i for t in self.trackers for i in t.obj_ids] == self.obj_ids.tolist()
        if not masks:
            return (
                np.zeros((0, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE), np.float32),
                np.zeros(0, np.float32),
            )
        return _fill_holes_in_mask_scores(np.concatenate(masks)), np.concatenate(scores)

    def _associate_det_trk(
        self,
        det_masks: NDArray,
        det_scores: NDArray,
        trk_masks: NDArray,
        trk_obj_ids: NDArray[np.int64],
    ) -> tuple[NDArray, NDArray, dict[int, NDArray], dict[int, int]]:
        """Match the detections with the tracked objects, returning the new
        detections, the unmatched objects, the objects each detection matches,
        and the high-confidence detection of each object."""
        if len(trk_masks) == 0:
            return np.arange(len(det_masks)), np.array([], np.int64), {}, {}
        trk_is_nonempty = (trk_masks > 0).any((1, 2))
        if len(det_masks) == 0:
            return np.array([], np.int64), trk_obj_ids[trk_is_nonempty], {}, {}

        ious = _mask_iou(det_masks > 0, trk_masks > 0)
        trk_is_matched = (ious >= TRK_ASSOC_IOU_THRESH).any(0)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty & ~trk_is_matched]
        is_new_det = (det_scores >= NEW_DET_THRESH) & ~(ious >= ASSOC_IOU_THRESH).any(1)
        det_to_matched_trk_obj_ids = {
            d: trk_obj_ids[ious[d] >= ASSOC_IOU_THRESH] for d in range(len(det_masks))
        }
        is_high_conf_and_iou = (
            (det_scores >= HIGH_CONF_THRESH)
            & ~is_new_det
            & (ious.max(1) >= HIGH_IOU_THRESH)
        )
        trk_id_to_max_iou_high_conf_det = {
            int(trk_obj_ids[ious[d].argmax()]): int(d)
            for d in np.flatnonzero(is_high_conf_and_iou)
        }
        return (
            np.flatnonzero(is_new_det),
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
        )

    def _process_hotstart(
        self,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, NDArray],
        new_det_obj_ids: NDArray,
        unmatched_trk_obj_ids: NDArray,
    ) -> set[int]:
        """Remove the objects that, within HOTSTART_DELAY frames of first
        appearing, go unmatched or duplicate an earlier object for long."""
        hotstart_diff = (
            frame_idx - HOTSTART_DELAY if not reverse else frame_idx + HOTSTART_DELAY
        )

        def is_within_hotstart(obj_id: int) -> bool:
            first_frame_idx = self.obj_first_frame_idx[obj_id]
            if reverse:
                return first_frame_idx < hotstart_diff
            return first_frame_idx > hotstart_diff

        for obj_id in new_det_obj_ids.tolist():
            self.obj_first_frame_idx.setdefault(obj_id, frame_idx)
        for obj_id in unmatched_trk_obj_ids.tolist():
            self.unmatched_frame_inds[obj_id].append(frame_idx)

        obj_ids_newly_removed: set[int] = set()
        for obj_id, frame_indices in self.unmatched_frame_inds.items():
            if obj_id in self.removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if len(frame_indices) >= HOTSTART_UNMATCH_THRESH and is_within_hotstart(
                obj_id
            ):
                obj_ids_newly_removed.add(obj_id)

        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            if len(matched_trk_obj_ids) < 2:
                continue
            matched = matched_trk_obj_ids.tolist()
            first_appear_obj_id = (max if reverse else min)(
                matched, key=lambda x: self.obj_first_frame_idx[x]
            )
            for obj_id in matched:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    self.overlap_pair_to_frame_inds[key].append(frame_idx)
        for (_, obj_id), frame_indices in self.overlap_pair_to_frame_inds.items():
            if obj_id in self.removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if is_within_hotstart(obj_id) and len(frame_indices) >= HOTSTART_DUP_THRESH:
                obj_ids_newly_removed.add(obj_id)

        self.removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed

    def _recondition_masklets(
        self,
        frame_idx: int,
        det_masks: NDArray,
        trk_id_to_max_iou_high_conf_det: dict[int, int],
        obj_ids: NDArray[np.int64],
        trk_scores: NDArray,
    ) -> None:
        """Replace the masks of confidently tracked objects with their
        high-confidence detections."""
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            # Upstream compares the logit with the threshold for a probability.
            if not trk_scores[obj_ids.tolist().index(trk_obj_id)] > HIGH_CONF_THRESH:
                continue
            new_mask = (
                _resize(
                    det_masks[None, det_idx : det_idx + 1],
                    INPUT_MASK_SIZE,
                    INPUT_MASK_SIZE,
                )
                > 0
            )[0, 0]
            for tracker in self.trackers:
                if trk_obj_id in tracker.obj_ids:
                    tracker.add_new_mask(frame_idx, trk_obj_id, new_mask)
                    tracker.propagate_in_video_preflight()

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        trk_masks: NDArray,
        obj_ids: NDArray[np.int64],
        obj_ids_newly_removed: set[int],
        reverse: bool,
    ) -> dict[int, int]:
        """Suppress in place the mask of an object that overlaps another and
        was occluded more recently, and return the frame each object was last
        occluded."""
        never_occluded, always_occluded = -1, 100000
        last_occluded_prev = np.array(
            [
                self.obj_id_to_last_occluded.get(
                    obj_id,
                    always_occluded
                    if obj_id in obj_ids_newly_removed
                    else never_occluded,
                )
                for obj_id in obj_ids.tolist()
            ],
            np.int64,
        )
        binary = trk_masks > 0
        to_suppress = np.zeros(len(obj_ids), np.bool_)
        if len(obj_ids) > 1:
            iou = _mask_iou(binary, binary)
            overlapping_pairs = np.triu(
                iou >= SUPPRESS_OVERLAPPING_BASED_ON_RECENT_OCCLUSION_THRESHOLD, k=1
            )
            last_i = last_occluded_prev[:, None]
            last_j = last_occluded_prev[None, :]
            cmp_op = np.less if reverse else np.greater
            suppress_i = overlapping_pairs & cmp_op(last_i, last_j) & (last_j > -1)
            suppress_j = overlapping_pairs & cmp_op(last_j, last_i) & (last_i > -1)
            to_suppress = suppress_i.any(1) | suppress_j.any(0)
        is_occluded_or_suppressed = ~binary.any((1, 2)) | to_suppress
        last_occluded_new = np.where(
            is_occluded_or_suppressed, frame_idx, last_occluded_prev
        )
        trk_masks[to_suppress] = -10.0
        return dict(zip(obj_ids.tolist(), last_occluded_new.tolist()))

    def _tracker_update_memories(self, frame_idx: int, low_res_masks: NDArray) -> None:
        """Encode the memories of every tracker on this frame, after resolving
        the overlaps across trackers."""
        high_res_masks = _resize(
            low_res_masks[:, None], INPUT_MASK_SIZE, INPUT_MASK_SIZE
        )
        high_res_masks = _suppress_object_pw_area_shrinkage(high_res_masks)
        # Mask areas stand in for the object scores.
        object_score_logits = np.where(
            (high_res_masks > 0).any((-1, -2)), 10.0, -10.0
        ).astype(np.float32)
        start = 0
        for tracker in self.trackers:
            end = start + len(tracker.obj_ids)
            tracker.update_memories(
                frame_idx, high_res_masks[start:end], object_score_logits[start:end]
            )
            start = end

    def _tracker_add_new_objects(
        self, frame_idx: int, new_obj_ids: NDArray[np.int64], new_obj_masks: NDArray
    ) -> None:
        """Track the new objects of a frame together, in a new tracker."""
        tracker = self._new_tracker()
        new_obj_masks = (
            _resize(new_obj_masks[:, None], INPUT_MASK_SIZE, INPUT_MASK_SIZE)[:, 0] > 0
        )
        for obj_id, mask in zip(new_obj_ids.tolist(), new_obj_masks):
            tracker.add_new_mask(frame_idx, obj_id, mask)
        tracker.propagate_in_video_preflight()
        self.trackers.append(tracker)

    def _tracker_remove_object(self, obj_id: int) -> None:
        for tracker in self.trackers:
            tracker.remove_object(obj_id)
        self.trackers = [tracker for tracker in self.trackers if tracker.obj_ids]

    # }}
