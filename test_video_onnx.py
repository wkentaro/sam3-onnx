import numpy as np

from infer_video_onnx import draw
from track_onnx import FAR_OLD_FRAMES
from track_onnx import MAX_OBJ_PTRS_IN_ENCODER
from track_onnx import Tracker
from track_onnx import _new_output_dict
from track_onnx import _pad_memory_attention_inputs
from track_onnx import _resize
from video_onnx import VideoInference
from video_onnx import _is_argmax
from video_onnx import _mask_iou
from video_onnx import _nms_masks


def test_empty_masks() -> None:
    empty = np.zeros((0, 4, 4), np.float32)
    masks = np.ones((2, 4, 4), np.float32)
    assert _mask_iou(empty > 0, masks > 0).shape == (0, 2)
    assert _mask_iou(masks > 0, empty > 0).shape == (2, 0)
    assert _nms_masks(np.zeros(0, np.float32), empty, 0.1).tolist() == []
    assert _resize(empty[:, None], 8, 6).shape == (0, 1, 8, 6)


def test_nms_masks() -> None:
    masks = np.zeros((3, 4, 4), np.float32)
    masks[:2, :2] = 1  # two duplicates
    masks[2, 2:] = 1
    probs = np.array([0.6, 0.9, 0.7], np.float32)
    assert _nms_masks(probs, masks, 0.1).tolist() == [1, 2]


def test_is_argmax() -> None:
    scores = np.random.default_rng(0).integers(0, 3, (4, 5, 6))  # many ties
    expected = scores.argmax(0) == np.arange(4)[:, None, None]
    assert (_is_argmax(scores) == expected).all()


def test_propagation_history() -> None:
    video = object.__new__(VideoInference)
    video.num_frames, video.video_height, video.video_width = 10, 4, 4
    video.reset_state()
    video.frames_with_outputs.add(0)
    video._add_action_history("remove", None, [0])
    assert video._parse_action_history_for_propagation() == ("propagation_full", None)

    # A propagation from frame 0 that stopped at frame 4
    video._add_action_history("propagation_full", 0, None)
    video.cached_frame_outputs.update({i: {} for i in range(5)})
    assert len(list(video.propagate_in_video(max_frame_num_to_track=4))) == 5
    try:
        next(video.propagate_in_video(start_frame_idx=5))
    except RuntimeError:
        pass
    else:
        raise AssertionError("fetching uncached frames should raise")

    # Hotstart removes an object through _tracker_remove_object.
    video.cached_frame_outputs[0] = {7: np.ones((4, 4), np.bool_)}
    video._tracker_remove_object(7)
    assert video.cached_frame_outputs[0] == {}

    # Forward from the last frame covers one frame, so reverse runs in full.
    video.action_history = []
    video._add_action_history("propagation_full", 9, None)
    assert video._parse_action_history_for_propagation() == ("propagation_full", None)
    video._add_action_history("propagation_full", 9, None, reverse=True)
    assert video._parse_action_history_for_propagation()[0] == "propagation_fetch"


def test_trim_far_old_memory() -> None:
    far = FAR_OLD_FRAMES * MAX_OBJ_PTRS_IN_ENCODER
    tracker = object.__new__(Tracker)
    tracker.num_frames = far + 3
    tracker.output_dict = _new_output_dict()
    tracker.output_dict_per_obj = {0: _new_output_dict()}
    for frame_idx in [1, far + 2]:
        out = {
            "maskmem_features": np.ones((1, 1), np.float32),
            "pred_masks": np.ones((1, 1, 2, 2), np.float32),
            "obj_ptr": np.ones((1, 2), np.float32),
            "object_score_logits": np.ones((1, 1), np.float32),
            "iou_score": np.ones((1, 1), np.float32),
            "eff_iou_score": 1.0,
        }
        tracker.output_dict["non_cond_frame_outputs"][frame_idx] = out
        tracker._add_output_per_object(frame_idx, out, "non_cond_frame_outputs")

    def trimmed() -> list[list[bool]]:
        return [
            [out["maskmem_features"] is None for out in outputs.values()]
            for output_dict in [tracker.output_dict, tracker.output_dict_per_obj[0]]
            for outputs in [output_dict["non_cond_frame_outputs"]]
        ]

    tracker._trim_far_old_memory(far + 1, reverse=False)
    assert trimmed() == [[True, False]] * 2
    # The trimmed frame keeps its mask but is out of the memory selection.
    assert tracker.output_dict["non_cond_frame_outputs"][1]["pred_masks"] is not None
    assert 1 not in tracker._frame_filter(tracker.output_dict, False, far + 2)
    tracker._trim_far_old_memory(2, reverse=True)
    assert trimmed() == [[True, True]] * 2

    # Tracking next to a trimmed frame leaves its memory out.
    cond = {**tracker.output_dict["non_cond_frame_outputs"][far + 2]}
    cond["maskmem_features"] = np.ones((1, 1), np.float32)
    tracker.output_dict["cond_frame_outputs"][far + 2] = cond
    tracker._features = lambda frame_idx: [np.zeros((1, 1), np.float32)]
    tracker._sessions = {}
    inputs = {}
    tracker._run = lambda name, **kwargs: (
        inputs.update(kwargs) or {"pix_feat_with_mem": kwargs["image_embeddings"]}
    )
    tracker._prepare_memory_conditioned_features(
        frame_idx=0,
        is_init_cond_frame=False,
        output_dict=tracker.output_dict,
        track_in_reverse=True,
        use_prev_mem_frame=True,
        batch_size=1,
    )
    assert len(inputs["memory"]) == 1  # the conditioning frame alone


def test_pad_memory_attention_inputs() -> None:
    inputs = _pad_memory_attention_inputs(
        memory=np.ones((3, 64, 72, 72), np.float32),
        memory_tpos=np.array([6, 5, 4]),
        obj_ptrs=np.ones((5, 256), np.float32),
        obj_ptr_tpos=np.arange(5, dtype=np.float32),
    )
    assert inputs["memory"].shape == (7, 64, 72, 72)
    assert inputs["memory_tpos"].tolist() == [6, 5, 4, 0, 0, 0, 0]
    assert inputs["memory_tpos"].dtype == np.int64
    assert inputs["obj_ptrs"].shape == (16, 256)
    assert inputs["obj_ptr_tpos"].shape == (16,)
    # The padding is marked invalid, and holds nothing.
    assert inputs["memory_valid"].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert inputs["obj_ptr_valid"].tolist() == [1] * 5 + [0] * 11
    assert not inputs["memory"][3:].any() and not inputs["obj_ptrs"][5:].any()
    # No pointers at all, as reverse tracking from the last frame has.
    empty = _pad_memory_attention_inputs(
        memory=np.ones((1, 64, 72, 72), np.float32),
        memory_tpos=np.array([6]),
        obj_ptrs=np.zeros((0, 256), np.float32),
        obj_ptr_tpos=np.zeros(0, np.float32),
    )
    assert not empty["obj_ptr_valid"].any()


def test_draw() -> None:
    frame = np.zeros((4, 4, 3), np.uint8)
    masks = np.zeros((2, 4, 4), np.bool_)
    masks[0, 0], masks[1, 1] = True, True
    both = draw(frame, masks, [0, 300])
    alone = draw(frame, masks[1:], [300])
    # Object 300 keeps its color once object 0 is gone.
    assert (both[1] == alone[1]).all() and both[1].any()


if __name__ == "__main__":
    test_empty_masks()
    test_nms_masks()
    test_is_argmax()
    test_propagation_history()
    test_trim_far_old_memory()
    test_pad_memory_attention_inputs()
    test_draw()
    print("ok")
