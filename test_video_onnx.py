import numpy as np

from infer_video_onnx import draw
from track_onnx import _resize
from video_onnx import VideoInference
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
    test_propagation_history()
    test_draw()
    print("ok")
