import numpy as np

from video_onnx import _mask_iou
from video_onnx import _nms_masks


def test_empty_masks() -> None:
    empty = np.zeros((0, 4, 4), np.float32)
    masks = np.ones((2, 4, 4), np.float32)
    assert _mask_iou(empty > 0, masks > 0).shape == (0, 2)
    assert _mask_iou(masks > 0, empty > 0).shape == (2, 0)
    assert _nms_masks(np.zeros(0, np.float32), empty, 0.1).tolist() == []


def test_nms_masks() -> None:
    masks = np.zeros((3, 4, 4), np.float32)
    masks[:2, :2] = 1  # two duplicates
    masks[2, 2:] = 1
    probs = np.array([0.6, 0.9, 0.7], np.float32)
    assert _nms_masks(probs, masks, 0.1).tolist() == [1, 2]


if __name__ == "__main__":
    test_empty_masks()
    test_nms_masks()
    print("ok")
