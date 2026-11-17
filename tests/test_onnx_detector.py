import numpy as np
import pytest

from src.onnx_detector import decode, letterbox, nms


def test_letterbox_keeps_aspect_ratio_and_centres():
    image = np.zeros((1080, 1920, 3), np.uint8)
    padded, scale, (left, top) = letterbox(image, 736, 1280)
    assert padded.shape == (736, 1280, 3)
    assert scale == pytest.approx(1280 / 1920) and left == 0 and top == 8  # 720 px high, 8 px grey above and below
    assert padded[0, 0, 0] == 114 and padded[100, 100, 0] == 0


def test_nms_removes_overlapping_lower_scores():
    boxes = np.array([[0, 0, 10, 20], [1, 0, 11, 20], [50, 0, 60, 20]], float)
    keep = nms(boxes, np.array([0.8, 0.9, 0.7]), iou_threshold=0.7)
    assert keep.tolist() == [1, 2]


def test_decode_keeps_boxes_whose_best_class_is_person():
    # columns: candidates; rows: cx, cy, w, h, score person, score car
    output = np.array([[50, 150, 250], [50, 50, 50], [10, 10, 10], [20, 20, 20],
                       [0.9, 0.3, 0.05], [0.1, 0.6, 0.01]], float)
    boxes, scores = decode(output, conf=0.1)
    assert boxes.tolist() == [[45, 40, 55, 60]]  # the second is a car, the third too unsure
    assert scores.tolist() == [0.9]
