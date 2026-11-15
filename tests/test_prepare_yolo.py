import pandas as pd
import pytest

from src.prepare_yolo import label_boxes, split_frames, yolo_lines


def test_split_frames_leaves_a_gap_before_the_holdout():
    train, holdout = split_frames(100, step=3, holdout=0.2, gap=10)
    assert train[0] == 1 and max(train) <= 70 and train[1] - train[0] == 3
    assert holdout[0] == 81 and max(holdout) <= 100
    assert not set(train) & set(holdout)


def test_label_boxes_keeps_scored_pedestrians_and_extra_person_classes():
    gt = pd.DataFrame({"frame": [1] * 5, "id": range(5), "x": 1.0, "y": 1.0, "w": 10.0, "h": 20.0,
                       "flag": [1, 0, 0, 0, 0], "cls": [1, 1, 7, 8, 3], "visibility": 1.0})
    assert label_boxes(gt, [2, 7])["id"].tolist() == [0, 2]


def test_yolo_lines_are_clipped_and_normalised():
    boxes = pd.DataFrame({"x": [101.0, -49.0, 300.0], "y": [51.0, 1.0, 1.0], "w": [20.0, 100.0, 50.0],
                          "h": [40.0, 40.0, 40.0]})
    lines = yolo_lines(boxes, width=200, height=100)
    assert len(lines) == 2  # the third box is completely outside the image
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0" and float(cx) == pytest.approx(110 / 200) and float(cy) == pytest.approx(70 / 100)
    assert float(w) == pytest.approx(0.1) and float(h) == pytest.approx(0.4)
    assert float(lines[1].split()[3]) == pytest.approx(50 / 200)  # left part cut off at the border
