import numpy as np
import pandas as pd
import pytest

from src.metrics import FP, IGNORED, TP, average_precision, best_f1, iou_matrix, match, match_frame, recall_by_group


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]], float)
    assert iou_matrix(a, b) == pytest.approx(np.array([[1.0, 50 / 150, 0.0]]))
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_match_frame_labels():
    gt = np.array([[0, 0, 10, 10], [100, 0, 10, 10]], float)
    ignore = np.array([[200, 0, 10, 10]], float)
    det = np.array([[1, 0, 10, 10],     # matches GT 0
                    [0, 0, 10, 10],     # better box but lower score: GT 0 is taken -> duplicate, false positive
                    [201, 0, 10, 10],   # on an ignored person
                    [400, 0, 10, 10]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels, gt_score = match_frame(det, scores, gt, ignore, 0.5)
    assert labels.tolist() == [TP, FP, IGNORED, FP]
    assert gt_score[0] == 0.9 and np.isnan(gt_score[1])


def test_average_precision_perfect_and_empty():
    assert average_precision(np.array([0.9, 0.8]), np.array([TP, TP]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9]), np.array([TP]), 2) == pytest.approx(51 / 101)  # recall stops at 0.5
    assert average_precision(np.zeros(0), np.zeros(0, int), 3) == 0.0
    assert np.isnan(average_precision(np.zeros(0), np.zeros(0, int), 0))


def test_average_precision_uses_precision_envelope_and_skips_ignored():
    # order by score: TP, FP, TP -> precision 1, 0.5, 0.67; recall 0.5, 0.5, 1.0
    scores, labels = np.array([0.9, 0.8, 0.7, 0.75]), np.array([TP, FP, TP, IGNORED])
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert average_precision(scores, labels, 2) == pytest.approx(expected)


def test_best_f1():
    point = best_f1(np.array([0.9, 0.8, 0.7]), np.array([TP, FP, TP]), 2)
    assert point["conf"] == 0.7 and point["recall"] == 1.0 and point["f1"] == pytest.approx(0.8)


def frame_df(rows, cols=("frame", "x", "y", "w", "h")):
    return pd.DataFrame(rows, columns=list(cols))


def test_match_over_frames_and_recall_by_group():
    gt = frame_df([(1, 0, 0, 10, 10), (1, 50, 0, 10, 40), (2, 0, 0, 10, 10)])
    dets = pd.DataFrame({"frame": [1, 2, 3], "x": [0, 0, 0], "y": [0, 0, 0], "w": [10, 10, 10], "h": [10, 10, 10],
                         "score": [0.9, 0.3, 0.5]})
    scores, labels, gt_score = match(dets, gt, frame_df([]), 0.5)
    assert labels.tolist() == [TP, TP, FP]  # frame 3 has no ground truth
    assert gt_score[0] == 0.9 and np.isnan(gt_score[1]) and gt_score[2] == 0.3
    groups = recall_by_group(gt_score, gt["h"].to_numpy(), [0, 20, 100], conf=0.5)
    assert groups == [{"from": 0, "to": 20, "boxes": 2, "recall": 0.5}, {"from": 20, "to": 100, "boxes": 1, "recall": 0.0}]
