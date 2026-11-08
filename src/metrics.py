"""Detection metrics for one class: matching, average precision and recall by box group.

Matching follows COCO: within a frame, detections are taken from the highest score down and each
one is matched to the still-unmatched ground-truth box it overlaps most (IoU >= threshold). A
detection without a match that overlaps an ignore box is left out instead of counted as a false
positive. Because matching goes from high to low scores, the matches of the boxes above any score
cut-off do not depend on the boxes below it; so "recall at confidence c" is simply the share of
ground-truth boxes whose matching detection scored at least c.
"""
import numpy as np
import pandas as pd

from src.mot import boxes

TP, FP, IGNORED = 1, 0, -1
RECALL_POINTS = np.linspace(0, 1, 101)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n, 4) and every box in b (m, 4), boxes as x, y, w, h."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = w * h
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def match_frame(det: np.ndarray, scores: np.ndarray, gt: np.ndarray, ignore: np.ndarray, threshold: float):
    """Label each detection TP / FP / IGNORED and give, for each GT box, the score of its match (NaN if none)."""
    labels = np.full(len(det), FP)
    gt_score = np.full(len(gt), np.nan)
    iou_gt, iou_ignore = iou_matrix(det, gt), iou_matrix(det, ignore)
    for d in np.argsort(-scores, kind="stable"):
        free = np.isnan(gt_score) & (iou_gt[d] >= threshold) if len(gt) else np.zeros(0, bool)
        if free.any():
            g = np.flatnonzero(free)[np.argmax(iou_gt[d, free])]
            gt_score[g], labels[d] = scores[d], TP
        elif len(ignore) and iou_ignore[d].max() >= threshold:
            labels[d] = IGNORED
    return labels, gt_score


def match(dets: pd.DataFrame, gt: pd.DataFrame, ignore: pd.DataFrame, threshold: float):
    """match_frame over all frames of one sequence. Returns (det scores, det labels, GT match scores)."""
    det_groups, ign_groups = dict(list(dets.groupby("frame"))), dict(list(ignore.groupby("frame")))
    gt_score = np.full(len(gt), np.nan)
    gt_index = gt.reset_index(drop=True).groupby("frame").indices
    scores, labels = [], []
    for frame in sorted(set(det_groups) | set(gt_index)):
        d = det_groups.get(frame, dets.iloc[:0])
        rows = gt_index.get(frame, np.zeros(0, int))
        ign = ign_groups.get(frame, ignore.iloc[:0])
        s = d["score"].to_numpy(dtype=float)
        lab, gs = match_frame(boxes(d), s, boxes(gt.iloc[rows]), boxes(ign), threshold)
        gt_score[rows] = gs
        scores.append(s)
        labels.append(lab)
    return np.concatenate(scores or [np.zeros(0)]), np.concatenate(labels or [np.zeros(0, int)]), gt_score


def pr_curve(scores: np.ndarray, labels: np.ndarray, n_gt: int):
    """Precision and recall after each detection, from the highest score down (ignored ones left out)."""
    keep = labels != IGNORED
    order = np.argsort(-scores[keep], kind="stable")
    hits = labels[keep][order] == TP
    tp, fp = np.cumsum(hits), np.cumsum(~hits)
    return tp / np.maximum(tp + fp, 1), tp / max(n_gt, 1), scores[keep][order]


def average_precision(scores: np.ndarray, labels: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: mean of the interpolated precision at 101 recall points."""
    if n_gt == 0:
        return float("nan")
    precision, recall, _ = pr_curve(scores, labels, n_gt)
    if len(precision) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # best precision at this recall or higher
    index = np.searchsorted(recall, RECALL_POINTS, side="left")
    values = np.where(index < len(envelope), envelope[np.minimum(index, len(envelope) - 1)], 0.0)
    return float(values.mean())


def best_f1(scores: np.ndarray, labels: np.ndarray, n_gt: int) -> dict:
    """Operating point (confidence cut-off) with the highest F1."""
    precision, recall, ordered = pr_curve(scores, labels, n_gt)
    if len(precision) == 0:
        return {"conf": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    i = int(np.argmax(f1))
    return {"conf": float(ordered[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}


def recall_by_group(gt_score: np.ndarray, values: np.ndarray, bins: list[float], conf: float) -> list[dict]:
    """Recall at a confidence cut-off for GT boxes grouped by a property (visibility, height, ...)."""
    found = gt_score >= conf  # NaN (no match) compares False
    groups = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        inside = (values >= low) & (values < high)
        groups.append({"from": low, "to": high, "boxes": int(inside.sum()),
                       "recall": float(found[inside].mean()) if inside.any() else None})
    return groups
