"""Tracking metrics as used by the MOTChallenge benchmark: CLEAR MOT (MOTA, MOTP), IDF1, MT/ML.

- MOTA = 1 - (misses + false positives + ID switches) / ground-truth boxes. Matching is done per frame;
  a ground-truth person keeps its previous track if they still overlap (IoU >= 0.5), otherwise the
  Hungarian algorithm assigns one. An ID switch is counted when a person is matched to a different
  track than the last time. MOTA is dominated by detection errors.
- IDF1 matches whole trajectories once for the full sequence (each person to at most one track ID)
  and counts the boxes where the matched pair overlaps: it measures how long a track keeps its ID.
- MT / ML: people tracked for at least 80% / less than 20% of their life.

Before scoring, tracker boxes on ignored people (static persons, reflections, ...) are removed, as the
official evaluation does. The results agree with py-motmetrics (checked on the validation sequences).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.metrics import iou_matrix
from src.mot import boxes

IOU = 0.5


def _assign(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Maximum-IoU one-to-one matching, pairs below the threshold excluded."""
    if iou.size == 0:
        return []
    cost = np.where(iou >= threshold, 1 - iou, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= threshold]


def remove_ignored(tracks: pd.DataFrame, positives: pd.DataFrame, ignore: pd.DataFrame, threshold: float = IOU):
    """Drop tracker boxes that are matched to an ignored person when matched against all people at once."""
    if ignore.empty or tracks.empty:
        return tracks.reset_index(drop=True)
    people = pd.concat([positives.assign(_ignored=False), ignore.assign(_ignored=True)], ignore_index=True)
    by_frame = people.groupby("frame")
    drop = []
    for frame, hyp in tracks.groupby("frame"):
        if frame not in by_frame.groups:
            continue
        gt = by_frame.get_group(frame)
        for h, g in _assign(iou_matrix(boxes(hyp), boxes(gt)), threshold):
            if gt["_ignored"].iloc[g]:
                drop.append(hyp.index[h])
    return tracks.drop(index=drop).reset_index(drop=True)


def clear_mot(gt: pd.DataFrame, hyp: pd.DataFrame, threshold: float = IOU) -> dict:
    last_match: dict[int, int] = {}  # person id -> track id of their most recent match
    matched_frames: dict[int, int] = defaultdict(int)
    misses = false_positives = switches = 0
    overlaps: list[float] = []
    gt_frames, hyp_frames = dict(list(gt.groupby("frame"))), dict(list(hyp.groupby("frame")))
    for frame in sorted(set(gt_frames) | set(hyp_frames)):
        g = gt_frames.get(frame, gt.iloc[:0])
        h = hyp_frames.get(frame, hyp.iloc[:0])
        g_ids, h_ids = g["id"].to_numpy(), h["id"].to_numpy()
        iou = iou_matrix(boxes(g), boxes(h))
        pairs = []
        # 1. keep last correspondences that still overlap
        h_pos = {hid: j for j, hid in enumerate(h_ids)}
        for i, gid in enumerate(g_ids):
            j = h_pos.get(last_match.get(gid))
            if j is not None and iou[i, j] >= threshold and all(j != b for _, b in pairs):
                pairs.append((i, j))
        # 2. Hungarian on the rest
        used_g, used_h = {i for i, _ in pairs}, {j for _, j in pairs}
        rest_g = [i for i in range(len(g_ids)) if i not in used_g]
        rest_h = [j for j in range(len(h_ids)) if j not in used_h]
        for a, b in _assign(iou[np.ix_(rest_g, rest_h)], threshold):
            i, j = rest_g[a], rest_h[b]
            if g_ids[i] in last_match and last_match[g_ids[i]] != h_ids[j]:
                switches += 1
            pairs.append((i, j))
        for i, j in pairs:
            last_match[g_ids[i]] = h_ids[j]
            matched_frames[g_ids[i]] += 1
            overlaps.append(iou[i, j])
        misses += len(g_ids) - len(pairs)
        false_positives += len(h_ids) - len(pairs)

    n_gt = len(gt)
    lifetime = gt.groupby("id").size()
    coverage = pd.Series(matched_frames).reindex(lifetime.index, fill_value=0) / lifetime
    return {
        "mota": 1 - (misses + false_positives + switches) / n_gt if n_gt else float("nan"),
        "motp": float(np.mean(overlaps)) if overlaps else float("nan"),
        "gt_boxes": n_gt, "matches": len(overlaps), "misses": misses, "false_positives": false_positives,
        "id_switches": switches, "people": len(lifetime),
        "mostly_tracked": int((coverage >= 0.8).sum()), "mostly_lost": int((coverage < 0.2).sum()),
    }


def identity(gt: pd.DataFrame, hyp: pd.DataFrame, threshold: float = IOU) -> dict:
    """IDF1, ID precision and ID recall from one global matching of person IDs to track IDs."""
    overlap: dict[tuple[int, int], int] = defaultdict(int)
    hyp_frames = dict(list(hyp.groupby("frame")))
    for frame, g in gt.groupby("frame"):
        h = hyp_frames.get(frame)
        if h is None:
            continue
        iou = iou_matrix(boxes(g), boxes(h))
        for i, j in zip(*np.nonzero(iou >= threshold), strict=True):
            overlap[(g["id"].iloc[i], h["id"].iloc[j])] += 1
    id_tp = 0
    if overlap:
        g_ids = sorted({g for g, _ in overlap})
        h_ids = sorted({h for _, h in overlap})
        gi, hi = {v: k for k, v in enumerate(g_ids)}, {v: k for k, v in enumerate(h_ids)}
        counts = np.zeros((len(g_ids), len(h_ids)))
        for (g, h), n in overlap.items():
            counts[gi[g], hi[h]] = n
        rows, cols = linear_sum_assignment(-counts)
        id_tp = int(counts[rows, cols].sum())
    n_gt, n_hyp = len(gt), len(hyp)
    return {"idf1": 2 * id_tp / (n_gt + n_hyp) if n_gt + n_hyp else float("nan"),
            "idp": id_tp / n_hyp if n_hyp else float("nan"), "idr": id_tp / n_gt if n_gt else float("nan")}


def evaluate_sequence(tracks: pd.DataFrame, positives: pd.DataFrame, ignore: pd.DataFrame) -> dict:
    hyp = remove_ignored(tracks, positives, ignore)
    return {**clear_mot(positives, hyp), **identity(positives, hyp), "track_ids": int(hyp["id"].nunique())}


def combine(per_sequence: dict[str, dict]) -> dict:
    """Overall scores over several sequences, from the summed counts (as the benchmark does)."""
    total = {k: sum(r[k] for r in per_sequence.values())
             for k in ("gt_boxes", "matches", "misses", "false_positives", "id_switches", "people",
                       "mostly_tracked", "mostly_lost", "track_ids")}
    n_hyp = total["matches"] + total["false_positives"]
    id_tp = sum(r["idr"] * r["gt_boxes"] for r in per_sequence.values())
    motp = sum(r["motp"] * r["matches"] for r in per_sequence.values()) / max(total["matches"], 1)
    return {"mota": 1 - (total["misses"] + total["false_positives"] + total["id_switches"]) / total["gt_boxes"],
            "motp": motp, "idf1": 2 * id_tp / (total["gt_boxes"] + n_hyp), "idp": id_tp / max(n_hyp, 1),
            "idr": id_tp / total["gt_boxes"], **total}
