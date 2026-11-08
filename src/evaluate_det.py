"""Stage 3: score every cached detection run on the validation sequences.

Writes reports/detection.json (all numbers) and reports/detection.md (summary tables):
AP at IoU 0.5, AP averaged over IoU 0.5:0.95 (rewards tight boxes), the best-F1 operating point,
and recall split by visibility and box height to show *which* pedestrians the detector misses.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import average_precision, best_f1, match, recall_by_group
from src.mot import read_gt, read_results, split_gt
from src.params import DETECTIONS, RAW, REPORTS, load_params


def load_split(sequences: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {seq: split_gt(read_gt(RAW / seq / "gt" / "gt.txt")) for seq in sequences}


def evaluate_run(run_dir: Path, gt: dict, cfg: dict) -> dict:
    thresholds = cfg["iou_thresholds"]
    pooled = {t: ([], [], []) for t in thresholds}  # scores, labels, GT match scores over all sequences
    per_sequence = {}
    for seq, (positives, ignore) in gt.items():
        dets = read_results(run_dir / f"{seq}.txt")
        for t in thresholds:
            scores, labels, gt_score = match(dets, positives, ignore, t)
            for store, values in zip(pooled[t], (scores, labels, gt_score), strict=True):
                store.append(values)
            if t == 0.5:
                per_sequence[seq] = {"boxes": len(positives), "ap50": round(average_precision(scores, labels, len(positives)), 4)}

    n_gt = sum(len(p) for p, _ in gt.values())
    joined = {t: [np.concatenate(v) for v in pooled[t]] for t in thresholds}
    ap = {t: average_precision(joined[t][0], joined[t][1], n_gt) for t in thresholds}
    scores, labels, gt_score = joined[0.5]
    point = best_f1(scores, labels, n_gt)
    positives = pd.concat([p for p, _ in gt.values()])
    conf = point["conf"]
    return {
        "ap50": round(ap[0.5], 4),
        "ap75": round(ap[0.75], 4),
        "ap50_95": round(float(np.mean(list(ap.values()))), 4),
        "max_recall": round(float(np.mean(~np.isnan(gt_score))), 4),
        "best_f1": {k: round(v, 4) for k, v in point.items()},
        "recall_by_visibility": recall_by_group(gt_score, positives["visibility"].to_numpy(), cfg["visibility_bins"], conf),
        "recall_by_height": recall_by_group(gt_score, positives["h"].to_numpy(), cfg["height_bins"], conf),
        "per_sequence": per_sequence,
    }


def markdown(results: dict) -> str:
    lines = ["# Detection baseline (COCO-pretrained YOLO, class person, no training)", "",
             "Validation sequences, boxes with IoU >= 0.5 count as found. F1, precision and recall at the confidence "
             "cut-off with the best F1. FPS = pre-processing + inference + NMS per frame.", "",
             "| Run | AP50 | AP75 | AP50:95 | Max recall | Conf | Precision | Recall | F1 | ms/frame | FPS |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        f, t = r["best_f1"], r.get("timing", {})
        lines.append(f"| {name} | {r['ap50']:.3f} | {r['ap75']:.3f} | {r['ap50_95']:.3f} | {r['max_recall']:.3f} | "
                     f"{f['conf']:.2f} | {f['precision']:.3f} | {f['recall']:.3f} | {f['f1']:.3f} | "
                     f"{t.get('ms_per_frame', '-')} | {t.get('fps', '-')} |")
    first = next(iter(results.values()))
    for key, title, unit in (("recall_by_visibility", "visibility (share of the person not occluded)", ""),
                             ("recall_by_height", "box height in pixels", " px")):
        labels = [_label(g, unit) for g in first[key]]
        lines += ["", f"## Recall by {title}", "", "| Run | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|",
                  "| GT boxes | " + " | ".join(f"{g['boxes']:,}" for g in first[key]) + " |"]
        for name, r in results.items():
            lines.append(f"| {name} | " + " | ".join("-" if g["recall"] is None else f"{g['recall']:.3f}" for g in r[key]) + " |")
    sequences = list(first["per_sequence"])
    lines += ["", "## AP50 per sequence", "", "| Run | " + " | ".join(sequences) + " |", "|---" * (len(sequences) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['per_sequence'][s]['ap50']:.3f}" for s in sequences) + " |")
    return "\n".join(lines) + "\n"


def _label(group: dict, unit: str) -> str:
    low, high = group["from"], group["to"]
    if unit:  # heights: last bin is open-ended
        return f"{low:g}+{unit}" if high >= 10000 else f"{low:g}-{high:g}{unit}"
    return f"{low:g}-{min(high, 1):g}"


def run_order(path: Path) -> tuple[str, int]:
    """'yolo11s-1280' -> ('yolo11s', 1280), so runs sort by model, then by image size."""
    model, _, imgsz = path.name.rpartition("-")
    return model, int(imgsz) if imgsz.isdigit() else 0


def main() -> None:
    params = load_params()
    gt = load_split(params["split"]["val"])
    results = {}
    runs = [p for p in DETECTIONS.iterdir() if p.is_dir()]
    for run_dir in sorted(runs, key=run_order):
        results[run_dir.name] = evaluate_run(run_dir, gt, params["evaluate"])
        timing = run_dir / "timing.json"
        if timing.exists():
            results[run_dir.name]["timing"] = json.loads(timing.read_text())
        print(f"{run_dir.name}: AP50 {results[run_dir.name]['ap50']:.3f}, AP50:95 {results[run_dir.name]['ap50_95']:.3f}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "detection.json").write_text(json.dumps(results, indent=2) + "\n")
    (REPORTS / "detection.md").write_text(markdown(results))


if __name__ == "__main__":
    main()
