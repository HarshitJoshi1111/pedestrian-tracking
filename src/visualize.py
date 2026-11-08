"""Figures for the reports: precision-recall curves and one annotated validation frame.

- reports/figures/pr_curves.png: precision against recall (IoU 0.5) for every detection run
- reports/figures/<sequence>_<frame>.jpg: ground truth (green), ignored boxes (grey), missed people
  (orange) and detections above the best-F1 cut-off (red: false positive, blue: correct), with a legend on top
"""
import json

import cv2
import numpy as np
from matplotlib.figure import Figure

from src.evaluate_det import load_split
from src.metrics import FP, TP, match, match_frame, pr_curve
from src.mot import boxes, read_results
from src.params import DETECTIONS, RAW, REPORTS, load_params

FIGURES = REPORTS / "figures"
SAMPLE = ("MOT17-09", 200)
COLORS = {"gt": (60, 180, 60), "ignore": (150, 150, 150), "missed": (0, 140, 255), "tp": (255, 120, 0), "fp": (0, 0, 230)}


def pr_figure(gt: dict, runs: list[str]) -> None:
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    n_gt = sum(len(p) for p, _ in gt.values())
    for run in runs:
        parts = [match(read_results(DETECTIONS / run / f"{seq}.txt"), pos, ign, 0.5) for seq, (pos, ign) in gt.items()]
        precision, recall, _ = pr_curve(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), n_gt)
        ax.plot(recall, precision, label=run, linewidth=1.5)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Validation sequences, IoU 0.5")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curves.png", dpi=150)


def draw(image, box, color, thickness=2):
    x, y, w, h = (int(round(v)) for v in box)
    cv2.rectangle(image, (x - 1, y - 1), (x + w - 1, y + h - 1), color, thickness)


def legend(image, run: str, conf: float) -> None:
    items = [("ground truth, found", "gt"), ("ground truth, missed", "missed"), ("ignored person", "ignore"),
             ("correct detection", "tp"), ("false positive", "fp")]
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (255, 255, 255), -1)
    x = 8
    for text, key in items:
        cv2.rectangle(image, (x, 9), (x + 22, 21), COLORS[key], -1)
        cv2.putText(image, text, (x + 28, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        x += 34 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 14
    label = f"{run}, conf >= {conf:.2f}"
    width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(image, label, (image.shape[1] - width - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)


def sample_figure(gt: dict, run: str, conf: float) -> None:
    seq, frame = SAMPLE
    positives, ignore = gt[seq]
    pos, ign = positives[positives["frame"] == frame], ignore[ignore["frame"] == frame]
    dets = read_results(DETECTIONS / run / f"{seq}.txt")
    dets = dets[(dets["frame"] == frame) & (dets["score"] >= conf)]
    labels, gt_score = match_frame(boxes(dets), dets["score"].to_numpy(), boxes(pos), boxes(ign), 0.5)
    image = cv2.imread(str(RAW / seq / "img1" / f"{frame:06d}.jpg"))
    for box in boxes(ign):
        draw(image, box, COLORS["ignore"])
    for box, score in zip(boxes(pos), gt_score, strict=True):
        draw(image, box, COLORS["missed"] if np.isnan(score) else COLORS["gt"], 3)
    for box, label in zip(boxes(dets), labels, strict=True):
        if label in (TP, FP):
            draw(image, box, COLORS["tp" if label == TP else "fp"], 2)
    scale = 1280 / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    legend(image, run, conf)
    cv2.imwrite(str(FIGURES / f"{seq}_{frame:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    params = load_params()
    gt = load_split(params["split"]["val"])
    results = json.loads((REPORTS / "detection.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    pr_figure(gt, list(results))
    best = max(results, key=lambda r: results[r]["ap50"])
    sample_figure(gt, best, results[best]["best_f1"]["conf"])
    print(f"figures written, sample frame uses {best}")


if __name__ == "__main__":
    main()
