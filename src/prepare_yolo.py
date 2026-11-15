"""Stage 6: training data for the detector in the YOLO format, from the training sequences only.

    data/yolo/images/{train,holdout}/<sequence>_<frame>.jpg   hard links to data/raw (no extra disk space)
    data/yolo/labels/{train,holdout}/<sequence>_<frame>.txt   one line per person: 0 cx cy w h (0..1)
    data/yolo/mot17.yaml                                       dataset description for Ultralytics

The validation sequences are not touched. Model selection during training uses a holdout instead:
the last part of every training sequence, separated from the training part by a gap of frames so
that no near-identical neighbour frame ends up on both sides.
"""
import os
import shutil

import pandas as pd

from src.mot import PEDESTRIAN, read_gt, read_seqinfo
from src.params import RAW, ROOT, load_params

YOLO_DIR = ROOT / "data" / "yolo"


def split_frames(n_frames: int, step: int, holdout: float, gap: int) -> tuple[list[int], list[int]]:
    """(training frames, holdout frames), both every `step`-th frame."""
    cut = int(n_frames * (1 - holdout))
    return list(range(1, cut - gap + 1, step)), list(range(cut + 1, n_frames + 1, step))


def label_boxes(gt: pd.DataFrame, extra_classes: list[int]) -> pd.DataFrame:
    """Scored pedestrians plus the extra person classes (any flag)."""
    pedestrian = (gt["cls"] == PEDESTRIAN) & (gt["flag"] == 1)
    return gt[pedestrian | gt["cls"].isin(extra_classes)]


def yolo_lines(boxes: pd.DataFrame, width: int, height: int) -> list[str]:
    """MOT boxes (1-based pixels, may reach outside the image) -> clipped, normalised YOLO lines."""
    lines = []
    for x, y, w, h in boxes[["x", "y", "w", "h"]].to_numpy(dtype=float):
        x1, y1 = min(max(x - 1, 0), width), min(max(y - 1, 0), height)
        x2, y2 = min(max(x - 1 + w, 0), width), min(max(y - 1 + h, 0), height)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            lines.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def link(source, target) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    params = load_params()
    cfg = params["train"]
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    counts = {"train": [0, 0], "holdout": [0, 0]}  # images, boxes
    for seq in params["split"]["train"]:
        info = read_seqinfo(RAW / seq / "seqinfo.ini")
        boxes = label_boxes(read_gt(RAW / seq / "gt" / "gt.txt"), cfg["extra_classes"])
        by_frame = dict(list(boxes.groupby("frame")))
        parts = dict(zip(("train", "holdout"), split_frames(info.frames, cfg["frame_step"], cfg["holdout"], cfg["gap"]),
                         strict=True))
        for part, frames in parts.items():
            (YOLO_DIR / "images" / part).mkdir(parents=True, exist_ok=True)
            (YOLO_DIR / "labels" / part).mkdir(parents=True, exist_ok=True)
            for frame in frames:
                name = f"{seq}_{frame:06d}"
                link(RAW / seq / "img1" / f"{frame:06d}.jpg", YOLO_DIR / "images" / part / f"{name}.jpg")
                lines = yolo_lines(by_frame.get(frame, boxes.iloc[:0]), info.width, info.height)
                (YOLO_DIR / "labels" / part / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                counts[part][0] += 1
                counts[part][1] += len(lines)
    (YOLO_DIR / "mot17.yaml").write_text(f"path: {YOLO_DIR}\ntrain: images/train\nval: images/holdout\nnames:\n  0: person\n")
    for part, (images, n_boxes) in counts.items():
        print(f"{part}: {images} images, {n_boxes} boxes")


if __name__ == "__main__":
    main()
