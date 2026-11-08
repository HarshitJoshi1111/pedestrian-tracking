"""MOTChallenge files: seqinfo.ini, ground truth (gt.txt) and detection/tracking results.

gt.txt has one box per line: frame, id, x, y, w, h, flag, class, visibility. Frames and pixel
coordinates are 1-based, (x, y) is the top-left corner. flag 0 marks boxes the benchmark does not
score. Class 1 is a pedestrian; classes 2 (person on vehicle), 7 (static person), 8 (distractor) and
12 (reflection) are people too, so a detector that finds them should not be punished: detections on
these boxes are ignored instead of counted as false positives.
"""
import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PEDESTRIAN = 1
IGNORE_CLASSES = (2, 7, 8, 12)
GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "flag", "cls", "visibility"]
RESULT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "score"]


@dataclass(frozen=True)
class SeqInfo:
    name: str
    frames: int
    fps: float
    width: int
    height: int


def read_seqinfo(path: Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.read(path)
    s = parser["Sequence"]
    return SeqInfo(s["name"], int(s["seqLength"]), float(s["frameRate"]), int(s["imWidth"]), int(s["imHeight"]))


def read_gt(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path, header=None, names=GT_COLUMNS)
    return gt.astype({"frame": int, "id": int, "flag": int, "cls": int})


def split_gt(gt: pd.DataFrame, min_visibility: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(boxes a detector must find, boxes where a detection is neither right nor wrong)."""
    pedestrian = gt["cls"] == PEDESTRIAN
    scored = pedestrian & (gt["flag"] == 1) & (gt["visibility"] >= min_visibility)
    ignore = ~scored & (pedestrian | gt["cls"].isin(IGNORE_CLASSES))
    return gt[scored].reset_index(drop=True), gt[ignore].reset_index(drop=True)


def read_results(path: Path) -> pd.DataFrame:
    """Detections or tracks in MOTChallenge format (frame, id, x, y, w, h, score, -1, -1, -1)."""
    df = pd.read_csv(path, header=None, usecols=range(7), names=RESULT_COLUMNS)
    return df.astype({"frame": int, "id": int})


def write_results(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[RESULT_COLUMNS].round({"x": 2, "y": 2, "w": 2, "h": 2, "score": 4})
    for col in ("a", "b", "c"):
        out[col] = -1
    out.to_csv(path, header=False, index=False)


def boxes(df: pd.DataFrame) -> np.ndarray:
    """(n, 4) array of x, y, w, h."""
    return df[["x", "y", "w", "h"]].to_numpy(dtype=float)
