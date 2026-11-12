"""Stage 2: run pretrained detectors over the frames and cache the person boxes.

Boxes are saved in MOTChallenge format (data/detections/<model>-<imgsz>/<sequence>.txt), so the AP
evaluation and, later, the tracker can be rerun in seconds without running the detector again.
The COCO-pretrained models already know the class "person" (class 0); nothing is trained here.

    python -m src.detect                                    all models and sizes on the validation sequences
    python -m src.detect --split train --imgsz 1280         training sequences (used to tune the tracker)
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import pandas as pd

from src.mot import RESULT_COLUMNS, write_results
from src.params import DETECTIONS, RAW, ROOT, load_params

WEIGHTS = ROOT / "models" / "weights"
PERSON = 0  # COCO class id
WARMUP = 5  # first frames are slower (model loading, memory allocation) and are left out of the timing


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name: str):
    from ultralytics import YOLO

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(WEIGHTS):  # the weights are downloaded into the current directory
        return YOLO(name)


def run_name(model: str, imgsz: int) -> str:
    return f"{Path(model).stem}-{imgsz}"


def detect_sequence(model, frames_dir: Path, imgsz: int, conf: float, max_det: int, device: str):
    """(detections as a DataFrame, milliseconds per frame for pre-processing + inference + NMS)."""
    rows, times = [], []
    results = model.predict(source=str(frames_dir), stream=True, imgsz=imgsz, conf=conf, classes=[PERSON],
                            max_det=max_det, device=device, verbose=False)
    for result in results:
        frame = int(Path(result.path).stem)
        xyxy = result.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, result.boxes.conf.cpu().numpy(), strict=True):
            # +1: MOTChallenge pixel coordinates start at 1, the model's at 0
            rows.append((frame, -1, x1 + 1, y1 + 1, x2 - x1, y2 - y1, float(score)))
        times.append(sum(result.speed.values()))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), times


def main() -> None:
    params = load_params()
    cfg = params["detect"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=list(params["split"]))
    parser.add_argument("--models", nargs="+", default=cfg["models"])
    parser.add_argument("--imgsz", nargs="+", type=int, default=cfg["imgsz"])
    args = parser.parse_args()
    device = pick_device()
    for model_name in args.models:
        model = load_model(model_name)
        for imgsz in args.imgsz:
            out = DETECTIONS / run_name(model_name, imgsz)
            all_times = []
            for seq in params["split"][args.split]:
                if (out / f"{seq}.txt").exists():
                    continue  # cached already
                start = time.time()
                dets, times = detect_sequence(model, RAW / seq / "img1", imgsz, cfg["conf"], cfg["max_det"], device)
                write_results(dets, out / f"{seq}.txt")
                all_times += times[WARMUP:]
                print(f"{out.name} {seq}: {len(dets)} boxes, {time.time() - start:.0f} s", flush=True)
            if all_times and args.split == "val":  # speed is reported on the validation frames
                ms = sum(all_times) / len(all_times)
                timing = {"model": model_name, "imgsz": imgsz, "device": device, "frames": len(all_times),
                          "ms_per_frame": round(ms, 2), "fps": round(1000 / ms, 1)}
                (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    main()
