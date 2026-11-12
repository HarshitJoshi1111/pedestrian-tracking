"""Reference: the ByteTrack implementation that ships with Ultralytics, on the same validation frames.

It runs the detector again (tracking happens inside model.track), with the library's default ByteTrack
settings. Comparing it with src/tracker.py checks that the own tracker is in the right range.
Output: data/tracks/ultralytics-bytetrack/<sequence>.txt
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.detect import PERSON, load_model, pick_device
from src.mot import RESULT_COLUMNS, write_results
from src.params import RAW, load_params
from src.track import TRACKS

NAME = "ultralytics-bytetrack"


def main() -> None:
    params = load_params()
    cfg = params["track"]["reference"]
    model, device = load_model(cfg["model"]), pick_device()
    times = []
    for seq in params["split"]["val"]:
        rows, start = [], time.time()
        # persist=True keeps the tracker from frame to frame (each frame is a separate image file);
        # a new predictor per sequence starts a fresh tracker
        model.predictor = None
        results = model.track(source=str(RAW / seq / "img1"), stream=True, persist=True, tracker="bytetrack.yaml",
                              imgsz=cfg["imgsz"], conf=cfg["conf"], classes=[PERSON], device=device, verbose=False)
        for result in results:
            if result.boxes.id is None:
                continue
            frame = int(Path(result.path).stem)
            for (x, y, w, h), track_id, score in zip(result.boxes.xywh.cpu().numpy(), result.boxes.id.int().tolist(),
                                                     result.boxes.conf.cpu().numpy(), strict=True):
                rows.append((frame, track_id, x - w / 2 + 1, y - h / 2 + 1, w, h, float(score)))
            times.append(sum(result.speed.values()))
        write_results(pd.DataFrame(rows, columns=RESULT_COLUMNS), TRACKS / NAME / f"{seq}.txt")
        print(f"{NAME} {seq}: {len(rows)} boxes, {time.time() - start:.0f} s", flush=True)
    info = {"detector": f"{Path(cfg['model']).stem}-{cfg['imgsz']}", "conf": cfg["conf"], "tracker": "bytetrack.yaml",
            "ms_per_frame_with_detector": round(sum(times) / len(times), 2)}
    (TRACKS / NAME / "run.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
