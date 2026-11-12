"""Stage 4: run the tracker on the cached detections.

    python -m src.track          the experiments from params.yaml on the validation sequences
    python -m src.track --tune   grid search of tracker settings on the training sequences

The tracker settings are tuned on the training sequences and only then run on the validation
sequences, so the validation scores are not the result of picking the best of many tries on them.
Tracks go to data/tracks/<experiment>/<sequence>.txt (MOTChallenge format).
"""
import argparse
import itertools
import json
import time
from dataclasses import fields

import numpy as np
import pandas as pd

from src.evaluate_det import load_split
from src.mot import RESULT_COLUMNS, boxes, read_results, read_seqinfo, write_results
from src.params import DETECTIONS, RAW, REPORTS, ROOT, load_params
from src.track_metrics import combine, evaluate_sequence
from src.tracker import Tracker, TrackerConfig

TRACKS = ROOT / "data" / "tracks"
CONFIG_KEYS = {f.name for f in fields(TrackerConfig)}


def settings(track_params: dict, overrides: dict | None = None) -> tuple[TrackerConfig, str, int]:
    """(tracker config, detection run, interpolation gap) from params.yaml plus experiment overrides."""
    merged = {**track_params["tracker"], "detector": track_params["detector"],
              "interpolate_gap": track_params["interpolate_gap"], **(overrides or {})}
    config = TrackerConfig(**{k: v for k, v in merged.items() if k in CONFIG_KEYS})
    return config, merged["detector"], merged["interpolate_gap"]


def track_sequence(dets: pd.DataFrame, n_frames: int, config: TrackerConfig, fps: float) -> tuple[pd.DataFrame, float]:
    """Tracks of one sequence and the tracker's time per frame in milliseconds (detector not included)."""
    if config.low_thresh is not None:
        dets = dets[dets["score"] >= config.low_thresh]
    else:
        dets = dets[dets["score"] >= config.high_thresh]
    by_frame = dets.groupby("frame").indices
    tracker, rows = Tracker(config, fps), []
    start = time.perf_counter()
    for frame in range(1, n_frames + 1):
        d = dets.iloc[by_frame.get(frame, [])]
        for track_id, box, score in tracker.update(boxes(d), d["score"].to_numpy()):
            rows.append((frame, track_id, *box, score))
    ms = (time.perf_counter() - start) * 1000 / n_frames
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), ms


def interpolate(tracks: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Fill gaps of up to max_gap frames inside a track by linear interpolation (offline post-processing)."""
    if max_gap <= 0 or tracks.empty:
        return tracks
    added = []
    for track_id, t in tracks.sort_values("frame").groupby("id"):
        frames, values = t["frame"].to_numpy(), t[["x", "y", "w", "h"]].to_numpy()
        for k in np.flatnonzero((np.diff(frames) > 1) & (np.diff(frames) <= max_gap + 1)):
            f0, f1 = frames[k], frames[k + 1]
            for f in range(f0 + 1, f1):
                w = (f - f0) / (f1 - f0)
                added.append((f, track_id, *((1 - w) * values[k] + w * values[k + 1]), 0.0))
    filled = pd.concat([tracks, pd.DataFrame(added, columns=RESULT_COLUMNS)], ignore_index=True)
    return filled.sort_values(["frame", "id"]).reset_index(drop=True)


def run(sequences: list[str], config: TrackerConfig, detector: str, gap: int) -> tuple[dict[str, pd.DataFrame], float]:
    results, times = {}, []
    for seq in sequences:
        info = read_seqinfo(RAW / seq / "seqinfo.ini")
        tracks, ms = track_sequence(read_results(DETECTIONS / detector / f"{seq}.txt"), info.frames, config, info.fps)
        results[seq] = interpolate(tracks, gap)
        times.append(ms)
    return results, float(np.mean(times))


def score(results: dict[str, pd.DataFrame], gt: dict) -> dict:
    return combine({seq: evaluate_sequence(tracks, *gt[seq]) for seq, tracks in results.items()})


def tune(params: dict) -> None:
    track_params, cfg = params["track"], params["track"]["tune"]
    gt = load_split(params["split"]["train"])
    names, grid = list(cfg["grid"]), list(itertools.product(*cfg["grid"].values()))
    rows = []
    for values in grid:
        overrides = dict(zip(names, values, strict=True))
        config, detector, gap = settings(track_params, overrides)
        results, _ = run(params["split"]["train"], config, detector, gap)
        s = score(results, gt)
        rows.append({**overrides, **{k: s[k] for k in ("mota", "idf1", "id_switches")}})
        print(f"{overrides}: MOTA {s['mota']:.3f}, IDF1 {s['idf1']:.3f}, IDsw {s['id_switches']}", flush=True)
    table = pd.DataFrame(rows).sort_values(cfg["objective"], ascending=False)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tracker_tuning.json").write_text(json.dumps(table.to_dict("records"), indent=2) + "\n")
    lines = [f"# Tracker tuning on the training sequences ({track_params['detector']} detections)", "",
             f"{len(grid)} settings, sorted by {cfg['objective'].upper()}; all other settings as in params.yaml.", "",
             "| " + " | ".join(names) + " | MOTA | IDF1 | ID switches |", "|---" * (len(names) + 3) + "|"]
    for r in table.to_dict("records"):
        lines.append("| " + " | ".join(str(r[n]) for n in names) +
                     f" | {r['mota']:.3f} | {r['idf1']:.3f} | {r['id_switches']} |")
    (REPORTS / "tracker_tuning.md").write_text("\n".join(lines) + "\n")
    print("best:", table.iloc[0].to_dict())


def experiments(params: dict) -> None:
    track_params = params["track"]
    for name, overrides in track_params["experiments"].items():
        config, detector, gap = settings(track_params, overrides)
        results, ms = run(params["split"]["val"], config, detector, gap)
        for seq, tracks in results.items():
            write_results(tracks, TRACKS / name / f"{seq}.txt")
        info = {"detector": detector, "interpolate_gap": gap, "ms_per_frame": round(ms, 2),
                "config": {f.name: getattr(config, f.name) for f in fields(config)}}
        (TRACKS / name / "run.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"{name}: {sum(len(t) for t in results.values())} boxes, tracker {ms:.1f} ms/frame", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="grid search on the training sequences")
    params = load_params()
    if parser.parse_args().tune:
        tune(params)
    else:
        experiments(params)


if __name__ == "__main__":
    main()
