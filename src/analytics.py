"""Stage 8: what the tracks are used for: people counting at a line, crowd size, heatmaps.

For every sequence in params.yaml (static cameras only: on a moving camera a line drawn in the image
does not stay on the same spot of the street) the tracker is run and then:

- line counting: a person is counted when the path of their feet crosses the line, as "in" or "out"
  depending on the direction. The same counter runs on the ground-truth trajectories, so every
  crossing the tracker produces can be checked: a predicted crossing is correct if a real person
  crossed in the same direction within a few frames. A count can be right by luck (one missed and
  one extra crossing), so precision and recall of the crossing events are reported as well.
- crowd size: number of tracked people per frame against the true number (mean absolute error)
- heatmap: where people walk (density of feet positions), drawn over a frame of the video

Writes reports/analytics.json, reports/analytics.md and figures in reports/figures/.
"""
import json

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.evaluate_det import load_split
from src.mot import read_seqinfo
from src.params import RAW, REPORTS, load_params
from src.track import run, settings

FIGURES = REPORTS / "figures"


def feet(tracks: pd.DataFrame, smooth: int) -> pd.DataFrame:
    """Bottom centre of each box (where the person stands), smoothed along each track against box jitter."""
    out = tracks.sort_values(["id", "frame"]).assign(fx=lambda d: d["x"] + d["w"] / 2, fy=lambda d: d["y"] + d["h"])
    rolling = out.groupby("id")[["fx", "fy"]].rolling(smooth, center=True, min_periods=1).mean()
    out[["fx", "fy"]] = rolling.reset_index(level=0, drop=True)
    return out


def _side(px, py, a, b):
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


def crossings(tracks: pd.DataFrame, line, smooth: int = 5, min_gap: int = 15) -> pd.DataFrame:
    """Line crossings as rows (id, frame, direction). Direction +1 ("in") for crossing in the direction that
    is right to left for a line drawn from top (a) to bottom (b) on the screen; -1 ("out") for the other way.

    A crossing back within min_gap frames cancels both (a person standing on the line, or jitter).
    """
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    events = []
    for track_id, t in feet(tracks, smooth).groupby("id"):
        px, py, frames = t["fx"].to_numpy(), t["fy"].to_numpy(), t["frame"].to_numpy()
        side = np.where(_side(px, py, a, b) >= 0, 1, -1)  # a point exactly on the line counts as side +1
        for k in np.flatnonzero(side[:-1] * side[1:] < 0):
            # the step must cross the segment itself, not its extension: a and b on different sides of the step
            p, q = (px[k], py[k]), (px[k + 1], py[k + 1])
            if _side(*a, p, q) * _side(*b, p, q) < 0:
                events.append((track_id, frames[k + 1], int(side[k + 1])))
    kept = []
    for event in events:
        if kept and kept[-1][0] == event[0] and kept[-1][2] == -event[2] and event[1] - kept[-1][1] <= min_gap:
            kept.pop()  # back and forth: neither counts
        else:
            kept.append(event)
    return pd.DataFrame(kept, columns=["id", "frame", "direction"])


def match_events(predicted: pd.DataFrame, truth: pd.DataFrame, window: int) -> int:
    """Number of predicted crossings that match a true crossing (same direction, within `window` frames)."""
    matched = 0
    for direction in (1, -1):
        p = np.sort(predicted.loc[predicted["direction"] == direction, "frame"].to_numpy())
        t = list(np.sort(truth.loc[truth["direction"] == direction, "frame"].to_numpy()))
        for frame in p:  # greedy in time order: take the closest free true crossing
            if t:
                j = int(np.argmin(np.abs(np.array(t) - frame)))
                if abs(t[j] - frame) <= window:
                    t.pop(j)
                    matched += 1
    return matched


def count_report(predicted: pd.DataFrame, truth: pd.DataFrame, window: int) -> dict:
    n_pred, n_true = len(predicted), len(truth)
    hits = match_events(predicted, truth, window)
    return {"true_in": int((truth["direction"] == 1).sum()), "true_out": int((truth["direction"] == -1).sum()),
            "counted_in": int((predicted["direction"] == 1).sum()),
            "counted_out": int((predicted["direction"] == -1).sum()),
            "count_error": round(abs(n_pred - n_true) / n_true, 3) if n_true else None,
            "precision": round(hits / n_pred, 3) if n_pred else None,
            "recall": round(hits / n_true, 3) if n_true else None}


def people_per_frame(df: pd.DataFrame, n_frames: int) -> np.ndarray:
    return df.groupby("frame").size().reindex(range(1, n_frames + 1), fill_value=0).to_numpy()


def heatmap(tracks: pd.DataFrame, seq: str, width: int, height: int, smooth: int) -> None:
    f = feet(tracks, smooth)
    density, _, _ = np.histogram2d(f["fy"].clip(0, height - 1), f["fx"].clip(0, width - 1),
                                   bins=(height // 8, width // 8), range=((0, height), (0, width)))
    density = np.log1p(cv2.GaussianBlur(density.astype(np.float32), (0, 0), 2))  # log: people standing still
    density = cv2.resize(density / max(float(density.max()), 1e-9), (width, height))  # would hide the walking paths
    colors = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    frame = cv2.imread(str(RAW / seq / "img1" / "000001.jpg"))
    alpha = np.clip(density * 2, 0, 0.8)[..., None]
    image = (frame * (1 - alpha) + colors * alpha).astype(np.uint8)
    image = cv2.resize(image, (960, round(960 * height / width)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(FIGURES / f"heatmap_{seq}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def crowd_figure(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig = Figure(figsize=(9, 2.6 * len(series)))
    for ax, (seq, (truth, tracked)) in zip(fig.subplots(len(series), 1, squeeze=False)[:, 0], series.items(), strict=True):
        ax.plot(truth, label="true (incl. hidden people)", linewidth=1.2)
        ax.plot(tracked, label="tracked", linewidth=1.2)
        ax.set(title=seq, xlabel="Frame", ylabel="People")
        ax.grid(alpha=0.3)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "crowd_size.png", dpi=130)


def main() -> None:
    params = load_params()
    cfg = params["analytics"]
    sequences = list(cfg["sequences"])
    config, detector, gap = settings(params["track"])
    tracks, _ = run(sequences, config, detector, gap)
    gt = load_split(sequences)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results, series = {}, {}
    for seq in sequences:
        info = read_seqinfo(RAW / seq / "seqinfo.ini")
        line = cfg["sequences"][seq]["line"]
        truth = crossings(gt[seq][0], line, cfg["smooth"], cfg["min_gap"])
        predicted = crossings(tracks[seq], line, cfg["smooth"], cfg["min_gap"])
        true_count, tracked_count = people_per_frame(gt[seq][0], info.frames), people_per_frame(tracks[seq], info.frames)
        results[seq] = {"split": next(s for s, seqs in params["split"].items() if seq in seqs), "line": line,
                        **count_report(predicted, truth, cfg["match_window"]),
                        "mean_people": round(float(true_count.mean()), 1),
                        "crowd_mae": round(float(np.abs(true_count - tracked_count).mean()), 2)}
        series[seq] = (true_count, tracked_count)
        heatmap(tracks[seq], seq, info.width, info.height, cfg["smooth"])
        print(f"{seq}: {results[seq]}")
    crowd_figure(series)
    (REPORTS / "analytics.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Analytics on the static-camera sequences", "",
             f"Tracks: {detector} + tracker settings from params.yaml (interpolation of gaps up to {gap} frames). "
             f"A counted crossing is correct if a real person crossed in the same direction within "
             f"{cfg['match_window']} frames. Crowd MAE: mean absolute difference between tracked and true people "
             "per frame (the truth includes fully hidden people).", "",
             "| Sequence | Split | True in / out | Counted in / out | Count error | Precision | Recall | People/frame "
             "| Crowd MAE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for seq, r in results.items():
        lines.append(f"| {seq} | {r['split']} | {r['true_in']} / {r['true_out']} | {r['counted_in']} / {r['counted_out']} | "
                     f"{r['count_error']:.0%} | {r['precision']:.2f} | {r['recall']:.2f} | {r['mean_people']} | "
                     f"{r['crowd_mae']} |")
    lines += ["", "The tracker settings were tuned on the training sequences, so MOT17-04 is not an independent test."]
    (REPORTS / "analytics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
