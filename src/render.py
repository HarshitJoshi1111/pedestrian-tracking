"""Demo video: tracked people with their ID, a trail of where they walked, the counting line and live counts.

    python -m src.render              MP4 of the sequence in params.yaml -> data/demo/<sequence>.mp4
    python -m src.render --gif        additionally a short, small GIF   -> data/demo/<sequence>.gif

Output stays in data/ (not versioned).
"""
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from PIL import Image

from src.analytics import crossings
from src.mot import read_seqinfo
from src.params import RAW, ROOT, load_params
from src.track import run, settings

DEMO = ROOT / "data" / "demo"


def color(track_id: int) -> tuple[int, int, int]:
    """A fixed, well-separated colour per ID (golden-ratio steps around the hue circle)."""
    hue = int((track_id * 0.618033988749895 % 1) * 179)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_frame(image, rows, trails, line, counts) -> None:
    """Boxes, IDs and trails of the tracked people; the counting line and the counts if a line is given."""
    if line is not None:
        a, b = (tuple(int(v) for v in p) for p in line)
        cv2.line(image, a, b, (255, 255, 255), 4, cv2.LINE_AA)
    for _, r in rows.iterrows():
        x, y, w, h, tid = int(r.x), int(r.y), int(r.w), int(r.h), int(r.id)
        c = color(tid)
        trail = trails[tid]
        trail.append((x + w // 2, y + h))
        if len(trail) > 1:
            cv2.polylines(image, [np.array(trail, np.int32)], False, c, 3, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + w, y + h), c, 3)
        cv2.putText(image, str(tid), (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3, cv2.LINE_AA)
    text = (f"in {counts[1]}   out {counts[-1]}   " if line is not None else "") + f"people {len(rows)}"
    cv2.rectangle(image, (20, 20), (40 + 23 * len(text), 85), (0, 0, 0), -1)
    cv2.putText(image, text, (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    params = load_params()
    cfg, analytics = params["analytics"]["render"], params["analytics"]
    seq = cfg["sequence"]
    info = read_seqinfo(RAW / seq / "seqinfo.ini")
    config, detector, gap = settings(params["track"])
    tracks = run([seq], config, detector, gap)[0][seq]
    line = analytics["sequences"][seq]["line"]
    events = crossings(tracks, line, analytics["smooth"], analytics["min_gap"])
    by_frame = dict(list(tracks.groupby("frame")))
    trails = defaultdict(lambda: deque(maxlen=cfg["trail"]))
    counts = {1: 0, -1: 0}
    DEMO.mkdir(parents=True, exist_ok=True)
    size = (cfg["video_width"], round(cfg["video_width"] * info.height / info.width))
    video = cv2.VideoWriter(str(DEMO / f"{seq}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cfg["fps"], size)
    first, last = cfg["gif_frames"]
    gif_frames = []
    for frame in range(1, info.frames + 1):
        for direction in events.loc[events["frame"] == frame, "direction"]:
            counts[int(direction)] += 1
        image = cv2.imread(str(RAW / seq / "img1" / f"{frame:06d}.jpg"))
        draw_frame(image, by_frame.get(frame, tracks.iloc[:0]), trails, line, counts)
        video.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        if args.gif and first <= frame <= last and (frame - first) % cfg["gif_step"] == 0:
            small = cv2.resize(image, (cfg["gif_width"], round(cfg["gif_width"] * info.height / info.width)),
                               interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).quantize(colors=96))
    video.release()
    if gif_frames:
        duration = round(1000 * cfg["gif_step"] / info.fps)
        gif_frames[0].save(DEMO / f"{seq}.gif", save_all=True, append_images=gif_frames[1:], duration=duration, loop=0)
    print(f"{seq}: {info.frames} frames, counted in {counts[1]}, out {counts[-1]} -> {DEMO}")


if __name__ == "__main__":
    main()
