"""The whole pipeline live on any video: detect (ONNX Runtime), track, count people crossing a line.

    python -m src.app data/raw/MOT17-09/img1/%06d.jpg --line 1100 500 1100 1080 --out data/demo/live.mp4
    python -m src.app my_video.mp4 --line 0 600 1920 600 --show
    python -m src.app 0 --show                      webcam (no file written unless --out is given)

Frames are processed one by one as they arrive, as on a live camera: no detections are cached and the
counting does not look ahead. Prints the counts and the achieved frames per second at the end.
"""
import argparse
import time
from collections import defaultdict, deque

import cv2
import pandas as pd

from src.analytics import LineCounter
from src.onnx_detector import OnnxDetector
from src.params import ROOT, load_params
from src.render import draw_frame
from src.track import settings
from src.tracker import Tracker


def open_source(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {source}")
    return capture


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file, image sequence pattern (img1/%%06d.jpg) or webcam number")
    parser.add_argument("--model", default=str(ROOT / params["deploy"]["app_model"]))
    parser.add_argument("--line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--out", help="write the annotated video here (.mp4)")
    parser.add_argument("--show", action="store_true", help="show the video in a window (q to stop)")
    args = parser.parse_args()

    capture = open_source(args.source)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    detector = OnnxDetector(args.model, conf=params["deploy"]["app_conf"])
    config, _, _ = settings(params["track"])
    tracker = Tracker(config, fps)
    line = [args.line[:2], args.line[2:]] if args.line else None
    counter = LineCounter(line, params["analytics"]["smooth"], params["analytics"]["min_gap"]) if line else None
    trails = defaultdict(lambda: deque(maxlen=params["analytics"]["render"]["trail"]))
    writer, frame, start = None, 0, time.perf_counter()
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame += 1
        boxes, scores = detector(image)
        tracks = tracker.update(boxes, scores)
        if counter:
            counter.update(frame, [(track_id, box) for track_id, box, _ in tracks])
        rows = pd.DataFrame([(track_id, *box) for track_id, box, _ in tracks], columns=["id", "x", "y", "w", "h"])
        draw_frame(image, rows, trails, line, counter.counts if counter else None)
        if args.out:
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(image)
        if args.show:
            cv2.imshow("pedestrian tracking", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    seconds = time.perf_counter() - start
    if writer:
        writer.release()
    capture.release()
    cv2.destroyAllWindows()
    summary = f"{frame} frames in {seconds:.1f} s ({frame / max(seconds, 1e-9):.1f} FPS incl. reading and drawing)"
    if counter:
        summary += f", counted in {counter.counts[1]}, out {counter.counts[-1]}"
    print(summary)


if __name__ == "__main__":
    main()
