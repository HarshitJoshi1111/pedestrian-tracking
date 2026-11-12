"""Multi-object tracker: Kalman prediction plus IoU matching with the Hungarian algorithm (ByteTrack style).

Every frame:
1. predict each track one frame ahead with its Kalman filter
2. match confirmed and lost tracks to the high-score boxes (cost 1 - IoU, optionally weighted by score)
3. match the confirmed tracks that are still free to the low-score boxes. A half-occluded person often
   gets a low score; throwing those boxes away (as SORT does) breaks the track exactly when it is hard
4. match tentative tracks (seen in one frame so far) to the remaining high-score boxes, drop the others
5. start a tentative track for every remaining high-score box with score >= new_track_thresh
6. forget lost tracks that have not been seen for track_buffer frames

A track gets its ID when it is confirmed (matched in two frames), so one-frame false positives use no ID.
With low_thresh = None step 3 is skipped, which gives a SORT-style tracker.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.kalman import KalmanBox
from src.metrics import iou_matrix


class State(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


@dataclass(frozen=True)
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float | None = 0.1  # None: ignore low-score boxes (SORT style)
    new_track_thresh: float | None = None  # None: high_thresh + 0.1
    match_thresh: float = 0.8  # largest cost (1 - IoU) accepted in the first matching
    low_match_thresh: float = 0.5
    tentative_match_thresh: float = 0.7
    track_buffer: int = 30  # frames a lost track is kept, given for 30 fps (scaled to the video's rate)
    fuse_score: bool = True  # multiply IoU with the box score: confident boxes win close calls
    min_box_area: float = 100  # output filter: tiny boxes
    max_aspect: float = 1.6  # output filter: boxes much wider than tall are not standing people

    @property
    def start_thresh(self) -> float:
        return self.high_thresh + 0.1 if self.new_track_thresh is None else self.new_track_thresh


class Track:
    def __init__(self, box: np.ndarray, score: float, frame: int):
        self.id: int | None = None
        self.kf = KalmanBox(box)
        self.score = score
        self.state = State.TENTATIVE
        self.last_frame = frame

    def predict(self) -> None:
        if self.state != State.CONFIRMED:
            self.kf.mean[7] = 0  # no growth of the box while the person is not seen
        self.kf.predict()

    def update(self, box: np.ndarray, score: float, frame: int) -> None:
        self.kf.update(box)
        self.score = score
        self.last_frame = frame


def associate(tracks: list[Track], boxes: np.ndarray, scores: np.ndarray, max_cost: float, fuse_score: bool):
    """Hungarian matching on 1 - IoU. Returns (pairs, unmatched track indices, unmatched box indices)."""
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    similarity = iou_matrix(np.array([t.kf.box for t in tracks]), boxes)
    if fuse_score:
        similarity = similarity * scores[None, :]
    cost = 1 - similarity
    cost[cost > max_cost] = max_cost + 1e3  # forbidden pairs, so they cannot push others out of the optimum
    rows, cols = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    matched_tracks, matched_boxes = {r for r, _ in pairs}, {c for _, c in pairs}
    return (pairs, [i for i in range(len(tracks)) if i not in matched_tracks],
            [j for j in range(len(boxes)) if j not in matched_boxes])


class Tracker:
    def __init__(self, config: TrackerConfig | None = None, fps: float = 30):
        self.cfg = config = config or TrackerConfig()
        self.max_lost = max(1, round(config.track_buffer * fps / 30))
        self.tracks: list[Track] = []
        self.frame = 0
        self.next_id = 1

    def _confirm(self, track: Track) -> None:
        if track.id is None:
            track.id, self.next_id = self.next_id, self.next_id + 1
        track.state = State.CONFIRMED

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """Feed the boxes (x, y, w, h) of the next frame; returns (track id, box, score) of the tracked people."""
        cfg = self.cfg
        self.frame += 1
        boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
        high = scores >= cfg.high_thresh
        low = ~high & (scores >= cfg.low_thresh) if cfg.low_thresh is not None else np.zeros_like(high)
        high_boxes, high_scores = boxes[high], scores[high]

        for track in self.tracks:
            track.predict()
        tentative = [t for t in self.tracks if t.state == State.TENTATIVE]
        pool = [t for t in self.tracks if t.state != State.TENTATIVE]

        pairs, free_tracks, free_boxes = associate(pool, high_boxes, high_scores, cfg.match_thresh, cfg.fuse_score)
        for ti, bi in pairs:
            pool[ti].update(high_boxes[bi], high_scores[bi], self.frame)
            self._confirm(pool[ti])  # a lost track that is found again keeps its ID

        was_tracked = [pool[i] for i in free_tracks if pool[i].state == State.CONFIRMED]
        pairs, still_free, _ = associate(was_tracked, boxes[low], scores[low], cfg.low_match_thresh, False)
        for ti, bi in pairs:
            was_tracked[ti].update(boxes[low][bi], scores[low][bi], self.frame)
        for ti in still_free:
            was_tracked[ti].state = State.LOST

        rest_boxes, rest_scores = high_boxes[free_boxes], high_scores[free_boxes]
        pairs, dropped, new = associate(tentative, rest_boxes, rest_scores, cfg.tentative_match_thresh, cfg.fuse_score)
        for ti, bi in pairs:
            tentative[ti].update(rest_boxes[bi], rest_scores[bi], self.frame)
            self._confirm(tentative[ti])
        removed = {id(tentative[ti]) for ti in dropped}

        for bi in new:
            if rest_scores[bi] >= cfg.start_thresh:
                track = Track(rest_boxes[bi], rest_scores[bi], self.frame)
                if self.frame == 1:
                    self._confirm(track)  # nothing to confirm against in the first frame
                self.tracks.append(track)

        self.tracks = [t for t in self.tracks
                       if id(t) not in removed and not (t.state == State.LOST and self.frame - t.last_frame > self.max_lost)]
        output = []
        for t in self.tracks:
            if t.state == State.CONFIRMED and t.last_frame == self.frame:
                box = t.kf.box
                if box[2] * box[3] >= cfg.min_box_area and box[2] / box[3] <= cfg.max_aspect:
                    output.append((t.id, box, t.score))
        return output
