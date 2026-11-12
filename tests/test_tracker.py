import numpy as np

from src.tracker import Tracker, TrackerConfig

BOX = (40.0, 100.0)  # width, height


def walk(x0: float, t: int, speed: float = 4.0) -> list[float]:
    return [x0 + speed * t, 50.0, *BOX]


def run(tracker: Tracker, frames: list[list[tuple[list[float], float]]]) -> list[dict[int, list[float]]]:
    out = []
    for dets in frames:
        boxes = np.array([b for b, _ in dets]).reshape(-1, 4)
        scores = np.array([s for _, s in dets])
        out.append({tid: box for tid, box, _ in tracker.update(boxes, scores)})
    return out


def test_two_people_keep_their_ids():
    frames = [[(walk(0, t), 0.9), (walk(300, t, -4.0), 0.8)] for t in range(20)]
    out = run(Tracker(), frames)
    assert all(set(o) == {1, 2} for o in out)
    assert out[-1][1][0] < out[-1][2][0]  # IDs did not swap


def test_person_found_again_after_occlusion_keeps_id():
    frames = [[(walk(0, t), 0.9)] if not 10 <= t < 16 else [] for t in range(25)]
    out = run(Tracker(TrackerConfig(track_buffer=30)), frames)
    assert set(out[9]) == {1} and out[12] == {} and set(out[20]) == {1}
    out = run(Tracker(TrackerConfig(track_buffer=1)), frames)
    assert set(out[20]) == {2}  # forgotten during the occlusion: a new ID


def test_low_score_boxes_continue_a_track():
    # the person is half occluded in frames 10-14 and gets only a low score
    frames = [[(walk(0, t), 0.9 if not 10 <= t < 15 else 0.3)] for t in range(20)]
    byte = run(Tracker(TrackerConfig(low_thresh=0.1, track_buffer=1)), frames)
    sort = run(Tracker(TrackerConfig(low_thresh=None, track_buffer=1)), frames)
    assert all(set(o) == {1} for o in byte)
    assert sort[12] == {} and set(sort[-1]) == {2}


def test_one_frame_false_positive_gets_no_id():
    frames = [[(walk(0, t), 0.9)] + ([([500.0, 50.0, *BOX], 0.95)] if t == 5 else []) for t in range(10)]
    out = run(Tracker(), frames)
    assert all(set(o) == {1} for o in out)
    assert Tracker().update(np.zeros((0, 4)), np.zeros(0)) == []
