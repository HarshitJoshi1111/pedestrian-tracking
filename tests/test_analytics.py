import pandas as pd
import pytest

from src.analytics import count_report, crossings, match_events

LINE = [[100, 0], [100, 200]]  # drawn top to bottom: walking right to left is "in" (+1)


def walk(track_id, xs, y=100.0, start=1):
    return [(start + i, track_id, x - 5, y - 40, 10.0, 40.0) for i, x in enumerate(xs)]


def tracks(*rows):
    return pd.DataFrame([r for rs in rows for r in rs], columns=["frame", "id", "x", "y", "w", "h"])


def test_crossings_in_both_directions():
    df = tracks(walk(1, range(50, 150, 5)), walk(2, range(150, 50, -5)))
    events = crossings(df, LINE, smooth=1)
    assert sorted(zip(events["id"], events["direction"], strict=True)) == [(1, -1), (2, 1)]


def test_crossing_outside_the_segment_does_not_count():
    df = tracks(walk(1, range(50, 150, 5), y=300.0))  # passes below the end of the line
    assert crossings(df, LINE, smooth=1).empty


def test_back_and_forth_on_the_line_cancels():
    df = tracks(walk(1, [90, 95, 105, 110, 104, 96, 90, 85]))
    assert crossings(df, LINE, smooth=1, min_gap=15).empty
    assert len(crossings(df, LINE, smooth=1, min_gap=0)) == 2


def test_smoothing_removes_jitter_around_the_line():
    xs = [80, 85, 90, 95, 101, 99, 102, 98, 103, 106, 110, 115, 120]
    assert len(crossings(tracks(walk(1, xs)), LINE, smooth=5, min_gap=0)) == 1


def test_event_matching_and_report():
    truth = pd.DataFrame({"id": [1, 2, 3], "frame": [10, 50, 90], "direction": [1, 1, -1]})
    predicted = pd.DataFrame({"id": [7, 8, 9], "frame": [12, 75, 95], "direction": [1, 1, 1]})
    assert match_events(predicted, truth, window=15) == 1  # frame 75 is too late, frame 95 has the wrong direction
    r = count_report(predicted, truth, window=15)
    assert (r["true_in"], r["counted_in"], r["count_error"]) == (2, 3, 0.0)
    assert r["precision"] == pytest.approx(0.333) and r["recall"] == pytest.approx(0.333)


def run_counter(df, **kwargs):
    from src.analytics import LineCounter

    counter = LineCounter(LINE, **kwargs)
    for frame, g in df.groupby("frame"):
        counter.update(frame, [(r.id, (r.x, r.y, r.w, r.h)) for r in g.itertuples()])
    return counter.counts


def test_live_counter_matches_offline_counting():
    df = tracks(walk(1, range(50, 150, 5)), walk(2, range(150, 50, -5)), walk(3, range(40, 160, 4), start=5))
    offline = crossings(df, LINE, smooth=1)
    assert run_counter(df, smooth=1) == {1: int((offline["direction"] == 1).sum()), -1: int((offline["direction"] == -1).sum())}


def test_live_counter_takes_back_and_forth_back():
    df = tracks(walk(1, [90, 95, 105, 110, 104, 96, 90, 85]))
    assert run_counter(df, smooth=1, min_gap=15) == {1: 0, -1: 0}
