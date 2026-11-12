import pandas as pd
import pytest

from src.track import interpolate
from src.track_metrics import clear_mot, combine, evaluate_sequence, identity, remove_ignored

COLS = ["frame", "id", "x", "y", "w", "h"]


def boxes(rows):
    return pd.DataFrame(rows, columns=COLS).assign(score=1.0)


def person(pid, frames, x=0.0):
    return [(f, pid, x + f, 0.0, 10.0, 20.0) for f in frames]


def test_perfect_tracking():
    gt = boxes(person(1, range(1, 11)) + person(2, range(1, 11), x=100))
    hyp = boxes(person(7, range(1, 11)) + person(9, range(1, 11), x=100))
    m = clear_mot(gt, hyp)
    assert m["mota"] == 1 and m["id_switches"] == 0 and m["mostly_tracked"] == 2
    assert identity(gt, hyp)["idf1"] == 1


def test_id_switch_halfway():
    gt = boxes(person(1, range(1, 11)))
    hyp = boxes(person(7, range(1, 6)) + person(8, range(6, 11)))
    assert clear_mot(gt, hyp)["id_switches"] == 1
    assert clear_mot(gt, hyp)["mota"] == pytest.approx(0.9)
    assert identity(gt, hyp)["idf1"] == pytest.approx(0.5)


def test_misses_and_false_positives():
    gt = boxes(person(1, range(1, 11)))
    hyp = boxes(person(7, range(1, 6)) + [(3, 9, 500.0, 0.0, 10.0, 20.0)])
    m = clear_mot(gt, hyp)
    assert (m["misses"], m["false_positives"], m["mota"]) == (5, 1, pytest.approx(0.4))
    assert m["mostly_tracked"] == 0 and m["mostly_lost"] == 0  # tracked half of the time


def test_overlapping_track_is_kept_instead_of_switching():
    # a second track appears on top of the person; the matching keeps the existing one
    gt = boxes(person(1, range(1, 6)))
    hyp = boxes(person(7, range(1, 6)) + [(f, 8, f + 0.5, 0.0, 10.0, 20.0) for f in range(3, 6)])
    m = clear_mot(gt, hyp)
    assert m["id_switches"] == 0 and m["false_positives"] == 3


def test_boxes_on_ignored_people_are_removed():
    positives = boxes(person(1, [1]))
    ignore = boxes(person(2, [1], x=200))
    hyp = boxes([(1, 5, 1.0, 0.0, 10.0, 20.0), (1, 6, 201.0, 0.0, 10.0, 20.0), (1, 7, 400.0, 0.0, 10.0, 20.0)])
    assert remove_ignored(hyp, positives, ignore)["id"].tolist() == [5, 7]
    assert evaluate_sequence(hyp, positives, ignore)["false_positives"] == 1


def test_combine_sums_counts():
    gt = boxes(person(1, range(1, 11)))
    a = evaluate_sequence(boxes(person(7, range(1, 11))), gt, gt.iloc[:0])
    b = evaluate_sequence(boxes(person(7, range(1, 6))), gt, gt.iloc[:0])
    total = combine({"a": a, "b": b})
    assert total["mota"] == pytest.approx(15 / 20)
    assert total["idf1"] == pytest.approx(2 * 15 / (20 + 15))


def test_interpolate_fills_short_gaps_only():
    tracks = boxes(person(1, [1, 2, 5, 30]))
    filled = interpolate(tracks, max_gap=5)
    assert filled["frame"].tolist() == [1, 2, 3, 4, 5, 30]
    assert filled.loc[filled["frame"] == 3, "x"].item() == pytest.approx(3.0)
