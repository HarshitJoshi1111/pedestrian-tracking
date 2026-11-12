import numpy as np

from src.kalman import KalmanBox, to_xyah, to_xywh


def test_box_conversion_round_trip():
    box = np.array([10.0, 20.0, 40.0, 100.0])
    assert np.allclose(to_xyah(box), [30, 70, 0.4, 100])
    assert np.allclose(to_xywh(to_xyah(box)), box)


def test_filter_learns_constant_velocity():
    kf = KalmanBox(np.array([0.0, 0.0, 40.0, 100.0]))
    for t in range(1, 30):
        kf.predict()
        kf.update(np.array([5.0 * t, 0.0, 40.0, 100.0]))
    kf.predict()  # no measurement: the filter moves the box on by itself
    assert abs(kf.box[0] - 5.0 * 30) < 1.0
    assert abs(kf.mean[4] - 5.0) < 0.2  # learned velocity of the centre, pixels per frame
