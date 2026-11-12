"""Constant-velocity Kalman filter for one bounding box.

State: centre x, centre y, aspect ratio w/h, height, and the velocity of each. Process and
measurement noise grow with the box height, so a large (near) person may move more pixels per frame
than a small (far) one. The noise factors are the ones used by DeepSORT and ByteTrack.
"""
import numpy as np

STD_POSITION = 1 / 20
STD_VELOCITY = 1 / 160

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)  # position += velocity (one frame)
_H = np.eye(4, 8)  # we measure the position part of the state


def to_xyah(box: np.ndarray) -> np.ndarray:
    """x, y, w, h (top-left corner) -> centre x, centre y, w / h, h."""
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2, w / h, h])


def to_xywh(state: np.ndarray) -> np.ndarray:
    cx, cy, a, h = state[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, w, h])


class KalmanBox:
    def __init__(self, box: np.ndarray):
        z = to_xyah(box)
        self.mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * STD_POSITION * h, 2 * STD_POSITION * h, 1e-2, 2 * STD_POSITION * h,
               10 * STD_VELOCITY * h, 10 * STD_VELOCITY * h, 1e-5, 10 * STD_VELOCITY * h]
        self.cov = np.diag(np.square(std))

    def predict(self) -> None:
        h = self.mean[3]
        std = [STD_POSITION * h, STD_POSITION * h, 1e-2, STD_POSITION * h,
               STD_VELOCITY * h, STD_VELOCITY * h, 1e-5, STD_VELOCITY * h]
        self.mean = _F @ self.mean
        self.cov = _F @ self.cov @ _F.T + np.diag(np.square(std))

    def update(self, box: np.ndarray) -> None:
        z = to_xyah(box)
        h = self.mean[3]
        noise = np.diag(np.square([STD_POSITION * h, STD_POSITION * h, 1e-1, STD_POSITION * h]))
        innovation_cov = _H @ self.cov @ _H.T + noise
        gain = np.linalg.solve(innovation_cov, _H @ self.cov).T  # cov H^T S^-1 (S is symmetric)
        self.mean = self.mean + gain @ (z - _H @ self.mean)
        self.cov = (np.eye(8) - gain @ _H) @ self.cov

    @property
    def box(self) -> np.ndarray:
        return to_xywh(self.mean)
