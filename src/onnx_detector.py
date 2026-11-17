"""Person detector on ONNX Runtime: the exported YOLO model with own pre- and post-processing.

Needs only onnxruntime, numpy and OpenCV (no PyTorch), which is what a deployment would ship.
Pre-processing: letterbox (keep the aspect ratio, pad with grey to the model's fixed input size).
Post-processing: the model returns, for every candidate box, its centre, size and one score per class;
a box is a person if "person" is its best class and the score passes the threshold; overlapping
boxes are then reduced with non-maximum suppression (IoU 0.5, the default of Ultralytics' predict).
"""
from pathlib import Path

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(image: np.ndarray, height: int, width: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping the aspect ratio and pad to (height, width), image centred. Returns (image, scale, padding)."""
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w, new_h) != (w, h) else image
    pad_x, pad_y = (width - new_w) / 2, (height - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    padded = cv2.copyMakeBorder(resized, top, height - new_h - top, left, width - new_w - left, cv2.BORDER_CONSTANT,
                                value=(PAD_VALUE,) * 3)
    return padded, scale, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression on x1, y1, x2, y2 boxes; returns the kept indices, best first."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        overlap = w * h / (areas[i] + areas[rest] - w * h + 1e-9)
        order = rest[overlap < iou_threshold]
    return np.array(keep, dtype=int)


def decode(output: np.ndarray, conf: float, class_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw model output (4 + classes, candidates) -> person boxes x1, y1, x2, y2 (model input pixels) and scores."""
    preds = output.T
    class_scores = preds[:, 4:]
    best = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(preds)), best]
    keep = (best == class_id) & (scores >= conf)
    cx, cy, w, h = preds[keep, :4].T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), scores[keep]


FAST_PROVIDERS = ("CUDAExecutionProvider", "CoreMLExecutionProvider")  # NVIDIA GPU, Apple GPU / Neural Engine


def best_providers() -> list[str]:
    """The fastest execution provider available on this machine, with the CPU as fallback for unsupported layers."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    return [p for p in FAST_PROVIDERS if p in available][:1] + ["CPUExecutionProvider"]


class OnnxDetector:
    def __init__(self, path: Path, conf: float = 0.1, iou: float = 0.5, max_det: int = 300,
                 providers: list[str] | None = None, class_id: int = 0):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=providers or best_providers())
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.height, self.width = model_input.shape[2], model_input.shape[3]
        self.conf, self.iou, self.max_det, self.class_id = conf, iou, max_det, class_id

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR image -> person boxes (x, y, w, h in image pixels, 0-based) and scores."""
        padded, scale, (left, top) = letterbox(image, self.height, self.width)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        output = self.session.run(None, {self.input_name: blob})[0][0]
        xyxy, scores = decode(output, self.conf, self.class_id)
        keep = nms(xyxy, scores, self.iou)[: self.max_det]
        xyxy, scores = xyxy[keep], scores[keep]
        xyxy = (xyxy - [left, top, left, top]) / scale
        h, w = image.shape[:2]
        xyxy = np.clip(xyxy, 0, [w, h, w, h])
        return np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], scores
