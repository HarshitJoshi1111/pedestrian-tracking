"""Stage 9: export the detectors to ONNX, check that nothing is lost, and measure speed.

    python -m src.deploy export      models/weights/<model>-<width>.onnx, fixed input size from params.yaml
    python -m src.deploy detect      ONNX detections on the validation sequences (-> data/detections/<model>-onnx-...),
                                     scored by python -m src.evaluate_det next to the PyTorch runs; with NMS IoU 0.5
                                     (as PyTorch) to check that the export loses nothing, and with 0.7
    python -m src.deploy benchmark   time per frame for PyTorch (CPU, Apple GPU) and ONNX Runtime (CPU, Core ML)
                                     -> reports/benchmark.json, reports/benchmark.md

The input size is fixed to 736 x 1280: a 16:9 video scaled to 1280 px wide is 720 px high, padded to a
multiple of the model's stride (32). It is the same input PyTorch builds for these videos.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detect import PERSON, WEIGHTS, load_model, pick_device
from src.mot import RESULT_COLUMNS, read_seqinfo, write_results
from src.onnx_detector import OnnxDetector
from src.params import DETECTIONS, RAW, REPORTS, load_params


def onnx_path(model: str, width: int) -> Path:
    return WEIGHTS / f"{Path(model).stem}-{width}.onnx"


def export(cfg: dict) -> None:
    height, width = cfg["imgsz"]
    for name in cfg["models"]:
        model = load_model(name)
        exported = Path(model.export(format="onnx", imgsz=[height, width], dynamic=False, simplify=True, nms=False))
        shutil.move(exported, onnx_path(name, width))
        print(f"{name} -> {onnx_path(name, width)} ({onnx_path(name, width).stat().st_size / 1e6:.1f} MB)")


def onnx_run_name(model: str, iou: float, width: int) -> str:
    """'yolo11n-onnx-1280' with the default NMS threshold 0.5, 'yolo11n-onnx-iou07-1280' with 0.7."""
    return f"{Path(model).stem}-onnx{'' if iou == 0.5 else f'-iou{iou:.1f}'.replace('.', '')}-{width}"


def detect(cfg: dict, sequences: list[str]) -> None:
    """On the CPU, so that the boxes can be compared with the PyTorch ones (Core ML may compute in half precision)."""
    width = cfg["imgsz"][1]
    for name in cfg["models"]:
        for iou in cfg["nms_iou"]:
            run_detector(OnnxDetector(onnx_path(name, width), conf=cfg["conf"], iou=iou, providers=["CPUExecutionProvider"]),
                         DETECTIONS / onnx_run_name(name, iou, width), sequences)


def run_detector(detector: OnnxDetector, out: Path, sequences: list[str]) -> None:
    for seq in sequences:
        if (out / f"{seq}.txt").exists():
            continue  # cached already
        rows = []
        for frame in range(1, read_seqinfo(RAW / seq / "seqinfo.ini").frames + 1):
            boxes, scores = detector(cv2.imread(str(RAW / seq / "img1" / f"{frame:06d}.jpg")))
            rows += [(frame, -1, x + 1, y + 1, w, h, s) for (x, y, w, h), s in zip(boxes, scores, strict=True)]
        write_results(pd.DataFrame(rows, columns=RESULT_COLUMNS), out / f"{seq}.txt")
        print(f"{out.name} {seq}: {len(rows)} boxes", flush=True)


def timed(run, frames: list[Path], warmup: int) -> dict:
    """Milliseconds per frame for run(image), the image already in memory (reading the file is not timed)."""
    times = []
    for i, path in enumerate(frames):
        image = cv2.imread(str(path))
        start = time.perf_counter()
        run(image)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    t = np.array(times)
    return {"ms_mean": round(float(t.mean()), 1), "ms_p50": round(float(np.median(t)), 1),
            "ms_p95": round(float(np.percentile(t, 95)), 1), "fps": round(1000 / float(t.mean()), 1)}


def benchmark(cfg: dict) -> None:
    import onnxruntime as ort

    bench, (height, width) = cfg["benchmark"], cfg["imgsz"]
    frames = sorted((RAW / bench["sequence"] / "img1").glob("*.jpg"))[: bench["frames"] + bench["warmup"]]
    devices = ["cpu"] + ([pick_device()] if pick_device() != "cpu" else [])
    providers = [p for p in ("CPUExecutionProvider", "CoreMLExecutionProvider") if p in ort.get_available_providers()]
    results = []
    for name in cfg["models"]:
        model = load_model(name)
        for device in devices:
            def run(image, model=model, device=device):
                result = model.predict(image, imgsz=width, conf=cfg["app_conf"], classes=[PERSON], device=device,
                                       verbose=False)[0]
                result.boxes.xyxy.cpu()  # wait for the GPU
            results.append({"model": name, "runtime": f"PyTorch {device}", **timed(run, frames, bench["warmup"])})
            print(results[-1], flush=True)
        for provider in providers:
            label = f"ONNX Runtime {provider.replace('ExecutionProvider', '')}"
            try:
                fallback = [] if provider == "CPUExecutionProvider" else ["CPUExecutionProvider"]
                detector = OnnxDetector(onnx_path(name, width), conf=cfg["app_conf"], providers=[provider, *fallback])
                results.append({"model": name, "runtime": label, **timed(detector, frames, bench["warmup"])})
            except Exception as error:  # an execution provider may not support every operation
                print(f"{name} {label} failed: {error}")
                continue
            print(results[-1], flush=True)
    sizes = {name: {"pt_mb": round((WEIGHTS / name).stat().st_size / 1e6, 1),
                    "onnx_mb": round(onnx_path(name, width).stat().st_size / 1e6, 1)} for name in cfg["models"]}
    (REPORTS / "benchmark.json").write_text(json.dumps({"input": [height, width], "results": results, "sizes": sizes},
                                                       indent=2) + "\n")
    lines = [f"# Speed on {platform()}", "",
             f"{bench['frames']} frames of {bench['sequence']} (1920x1080), model input {height}x{width}, after "
             f"{bench['warmup']} warm-up frames. Time per frame including pre-processing and non-maximum suppression, "
             "without reading the file. The tracker adds about 0.5 ms per frame.", "",
             "| Model | Runtime | ms/frame (mean) | median | 95th percentile | FPS |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['runtime']} | {r['ms_mean']} | {r['ms_p50']} | {r['ms_p95']} | {r['fps']} |"
              for r in results]
    lines += ["", "| Model | PyTorch file | ONNX file |", "|---|---|---|"]
    lines += [f"| {n} | {s['pt_mb']} MB | {s['onnx_mb']} MB |" for n, s in sizes.items()]
    (REPORTS / "benchmark.md").write_text("\n".join(lines) + "\n")


def platform() -> str:
    import platform as p
    import subprocess

    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = ""
    return chip or p.processor() or p.machine()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["export", "detect", "benchmark"])
    params = load_params()
    step = parser.parse_args().step
    if step == "export":
        export(params["deploy"])
    elif step == "detect":
        detect(params["deploy"], params["split"]["val"])
    else:
        benchmark(params["deploy"])


if __name__ == "__main__":
    main()
