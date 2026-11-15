"""Stage 7: fine-tune the COCO-pretrained detector on the MOT17 training sequences.

The COCO model has never seen how MOT17 draws its boxes (the full body, also the hidden part of a
person behind an obstacle), which is why its boxes were too loose (low AP75) and occluded people
were missed. Fine-tuning on the training sequences teaches that convention.

Saves the best checkpoint (holdout mAP) as models/weights/<name>.pt, the training curve as
reports/training.json and reports/figures/training.png. Training runs go to runs/ (not versioned).
"""
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from src.detect import WEIGHTS, load_model, pick_device
from src.params import REPORTS, ROOT, load_params
from src.prepare_yolo import YOLO_DIR

RUNS = ROOT / "runs"


def read_curve(results_csv: Path) -> pd.DataFrame:
    curve = pd.read_csv(results_csv)
    curve.columns = [c.strip() for c in curve.columns]
    return curve.rename(columns={"metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95",
                                 "train/box_loss": "train_box_loss", "val/box_loss": "holdout_box_loss"})


def figure(curve: pd.DataFrame, path: Path) -> None:
    fig = Figure(figsize=(8, 3.4))
    loss_ax, map_ax = fig.subplots(1, 2)
    loss_ax.plot(curve["epoch"], curve["train_box_loss"], label="training")
    loss_ax.plot(curve["epoch"], curve["holdout_box_loss"], label="holdout")
    loss_ax.set(xlabel="Epoch", ylabel="Box loss", title="Box regression loss")
    loss_ax.legend()
    map_ax.plot(curve["epoch"], curve["map50"], label="mAP50")
    map_ax.plot(curve["epoch"], curve["map50_95"], label="mAP50-95")
    map_ax.set(xlabel="Epoch", ylabel="mAP", ylim=(0, 1), title="Holdout frames (training sequences)")
    map_ax.legend()
    for ax in (loss_ax, map_ax):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    cfg = load_params()["train"]
    device = pick_device()
    model = load_model(cfg["model"])
    start = time.time()
    model.train(data=str(YOLO_DIR / "mot17.yaml"), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
                device=device, seed=cfg["seed"], project=str(RUNS), name=cfg["name"], exist_ok=True, plots=False,
                workers=4)
    minutes = (time.time() - start) / 60
    run_dir = Path(model.trainer.save_dir)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS / f"{cfg['name']}.pt")

    curve = read_curve(run_dir / "results.csv")
    best = curve.loc[curve["map50_95"].idxmax()]
    summary = {"model": cfg["model"], "name": cfg["name"], "device": device, "imgsz": cfg["imgsz"],
               "epochs": int(curve["epoch"].max()), "minutes": round(minutes, 1), "best_epoch": int(best["epoch"]),
               "holdout_map50": round(float(best["map50"]), 4), "holdout_map50_95": round(float(best["map50_95"]), 4),
               "curve": curve[["epoch", "train_box_loss", "holdout_box_loss", "map50", "map50_95"]].round(4).to_dict("records")}
    (REPORTS / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    figure(curve, REPORTS / "figures" / "training.png")
    print(f"trained {summary['epochs']} epochs in {minutes:.0f} min, best epoch {summary['best_epoch']}: "
          f"holdout mAP50 {summary['holdout_map50']:.3f}, mAP50-95 {summary['holdout_map50_95']:.3f}")


if __name__ == "__main__":
    main()
