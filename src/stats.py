"""Dataset overview per sequence: size, frame rate, number of people and how hard the boxes are.

Writes reports/dataset.md and reports/dataset.json.
"""
import json

from src.mot import read_gt, read_seqinfo, split_gt
from src.params import RAW, REPORTS, load_params


def sequence_stats(seq: str) -> dict:
    info = read_seqinfo(RAW / seq / "seqinfo.ini")
    positives, ignore = split_gt(read_gt(RAW / seq / "gt" / "gt.txt"))
    return {
        "frames": info.frames, "fps": info.fps, "resolution": f"{info.width}x{info.height}",
        "tracks": int(positives["id"].nunique()), "boxes": len(positives), "ignore_boxes": len(ignore),
        "boxes_per_frame": round(len(positives) / info.frames, 1),
        "median_height": round(float(positives["h"].median()), 1),
        "share_small": round(float((positives["h"] < 50).mean()), 3),
        "share_occluded": round(float((positives["visibility"] < 0.3).mean()), 3),
    }


def main() -> None:
    params = load_params()
    split = {seq: name for name, seqs in params["split"].items() for seq in seqs}
    stats = {seq: {"split": split.get(seq, "-"), **sequence_stats(seq)} for seq in params["data"]["sequences"]}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "dataset.json").write_text(json.dumps(stats, indent=2) + "\n")
    lines = ["# MOT17 training sequences", "",
             "Pedestrian boxes that are scored (class 1, flag 1). Small: box height < 50 px. "
             "Occluded: visibility < 0.3.", "",
             "| Sequence | Split | Frames | FPS | Resolution | People | Boxes | Boxes/frame | Median height | Small | Occluded |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for seq, s in stats.items():
        lines.append(f"| {seq} | {s['split']} | {s['frames']} | {s['fps']:g} | {s['resolution']} | {s['tracks']} | "
                     f"{s['boxes']:,} | {s['boxes_per_frame']} | {s['median_height']:g} px | {s['share_small']:.0%} | "
                     f"{s['share_occluded']:.0%} |")
    for name in params["split"]:
        part = [s for s in stats.values() if s["split"] == name]
        lines.append(f"| **{name}** | | {sum(s['frames'] for s in part):,} | | | {sum(s['tracks'] for s in part)} | "
                     f"{sum(s['boxes'] for s in part):,} | | | | |")
    (REPORTS / "dataset.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
