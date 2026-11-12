"""Stage 5: score every tracking run on the validation sequences.

Writes reports/tracking.json and reports/tracking.md: overall MOTA, IDF1 and error counts per run,
and the per-sequence scores of the main run.
"""
import json

from src.evaluate_det import load_split
from src.mot import read_results
from src.params import REPORTS, load_params
from src.reference_track import NAME as REFERENCE
from src.track import TRACKS
from src.track_metrics import combine, evaluate_sequence

MAIN = "bytetrack+interp"


def evaluate_run(name: str, gt: dict) -> dict:
    per_sequence = {seq: evaluate_sequence(read_results(TRACKS / name / f"{seq}.txt"), *gt[seq]) for seq in gt}
    info = json.loads((TRACKS / name / "run.json").read_text())
    return {"overall": combine(per_sequence), "per_sequence": per_sequence, "run": info}


def row(name: str, r: dict) -> str:
    o, run = r["overall"], r["run"]
    speed = run.get("ms_per_frame", "-")
    return (f"| {name} | {run['detector']} | {o['mota']:.3f} | {o['idf1']:.3f} | {o['idp']:.3f} | {o['idr']:.3f} | "
            f"{o['id_switches']:,} | {o['false_positives']:,} | {o['misses']:,} | {o['mostly_tracked']} | "
            f"{o['mostly_lost']} | {o['track_ids']} | {speed} |")


def markdown(results: dict, people: int) -> str:
    header = ("| Run | Detector | MOTA | IDF1 | IDP | IDR | ID switches | False positives | Misses | MT | ML | "
              "Track IDs | Tracker ms/frame |")
    lines = ["# Tracking on the validation sequences", "",
             f"MOT17-02, -09, -10: {people} people. MT/ML: people tracked for at least 80% / less than 20% of the "
             "frames they appear in. Tracker time excludes the detector. Settings tuned on the training sequences. "
             f"{REFERENCE}: the library's own tracker with its default settings, run inside the detector.", "",
             header, "|---" * (header.count("|") - 1) + "|"]
    lines += [row(name, r) for name, r in results.items()]
    if MAIN in results:
        seqs = results[MAIN]["per_sequence"]
        lines += ["", f"## {MAIN} per sequence", "", "| Sequence | MOTA | IDF1 | ID switches | People | Track IDs |",
                  "|---|---|---|---|---|---|"]
        for seq, s in seqs.items():
            lines.append(f"| {seq} | {s['mota']:.3f} | {s['idf1']:.3f} | {s['id_switches']} | {s['people']} | "
                         f"{s['track_ids']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    params = load_params()
    gt = load_split(params["split"]["val"])
    names = [n for n in [*params["track"]["experiments"], REFERENCE] if (TRACKS / n / "run.json").exists()]
    results = {}
    for name in names:
        results[name] = evaluate_run(name, gt)
        o = results[name]["overall"]
        print(f"{name}: MOTA {o['mota']:.3f}, IDF1 {o['idf1']:.3f}, ID switches {o['id_switches']}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tracking.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    people = next(iter(results.values()))["overall"]["people"]
    (REPORTS / "tracking.md").write_text(markdown(results, people))


if __name__ == "__main__":
    main()
