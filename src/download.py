"""Stage 1: download the MOT17 training sequences (frames, ground truth and sequence info).

Only the members we need are read out of the 5.5 GB archive (see remote_zip.py). Files are written
atomically and existing files of the right size are skipped, so an interrupted download can be rerun.
"""
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.mot import read_seqinfo
from src.params import RAW, load_params
from src.remote_zip import Fetch, HttpRange, list_members, read_member

# e.g. "MOT17/train/MOT17-02-FRCNN/img1/000001.jpg"
MEMBER = re.compile(r"(?:^|/)train/(MOT17-\d{2})-(\w+)/(img1/\d+\.jpg|gt/gt\.txt|seqinfo\.ini)$")


def select(names, sequences: list[str], variant: str) -> dict[str, Path]:
    """Archive member -> path relative to data/raw, e.g. 'MOT17-02/img1/000001.jpg'."""
    wanted = {}
    for name in names:
        m = MEMBER.search(name)
        if m and m.group(1) in sequences and m.group(2) == variant:
            wanted[name] = Path(m.group(1)) / m.group(3)
    found = {p.parts[0] for p in wanted.values()}
    if missing := sorted(set(sequences) - found):
        raise ValueError(f"sequences not in the archive: {missing}")
    return wanted


def save(fetch: Fetch, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(read_member(fetch, info))
    partial.replace(target)


def check(root: Path, sequences: list[str]) -> None:
    """Every sequence needs its ground truth and exactly as many frames as seqinfo.ini says."""
    for seq in sequences:
        info = read_seqinfo(root / seq / "seqinfo.ini")
        frames = len(list((root / seq / "img1").glob("*.jpg")))
        if frames != info.frames or not (root / seq / "gt" / "gt.txt").exists():
            raise RuntimeError(f"{seq}: {frames} of {info.frames} frames (or gt.txt missing)")


def main() -> None:
    params = load_params()["data"]
    remote = HttpRange(params["url"])
    members = list_members(remote.fetch, remote.size)
    wanted = select(members, params["sequences"], params["variant"])
    todo = [(members[name], RAW / rel) for name, rel in wanted.items()
            if not (RAW / rel).exists() or (RAW / rel).stat().st_size != members[name].file_size]
    total = sum(info.compress_size for info, _ in todo)
    print(f"{len(wanted)} files in {len(params['sequences'])} sequences, {len(todo)} to download ({total / 1e9:.2f} GB)")

    start = time.time()
    with ThreadPoolExecutor(params["workers"]) as pool:
        jobs = [pool.submit(save, remote.fetch, info, target) for info, target in todo]
        for done, job in enumerate(jobs, 1):
            job.result()
            if done % 500 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files, {time.time() - start:.0f} s", flush=True)
    check(RAW, params["sequences"])
    print("all sequences complete")


if __name__ == "__main__":
    main()
