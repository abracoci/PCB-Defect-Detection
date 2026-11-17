"""
Stage 1 - Dataset index, path safety and output mirroring.

Reads metadata.csv, validates every row against the files on disk, and maps each
raw image to an output path that mirrors the raw directory layout
(raw/good/B01/x.jpg -> <out_root>/good/B01/x.png).

This stage NEVER writes an image. It only reads, validates and reports.

Usage (from the folder that contains pcb_v3/ ... or point --root at pcb_v3):
    python stage1_dataset.py --root path/to/pcb_v3 --out path/to/pcb_processed
    python stage1_dataset.py --root path/to/pcb_v3 --out path/to/pcb_processed --check-images
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EXPECTED_W, EXPECTED_H = 2048, 1152
REQUIRED_COLUMNS = [
    "filename", "board", "condition", "component", "session", "timestamp",
    "led_red", "led_green", "led_blue", "shutter_us", "gain", "awb_gains",
    "lens_position", "relative_path",
]


@dataclass(frozen=True)
class Sample:
    filename: str
    board: str
    condition: str
    component: str
    session: str
    timestamp: str
    raw_path: Path      # absolute path of the raw image (read-only use)
    rel_path: Path      # path relative to dataset root, e.g. raw/good/B01/x.jpg
    out_path: Path      # where the processed PNG will go

    @property
    def is_good(self) -> bool:
        return self.condition == "good"


def expected_subdir(board: str, condition: str) -> Path:
    """Directory (relative to raw/) that this row should live in."""
    if condition == "good":
        return Path("good") / board
    return Path("defects") / condition / board


def load_samples(root: Path, out_root: Path, csv_name: str = "metadata.csv"):
    """Return (samples, problems). Problems are strings; empty list means clean."""
    root, out_root = root.resolve(), out_root.resolve()
    raw_dir = (root / "raw").resolve()
    problems: list[str] = []

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw/ folder not found under {root}")
    # Safety: never write to the dataset root or anywhere beneath it.
    # Also reject output paths that are ancestors of the dataset to avoid mixing trees.
    if out_root == root or root in out_root.parents or out_root in root.parents:
        raise ValueError(f"Unsafe output dir {out_root}: must be separate from dataset {root}.")

    csv_path = root / csv_name
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"metadata.csv is missing columns: {missing}")
        rows = list(reader)

    samples: list[Sample] = []
    seen: set[str] = set()
    for i, r in enumerate(rows, start=2):  # start=2 -> spreadsheet row number
        rel = Path(r["relative_path"].replace("\\", "/"))
        raw_path = (root / rel).resolve()
        tag = f"row {i} ({r['filename']})"

        if rel.parts[0] != "raw":
            problems.append(f"{tag}: relative_path does not start with raw/")
        if raw_dir not in raw_path.parents:
            problems.append(f"{tag}: path escapes raw/")
            continue
        if not raw_path.is_file():
            problems.append(f"{tag}: image file not found: {raw_path}")
        if raw_path.name != r["filename"]:
            problems.append(f"{tag}: filename column != basename of relative_path")
        if r["filename"] in seen:
            problems.append(f"{tag}: duplicate filename")
        seen.add(r["filename"])

        want = expected_subdir(r["board"], r["condition"])
        actual = raw_path.parent.relative_to(raw_dir)
        if actual != want:
            problems.append(f"{tag}: folder is raw/{actual}, labels imply raw/{want}")

        out_path = out_root / raw_path.relative_to(raw_dir).with_suffix(".png")
        samples.append(Sample(
            filename=r["filename"], board=r["board"], condition=r["condition"],
            component=r["component"], session=r["session"], timestamp=r["timestamp"],
            raw_path=raw_path, rel_path=rel, out_path=out_path,
        ))
    return samples, problems


def check_images(samples: list[Sample]) -> list[str]:
    """Open every image (read-only) and verify it decodes at the expected size."""
    import cv2
    problems = []
    for s in samples:
        img = cv2.imread(str(s.raw_path), cv2.IMREAD_COLOR)
        if img is None:
            problems.append(f"{s.filename}: could not be decoded")
        elif img.shape[:2] != (EXPECTED_H, EXPECTED_W):
            problems.append(f"{s.filename}: size {img.shape[1]}x{img.shape[0]}, "
                            f"expected {EXPECTED_W}x{EXPECTED_H}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="dataset root (contains metadata.csv and raw/)")
    ap.add_argument("--out", type=Path, required=True, help="NEW output root for processed images")
    ap.add_argument("--check-images", action="store_true", help="also decode every image and check its size")
    args = ap.parse_args()

    samples, problems = load_samples(args.root, args.out)
    if args.check_images:
        problems += check_images([s for s in samples if s.raw_path.is_file()])

    print(f"Rows loaded: {len(samples)}")
    print("By condition:", dict(Counter(s.condition for s in samples)))
    print("By board    :", dict(Counter(s.board for s in samples)))
    print("Good per board:", dict(Counter(s.board for s in samples if s.is_good)))
    print("\nExample mapping:")
    for s in samples[:3]:
        print(f"  {s.rel_path}\n    -> {s.out_path}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK: metadata, paths and labels are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())