"""
Stage 3a - Representative test (NOT the full batch).

  1. Measure ALL 16 good images + a small representative set of defective images
     (default: 1 per defect category per board = 12 images).
  2. Calibrate QC limits from the 16 GOOD images only, then freeze them.
  3. Evaluate every image with the frozen limits (defects never influence the limits).
  4. Check whether ONE shared reference frame suits both boards.
  5. Write: CSV report, calibration JSON, contact sheet.

No processed dataset images are written here; that is Stage 3b, after this passes.

Run from pcb_v3\\Scripts:
    py stage3a_probe.py --root .. --out ..\\pcb_processed
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from pcb_core import (CalibrationError, Measure, THUMB_W, board_comparison, calibrate, evaluate, measure)
from stage1_dataset import Sample, load_samples

CSV_FIELDS = ["filename", "board", "condition", "component", "session", "role", "status", "reason",
              "hole_top", "hole_bottom", "hole_left", "hole_right", "keystone_w_pct", "keystone_h_pct",
              "rot_top_deg", "rot_bottom_deg", "affine_rms_px",
              "edge_left", "edge_right", "edge_top", "edge_bottom",
              "edge_dev_left", "edge_dev_right", "edge_dev_top", "edge_dev_bottom",
              "n_dots", "grid_off_x", "grid_off_y", "grid_spread_x", "grid_spread_y",
              "grid_drift_x", "grid_drift_y",
              "crop_valid_fraction", "crop_touching", "crop_min_clearance", "outside_objects",
              "relative_path"]


def select_probe(samples: list[Sample], per_category: int) -> list[Sample]:
    """Deterministic: first N (by filename) of every defect category on every board."""
    out = []
    keys = sorted({(s.condition, s.board) for s in samples if not s.is_good})
    for cond, board in keys:
        group = sorted([s for s in samples if s.condition == cond and s.board == board], key=lambda s: s.filename)
        out += group[:per_category]
    return out


def write_csv(path: Path, results: list[Measure], roles: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            s = r.sample
            row = {"filename": s.filename, "board": s.board, "condition": s.condition, "component": s.component,
                   "session": s.session, "role": roles[s.filename], "status": r.status, "reason": r.reason,
                   "relative_path": s.rel_path.as_posix()}
            for k in CSV_FIELDS:
                if k in r.m and k not in row:
                    v = r.m[k]
                    row[k] = "" if v is None else (f"{v:.3f}" if isinstance(v, float) else v)
            w.writerow(row)


def contact_sheet(results: list[Measure], cols: int = 7) -> np.ndarray:
    th = 229
    cell_w, cell_h = THUMB_W + 12, th + 12 + 46
    rows = math.ceil(len(results) / cols)
    sheet = np.full((rows * cell_h + 40, cols * cell_w, 3), 30, np.uint8)
    cv2.putText(sheet, "Contact sheet (cropped, aligned, original colours).  green = OK   orange = FAIL_QC   red = FAIL_ALIGNMENT",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    colour = {"OK": (60, 200, 60), "FAIL_QC": (0, 150, 255), "FAIL_ALIGNMENT": (0, 0, 230)}
    for i, r in enumerate(results):
        x, y = (i % cols) * cell_w + 6, (i // cols) * cell_h + 46
        t = r.thumb[:th]
        sheet[y + 4:y + 4 + t.shape[0], x + 4:x + 4 + t.shape[1]] = t
        c = colour.get(r.status, (200, 200, 200))
        cv2.rectangle(sheet, (x, y), (x + THUMB_W + 8, y + th + 8), c, 4)
        s = r.sample
        cv2.putText(sheet, f"{s.board} {s.condition}", (x + 2, y + th + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{r.status}  {s.timestamp[9:15]}", (x + 2, y + th + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=1, help="defect images per category per board (default 1)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    samples, problems = load_samples(args.root, args.out)
    if problems:
        print(f"Stage 1 validation reported {len(problems)} problem(s); fix them first, e.g.: {problems[0]}")
        return 1
    good = [s for s in samples if s.is_good]
    probe = select_probe(samples, args.per_category)
    batch = good + probe
    roles = {s.filename: ("calibration+test" if s.is_good else "probe (defect)") for s in batch}
    out_dir = args.out.resolve() / "_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"STEP 1  measuring {len(good)} good + {len(probe)} defective images (this takes a few minutes)")
    results, t0 = [], time.time()
    for i, s in enumerate(batch, 1):
        r = measure(s)
        results.append(r)
        flag = "" if r.status == "MEASURED" else f"   <-- {r.status}: {r.reason}"
        print(f"  [{i:2d}/{len(batch)}] {s.board} {s.condition:20s} {s.filename[-26:]}{flag}")
    print(f"  done in {time.time() - t0:.0f}s")

    good_res = [r for r in results if r.sample.is_good]
    print("\nSTEP 2  calibrating QC limits from GOOD images only")
    try:
        cal = calibrate(good_res)
    except CalibrationError as e:
        print(f"CALIBRATION FAILED: {e}")
        write_csv(out_dir / "probe_report.csv", results, roles)
        cv2.imwrite(str(out_dir / "contact_sheet_probe.png"), contact_sheet(results))
        return 1
    g = cal.grid
    print(f"  used {len(cal.used_files)}/{len(good)} good images")
    print(f"  perforation pitch x/y = {g.px:.2f}/{g.py:.2f}px (periodicity strength {g.strength_x:.2f}/{g.strength_y:.2f})")
    print(f"  board edges (ref frame, median of good): { {k: round(v, 1) for k, v in cal.edges.items()} }")
    print(f"  frozen limits: { {k: round(v, 2) for k, v in cal.thresholds.items()} }")
    tc = cal.template_check
    print(f"  template check: good-only hole spacing {tc['good_median_hole_width']:.0f}x{tc['good_median_hole_height']:.0f} "
          f"vs template {tc['template_width']}x{tc['template_height']} "
          f"({tc['width_dev_pct']:+.2f}% / {tc['height_dev_pct']:+.2f}%)")

    print("\nSTEP 3  evaluating every image with the frozen limits")
    for r in results:
        evaluate(r, cal)
    for r in results:
        s, m = r.sample, r.m
        extra = ""
        if "grid_off_x" in m:
            extra = (f" off({m['grid_off_x']:+.1f},{m['grid_off_y']:+.1f}) drift({m['grid_drift_x']:+.1f},{m['grid_drift_y']:+.1f})"
                     f" edgeDev {max(m[f'edge_dev_{k}'] for k in cal.edges):.0f} clr {m['crop_min_clearance']}")
        print(f"  {r.status:14s} {s.board} {s.condition:20s}{extra}" + (f"\n      -> {r.reason}" if r.status != "OK" else ""))

    print("\nSTEP 4  shared reference-frame comparisons (good images only)")
    rows = board_comparison(good_res)
    if not rows:
        print("  not enough good images on both boards to compare")
    shared_ok = bool(rows) and all(row[7] for row in rows)
    for ref_board, other_board, name, a, b, diff, lim, ok in rows:
        print(f"  [{'PASS' if ok else 'WARN'}] {name:28s} {ref_board} {a:9.2f}  {other_board} {b:9.2f}  diff {diff:+7.2f} (limit {lim})")
    if rows:
        print("  verdict:", "a single shared frame looks appropriate" if shared_ok else
              "boards differ - consider separate per-board frames (decide after reviewing the contact sheet)")

    write_csv(out_dir / "probe_report.csv", results, roles)
    cal_json = {"grid": asdict(cal.grid), "edges": cal.edges, "thresholds": cal.thresholds,
                "template_check": cal.template_check, "calibrated_on": cal.used_files,
                "note": "calibrated on GOOD images only; defective images never used"}
    (out_dir / "calibration_good_only.json").write_text(json.dumps(cal_json, indent=2))
    sheet = contact_sheet(sorted(results, key=lambda r: (not r.sample.is_good, r.sample.condition, r.sample.board, r.sample.filename)))
    cv2.imwrite(str(out_dir / "contact_sheet_probe.png"), sheet)
    print(f"\nSaved in {out_dir}:\n  probe_report.csv\n  calibration_good_only.json\n  contact_sheet_probe.png")

    bad = [r for r in results if r.status != "OK"]
    print(f"\nRESULT: {len(results) - len(bad)}/{len(results)} OK, {len(bad)} not OK")
    if args.show:
        cv2.imshow("contact sheet", sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())