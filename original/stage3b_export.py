"""QC-gated export of aligned PCB crops. Requires the existing PCB geometry.

Run after Stage 3a, with an image dataset containing metadata.csv and raw/.
This is NOT a universal alignment model for unseen PCB designs.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from stage1_dataset import load_samples
from stage2_align_pair import process_one
from stage2b_crop import CROP_X0, CROP_Y0, CROP_W, CROP_H, find_outside_objects, verify_crop
from pcb_core import Grid, Calibration, measure, evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True, help='Folder with raw/ and metadata.csv')
    ap.add_argument('--out', type=Path, required=True, help='Processed output folder (outside raw/)')
    ap.add_argument('--calibration', type=Path, help='Stage 3a good-only calibration JSON (default: OUT/_stage3/calibration_good_only.json)')
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    calpath = args.calibration or out / '_stage3' / 'calibration_good_only.json'
    if not calpath.is_file():
        ap.error(f'Calibration not found: {calpath}. Run stage3a_probe.py first.')
    data = json.loads(calpath.read_text(encoding='utf-8'))
    cal = Calibration(Grid(**data['grid']), data['edges'], data['thresholds'],
                      data['template_check'], data['calibrated_on'])
    samples, problems = load_samples(root, out)
    if problems:
        raise RuntimeError('Dataset validation failed: ' + '; '.join(problems[:5]))
    export = out / 'dataset_aligned'
    rows = []
    failures = 0
    for i, s in enumerate(samples, 1):
        row = dict(filename=s.filename, board=s.board, condition=s.condition,
                   image_path='', mask_path='', valid_fraction='', status='', reason='')
        try:
            result = evaluate(measure(s), cal)
            row['status'], row['reason'] = result.status, result.reason
            if result.status != 'OK':
                failures += 1
            else:
                d = process_one(s)
                x, y, w, h = CROP_X0, CROP_Y0, CROP_W, CROP_H
                img = d.aligned[y:y+h, x:x+w]
                valid = d.valid[y:y+h, x:x+w]
                if img.shape[:2] != (h, w) or valid.shape != (h, w):
                    raise ValueError('Crop exceeds canvas dimensions')
                # Preserve invalid background pixels and their real-pixel mask.
                # Dataset requires prior visual review of the board region.
                folder = export / s.board / s.condition
                folder.mkdir(parents=True, exist_ok=True)
                stem = Path(s.filename).stem
                image_path = folder / (stem + '.png')
                mask_path = folder / (stem + '_valid_mask.png')
                if not cv2.imwrite(str(image_path), img):
                    raise IOError('Could not write cropped image')
                if not cv2.imwrite(str(mask_path), valid):
                    image_path.unlink(missing_ok=True)
                    raise IOError('Could not write validity mask')
                row['image_path'] = image_path.relative_to(export).as_posix()
                row['mask_path'] = mask_path.relative_to(export).as_posix()
                row['valid_fraction'] = float(np.mean(valid > 0))
        except Exception as exc:
            failures += (row['status'] == 'OK' or not row['status'])
            row['status'] = 'FAIL_EXPORT'
            row['reason'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print(f'[{i}/{len(samples)}] {row["status"]}: {s.filename} {row["reason"]}')
    export.mkdir(parents=True, exist_ok=True)
    with (export / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                                ['filename','board','condition','image_path','mask_path','valid_fraction','status','reason'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {sum(r["status"] == "OK" for r in rows)}/{len(rows)}; failures: {failures}')
    print('Manifest:', export / 'manifest.csv')
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
