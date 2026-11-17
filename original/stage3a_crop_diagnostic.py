
r"""Read-only Stage 3a crop diagnostics for the existing PCB pipeline.

From pcb_v3/Scripts:
    py stage3a_crop_diagnostic.py --root .. --out ..\pcb_processed

Uses the current alignment and crop; does not edit them or write processed PCB images.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from stage1_dataset import load_samples
from stage2_align_pair import process_one
from stage2b_crop import (
    CROP_X0, CROP_Y0, CROP_W, CROP_H,
    find_outside_objects, verify_crop
)

SIDES = ('top', 'bottom', 'left', 'right')

FIELDS = [
    'filename',
    'board',
    'condition',
    'status',
    'error',
    'valid_percent',
    'invalid_total',
    'invalid_top_20px',
    'invalid_bottom_20px',
    'invalid_left_20px',
    'invalid_right_20px',
    'object_touching',
    'min_object_clearance_px'
]


def largest_rectangle(mask):
    """Largest axis-aligned all-true rectangle (x,y,w,h)."""
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best = (0, 0, 0, 0)
    area_best = 0

    for y, line in enumerate(mask):
        heights = np.where(line, heights + 1, 0)
        stack = []

        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x

            while stack and stack[-1][1] > height:
                left, h = stack.pop()
                area = h * (x - left)

                if area > area_best:
                    area_best = area
                    best = (left, y - h + 1, x - left, h)

                start = left

            if height and (
                not stack or stack[-1][1] < height
            ):
                stack.append((start, height))

    return best


def object_clearance(mask, rect):
    x, y, w, h = rect

    if not w or not h:
        return None

    ys, xs = np.where(
        mask[y:y+h, x:x+w] > 0
    )

    if not len(xs):
        return None

    return int(
        min(
            xs.min(),
            ys.min(),
            w - 1 - xs.max(),
            h - 1 - ys.max()
        )
    )


def overlay(d, obj, proposed=None):
    image = d.aligned.copy()
    bad = d.valid == 0

    image[bad] = (
        0.4 * image[bad]
        + 0.6 * np.array([0, 0, 255])
    ).astype(np.uint8)

    cv2.rectangle(
        image,
        (CROP_X0, CROP_Y0),
        (
            CROP_X0 + CROP_W - 1,
            CROP_Y0 + CROP_H - 1
        ),
        (0, 255, 255),
        3
    )

    if proposed:
        x, y, w, h = proposed

        cv2.rectangle(
            image,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 255, 0),
            3
        )

    return cv2.resize(
        image,
        (1024, 576),
        interpolation=cv2.INTER_AREA
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        '--root',
        type=Path,
        required=True
    )

    ap.add_argument(
        '--out',
        type=Path,
        required=True
    )

    args = ap.parse_args()

    samples, problems = load_samples(
        args.root,
        args.out
    )

    if problems:
        raise RuntimeError(
            'Dataset validation failed: '
            + '; '.join(problems[:5])
        )

    out = (
        args.out.resolve()
        / '_stage3'
        / 'crop_diagnostic'
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []
    successful = []

    for i, s in enumerate(samples, 1):
        row = dict.fromkeys(FIELDS, '')

        row.update(
            filename=s.filename,
            board=s.board,
            condition=s.condition
        )

        try:
            d = process_one(s)

            x, y, w, h = (
                CROP_X0,
                CROP_Y0,
                CROP_W,
                CROP_H
            )

            valid = (
                d.valid[y:y+h, x:x+w] > 0
            )

            if valid.shape != (h, w):
                raise ValueError(
                    'Crop lies outside the reference canvas'
                )

            invalid = ~valid

            obj = find_outside_objects(d)
            rep = verify_crop(d, obj)

            clearances = [
                v
                for v in rep.clearance.values()
                if v is not None
            ]

            row.update(
                status='MEASURED',
                valid_percent=f'{100 * valid.mean():.5f}',
                invalid_total=int(invalid.sum()),
                invalid_top_20px=int(
                    invalid[:20].sum()
                ),
                invalid_bottom_20px=int(
                    invalid[-20:].sum()
                ),
                invalid_left_20px=int(
                    invalid[:, :20].sum()
                ),
                invalid_right_20px=int(
                    invalid[:, -20:].sum()
                ),
                object_touching=len(rep.touching),
                min_object_clearance_px=(
                    min(clearances)
                    if clearances
                    else ''
                )
            )

            successful.append(
                (s, d, obj)
            )

        except Exception as e:
            row.update(
                status='FAIL_ALIGNMENT_OR_MEASUREMENT',
                error=f'{type(e).__name__}: {e}'
            )

        rows.append(row)

        print(
            f'[{i:02d}/{len(samples)}] '
            f'{s.board} {s.condition}: '
            f'{row["status"]} '
            f'valid={row["valid_percent"]}% '
            f'error={row["error"]}'
        )

    with (
        out / 'crop_diagnostic.csv'
    ).open(
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)

    for s, d, obj in successful:
        groups['ALL'].append(
            (s, d, obj)
        )

        groups[s.board].append(
            (s, d, obj)
        )

    report = {
        'current_crop_xywh': [
            CROP_X0,
            CROP_Y0,
            CROP_W,
            CROP_H
        ],
        'total_images': len(samples),
        'successfully_aligned': len(successful),
        'failed': [
            r['filename']
            for r in rows
            if r['status'] != 'MEASURED'
        ],
        'groups': {},
        'warning': (
            'Proposed rectangles are diagnostics, '
            'not approved production crops. '
            'Alignment failures were excluded from '
            'intersections; verify object clipping '
            'and board geometry visually.'
        )
    }

    for name, items in groups.items():
        common = np.logical_and.reduce(
            [
                d.valid > 0
                for _, d, _ in items
            ]
        )

        rect = largest_rectangle(common)

        x, y, w, h = rect

        object_clearances = [
            object_clearance(obj, rect)
            for _, _, obj in items
        ]

        near = sum(
            c is not None and c < 20
            for c in object_clearances
        )

        report['groups'][name] = {
            'images': len(items),
            'largest_all_valid_rectangle_xywh': list(rect),
            'rectangle_area_percent_of_current_crop': round(
                100 * w * h / (CROP_W * CROP_H),
                2
            ),
            'images_with_detected_objects_within_20px_of_proposed_border': near,
            'minimum_detected_object_clearance_px': min(
                (
                    c
                    for c in object_clearances
                    if c is not None
                ),
                default=None
            ),
            'note': (
                'Largest area does not necessarily '
                'preserve all overhanging components '
                'or the current coordinate framing.'
            )
        }

        print(
            f'{name}: {len(items)} aligned, '
            f'candidate valid rectangle '
            f'x={x}, y={y}, w={w}, h={h}; '
            f'object-edge warnings={near}'
        )

    # Save representative visual overlays.
    for s, d, obj in successful[:3]:
        filename = (
            f'overlay_{s.board}_'
            f'{s.filename.rsplit(".", 1)[0]}.png'
        )

        cv2.imwrite(
            str(out / filename),
            overlay(
                d,
                obj,
                report['groups']['ALL'][
                    'largest_all_valid_rectangle_xywh'
                ]
            )
        )

    (
        out / 'crop_summary.json'
    ).write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding='utf-8'
    )

    print(
        f'\nSaved reports and sample overlays in: {out}'
    )

    print(
        'No source or processed dataset images were modified.'
    )


if __name__ == '__main__':
    main()
