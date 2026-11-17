
"""Generate training-only PCB augmentations from an exported QC manifest."""

import argparse
import csv
from pathlib import Path
import shutil
import cv2
import numpy as np


def main():

    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--train-board', choices=['B01', 'B02'], default='B01')
    p.add_argument('--seed', type=int, default=42)

    a = p.parse_args()

    root = a.manifest.resolve().parent
    out = a.output.resolve()

    # Protect the original dataset
    if (
        out.exists()
        or out == root
        or root in out.parents
        or out in root.parents
    ):
        p.error(
            'Output must be a new directory separate from the input dataset'
        )

    rng = np.random.default_rng(a.seed)

    # ---------------------------------------------------------
    # 2. Read the preprocessing manifest
    # ---------------------------------------------------------

    with a.manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = [
            r for r in csv.DictReader(f)
            if r['status'] == 'OK'
        ]

    if not rows or {r['board'] for r in rows} != {'B01', 'B02'}:
        p.error('Expected accepted images from both B01 and B02')

    # ---------------------------------------------------------
    # 3. Image reading and writing functions
    # ---------------------------------------------------------

    def source(s):

        path = (root / s).resolve()

        if root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid source: {s}')

        return path

    def read(path, flag):

        im = cv2.imdecode(
            np.fromfile(path, dtype=np.uint8),
            flag
        )

        if im is None:
            raise ValueError(f'Cannot decode {path}')

        return im

    def write(path, im):

        ok, buf = cv2.imencode('.png', im)

        if not ok:
            raise IOError(f'Cannot encode {path}')

        buf.tofile(path)

    # Validate all source files before processing
    for r in rows:
        source(r['image_path'])
        source(r['mask_path'])

    out.mkdir(parents=True)

    records = []

    # ---------------------------------------------------------
    # 4. Process accepted images
    # ---------------------------------------------------------

    for i, r in enumerate(rows):

        train = r['board'] == a.train_board
        split = 'train' if train else 'test'

        ip = source(r['image_path'])
        mp = source(r['mask_path'])

        im = read(ip, cv2.IMREAD_COLOR)
        mask = read(mp, cv2.IMREAD_GRAYSCALE)

        if im.shape[:2] != mask.shape:
            raise ValueError(f'Mask size mismatch: {ip}')

        folder = out / split / r['condition'] / r['board']

        folder.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------
        # 5. Augmentation function
        # -----------------------------------------------------

        def save(tag, angle=0, brightness=1, sigma=0):

            name = f'{i:03d}_{ip.stem}_{tag}'

            dst = folder / (name + '.png')
            dm = folder / (name + '_valid_mask.png')

            if tag == 'original':

                shutil.copyfile(ip, dst)
                shutil.copyfile(mp, dm)

            else:

                h, w = mask.shape

                center = ((w - 1) / 2, (h - 1) / 2)

                # Rotation matrix
                mat = cv2.getRotationMatrix2D(
                    center,
                    angle,
                    1
                )

                if angle:

                    # Fit the rotated image inside the original
                    # dimensions to prevent component cropping.

                    c = abs(mat[0, 0])
                    s = abs(mat[0, 1])

                    scale = min(
                        (w - 1) / (c * (w - 1) + s * (h - 1)),
                        (h - 1) / (s * (w - 1) + c * (h - 1))
                    )

                    mat = cv2.getRotationMatrix2D(
                        center,
                        angle,
                        scale
                    )

                # Apply rotation to image
                v = cv2.warpAffine(
                    im,
                    mat,
                    (w, h),
                    flags=cv2.INTER_LINEAR
                )

                # Apply the same rotation to the validity mask
                m = cv2.warpAffine(
                    mask,
                    mat,
                    (w, h),
                    flags=cv2.INTER_NEAREST
                )

                # Brightness adjustment
                v = v.astype(np.float32) * brightness

                # Gaussian noise
                if sigma:

                    noise = rng.normal(
                        0,
                        sigma,
                        (h, w, 1)
                    ).astype(np.float32)

                    v += noise

                # Keep pixel values within the valid range
                v = np.clip(v, 0, 255).astype(np.uint8)

                # Remove invalid image regions
                v[m == 0] = 0

                # Save augmented image and mask
                write(dst, v)
                write(dm, m)

            # Store image metadata
            records.append(
                dict(
                    split=split,
                    board=r['board'],
                    condition=r['condition'],
                    source_filename=r['filename'],
                    variant=tag,
                    angle=angle,
                    brightness=brightness,
                    noise_sigma=sigma,
                    image_path=dst.relative_to(out).as_posix(),
                    mask_path=dm.relative_to(out).as_posix()
                )
            )

        # Preserve the original image
        save('original')

        # -----------------------------------------------------
        # 6. Generate training-only augmentations
        # -----------------------------------------------------

        if train:

            # Format: (rotation, brightness, noise_sigma)

            variants = [

                # Rotation
                (-5, 1, 0),
                (5, 1, 0),

                # Brightness
                (0, 0.9, 0),
                (0, 1.1, 0),

                # Gaussian noise
                (0, 1, 3),
                (0, 1, 7),

                # Combined transformations
                (-8, 0.9, 3),
                (8, 1.1, 3),
                (-3, 1.05, 5),
                (3, 0.95, 5)
            ]

            for j, args in enumerate(variants, 1):

                save(f'aug{j:02d}', *args)

    # ---------------------------------------------------------
    # 7. Export augmentation manifest
    # ---------------------------------------------------------

    with (out / 'manifest.csv').open(
        'w',
        newline='',
        encoding='utf-8'
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    # ---------------------------------------------------------
    # 8. Final report
    # ---------------------------------------------------------

    for split in ('train', 'test'):

        print(
            split,
            sum(r['split'] == split for r in records)
        )

    missing = {

        r['condition']
        for r in records
        if r['split'] == 'train'

    } - {

        r['condition']
        for r in records
        if r['split'] == 'test'
    }

    if missing:

        print(
            'WARNING: classes missing from test:',
            ', '.join(sorted(missing))
        )

    print(
        'Total:',
        len(records),
        'Manifest:',
        out / 'manifest.csv'
    )


if __name__ == '__main__':
    main()
