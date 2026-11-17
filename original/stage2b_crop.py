
"""
Stage 2b - Consistent cropping and quality verification.

Aligns one good/defective PCB pair, applies a shared crop,
and verifies that no invalid pixels or outside-board objects
are cut off.

Crop margins can be configured from the command line.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stage1_dataset import load_samples
from stage2_align_pair import (
    AlignmentError,
    Detection,
    DEFAULT_DEFECT,
    DEFAULT_GOOD,
    TPL_H,
    TPL_W,
    TPL_X,
    TPL_Y,
    TEMPLATE,
    _label as label_panel,
    pick,
    process_one,
)


# ============================================================
# CROP CONFIGURATION
# ============================================================

MARGIN_LEFT = 240
MARGIN_TOP = 210
MARGIN_RIGHT = 136
MARGIN_BOTTOM = 105

CROP_X0 = TPL_X - MARGIN_LEFT
CROP_Y0 = TPL_Y - MARGIN_TOP
CROP_W = TPL_W + MARGIN_LEFT + MARGIN_RIGHT
CROP_H = TPL_H + MARGIN_TOP + MARGIN_BOTTOM

EDGE_BAND = 12
OBJECT_RESIDUAL = 28
OBJECT_MIN_AREA = 120
QC_OBJECT_CLEARANCE_PX = 20
QC_MIN_VALID_FRACTION = 1.0


def configure_crop(left, top, right, bottom):
    """Configure one consistent crop in the reference frame."""
    global MARGIN_LEFT, MARGIN_TOP
    global MARGIN_RIGHT, MARGIN_BOTTOM
    global CROP_X0, CROP_Y0, CROP_W, CROP_H

    if any(v < 0 for v in (left, top, right, bottom)):
        raise ValueError("Crop margins cannot be negative.")

    MARGIN_LEFT = left
    MARGIN_TOP = top
    MARGIN_RIGHT = right
    MARGIN_BOTTOM = bottom

    CROP_X0 = TPL_X - left
    CROP_Y0 = TPL_Y - top
    CROP_W = TPL_W + left + right
    CROP_H = TPL_H + top + bottom


def validate_crop_bounds(img):
    """Check that the crop stays inside the aligned canvas."""
    height, width = img.shape[:2]

    if (
        CROP_X0 < 0
        or CROP_Y0 < 0
        or CROP_W <= 0
        or CROP_H <= 0
        or CROP_X0 + CROP_W > width
        or CROP_Y0 + CROP_H > height
    ):
        raise ValueError(
            "Crop rectangle leaves the aligned canvas. "
            "Adjust the crop margins."
        )


def crop(img: np.ndarray) -> np.ndarray:
    validate_crop_bounds(img)

    return img[
        CROP_Y0:CROP_Y0 + CROP_H,
        CROP_X0:CROP_X0 + CROP_W
    ].copy()


# ============================================================
# DETECT OBJECTS OUTSIDE THE BOARD
# ============================================================

def find_outside_objects(d: Detection) -> np.ndarray:
    """
    Detect objects outside the board using a fitted
    background surface.

    This function is used only for quality verification.
    It does not modify the original image.
    """
    H, W = d.valid.shape

    L, R, T, B = (
        d.edges[k]
        for k in ("left", "right", "top", "bottom")
    )

    outside = d.valid > 0

    outside[
        max(T - EDGE_BAND, 0):B + EDGE_BAND + 1,
        max(L - EDGE_BAND, 0):R + EDGE_BAND + 1
    ] = False

    yy, xx = np.mgrid[0:H, 0:W]

    X = np.c_[
        np.ones(H * W),
        xx.ravel() / 1e3,
        yy.ravel() / 1e3,
        (xx.ravel() / 1e3) ** 2,
        xx.ravel() * yy.ravel() / 1e6,
        (yy.ravel() / 1e3) ** 2
    ]

    pix = d.aligned.reshape(-1, 3).astype(np.float32)
    m = outside.ravel()

    resid = np.zeros((H * W, 3), np.float32)

    for c in range(3):
        keep = m.copy()

        for _ in range(5):
            coef = np.linalg.lstsq(
                X[keep][::7],
                pix[keep][::7, c],
                rcond=None
            )[0]

            r = pix[:, c] - X @ coef

            keep = m & (
                np.abs(r) < 2.5 * r[keep].std()
            )

        resid[:, c] = r

    obj = (
        (
            np.abs(resid).max(1) > OBJECT_RESIDUAL
        ).reshape(H, W) & outside
    ).astype(np.uint8) * 255

    obj = cv2.morphologyEx(
        obj,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    obj = cv2.morphologyEx(
        obj,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8)
    )

    n, lab, st, _ = cv2.connectedComponentsWithStats(obj)

    out = np.zeros_like(obj)

    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= OBJECT_MIN_AREA:
            out[lab == i] = 255

    return out


# ============================================================
# CROP QUALITY REPORT
# ============================================================

@dataclass
class CropReport:
    valid_fraction: float
    objects: list
    clearance: dict
    touching: list
    board_margin: dict
    overhang: dict


def verify_crop(
    d: Detection,
    obj_mask: np.ndarray
) -> CropReport:

    x0 = CROP_X0
    y0 = CROP_Y0
    x1 = CROP_X0 + CROP_W
    y1 = CROP_Y0 + CROP_H

    valid_frac = float(
        (d.valid[y0:y1, x0:x1] > 0).mean()
    )

    n, lab, st, _ = cv2.connectedComponentsWithStats(
        obj_mask
    )

    objs = []
    touching = []

    for i in range(1, n):
        x, y, w, h, a = (
            int(v) for v in st[i]
        )

        objs.append((x, y, w, h, a))

        intersects_crop = (
            x < x1
            and x + w > x0
            and y < y1
            and y + h > y0
        )

        touches_border = (
            x <= x0 + 2
            or y <= y0 + 2
            or x + w >= x1 - 2
            or y + h >= y1 - 2
        )

        if intersects_crop and touches_border:
            touching.append((x, y, w, h, a))

    ys, xs = np.where(
        obj_mask[y0:y1, x0:x1] > 0
    )

    clearance = {
        "left": None,
        "top": None,
        "right": None,
        "bottom": None
    }

    if len(xs):
        clearance = {
            "left": int(xs.min()),
            "top": int(ys.min()),
            "right": int(CROP_W - 1 - xs.max()),
            "bottom": int(CROP_H - 1 - ys.max())
        }

    e = d.edges

    board_margin = {
        "left": e["left"] - x0,
        "top": e["top"] - y0,
        "right": x1 - e["right"],
        "bottom": y1 - e["bottom"]
    }

    overhang = {
        "left": max(
            [e["left"] - o[0] for o in objs] + [0]
        ),
        "top": max(
            [e["top"] - o[1] for o in objs] + [0]
        ),
        "right": max(
            [o[0] + o[2] - e["right"] for o in objs] + [0]
        ),
        "bottom": max(
            [o[1] + o[3] - e["bottom"] for o in objs] + [0]
        )
    }

    return CropReport(
        valid_frac,
        objs,
        clearance,
        touching,
        board_margin,
        overhang
    )


def run_qc(rep: dict[str, CropReport]):
    qc = []

    for tag, r in rep.items():
        qc.append((
            f"{tag}: crop contains only real pixels",
            r.valid_fraction >= QC_MIN_VALID_FRACTION,
            f"{100 * r.valid_fraction:.2f}% valid"
        ))

        near = {
            k: v
            for k, v in r.clearance.items()
            if v is not None
            and v < QC_OBJECT_CLEARANCE_PX
        }

        qc.append((
            f"{tag}: outside objects clear of crop edge",
            not r.touching and not near,
            f"touching={len(r.touching)}, "
            f"L/T/R/B="
            f"{r.clearance['left']}/"
            f"{r.clearance['top']}/"
            f"{r.clearance['right']}/"
            f"{r.clearance['bottom']} "
            f"(min {QC_OBJECT_CLEARANCE_PX})"
        ))

    return qc


# ============================================================
# DIAGNOSTIC VISUALIZATION
# ============================================================

def draw_crop_box(img, color=(0, 255, 255)):
    out = img.copy()

    cv2.rectangle(
        out,
        (CROP_X0, CROP_Y0),
        (CROP_X0 + CROP_W, CROP_Y0 + CROP_H),
        color,
        4
    )

    return out


def annotate(d: Detection, obj_mask):
    out = draw_crop_box(d.aligned)

    cv2.polylines(
        out,
        [d.quad_ref.astype(np.int32)],
        True,
        (0, 255, 0),
        3
    )

    cnts, _ = cv2.findContours(
        obj_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        out,
        cnts,
        -1,
        (255, 0, 255),
        3
    )

    return out


def red_cyan(a: Detection, b: Detection):
    ga = cv2.cvtColor(
        a.aligned,
        cv2.COLOR_BGR2GRAY
    )

    gb = cv2.cvtColor(
        b.aligned,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.merge([gb, gb, ga])


def text_panel(lines, h=382, w=680):
    panel = np.zeros(
        (h, w, 3),
        np.uint8
    )

    for i, t in enumerate(lines):
        if "[PASS]" in t:
            col = (0, 255, 0)
        elif "[FAIL]" in t:
            col = (0, 0, 255)
        else:
            col = (230, 230, 230)

        cv2.putText(
            panel,
            t[:115],
            (8, 20 + 16 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            col,
            1,
            cv2.LINE_AA
        )

    return panel


def build_overview(a, b, masks, reps, qc):
    blend = draw_crop_box(
        cv2.addWeighted(
            a.aligned,
            0.5,
            b.aligned,
            0.5,
            0
        )
    )

    rc = draw_crop_box(red_cyan(a, b))

    info = [
        f"CROP: x {CROP_X0}..{CROP_X0 + CROP_W} "
        f"y {CROP_Y0}..{CROP_Y0 + CROP_H} "
        f"= {CROP_W}x{CROP_H}px",

        f"Margins L{MARGIN_LEFT} "
        f"T{MARGIN_TOP} "
        f"R{MARGIN_RIGHT} "
        f"B{MARGIN_BOTTOM}",
        ""
    ]

    for tag, r in reps.items():
        info += [
            f"{tag.upper()} board margin L/T/R/B: "
            + "/".join(
                str(r.board_margin[k])
                for k in (
                    "left", "top", "right", "bottom"
                )
            ),

            f"Object overhang L/T/R/B: "
            + "/".join(
                str(r.overhang[k])
                for k in (
                    "left", "top", "right", "bottom"
                )
            ),

            f"Outside objects: {len(r.objects)}",
            ""
        ]

    info.append("QUALITY CHECKS")

    info += [
        f"  [{'PASS' if ok else 'FAIL'}] "
        f"{n}: {detail}"
        for n, ok, detail in qc
    ]

    legend = [
        "Red/cyan diagnostic overlay:",
        "Grey = same in both images",
        "Cyan = dark only in GOOD image",
        "Red = dark only in DEFECTIVE image",
        "Yellow = crop rectangle",
        "Green = board edges",
        "Magenta = outside objects"
    ]

    row1 = np.hstack([
        label_panel(
            annotate(a, masks["good"]),
            "1. ALIGNED good + crop"
        ),
        label_panel(
            annotate(b, masks["defect"]),
            "2. ALIGNED defective + crop"
        ),
        text_panel(info)
    ])

    row2 = np.hstack([
        label_panel(
            blend,
            "4. 50/50 blend + crop"
        ),
        label_panel(
            rc,
            "5. Red/cyan overlay"
        ),
        text_panel(legend)
    ])

    return np.vstack([row1, row2])


def build_side_by_side(ca, cb):
    both = np.hstack([ca, cb])

    w = 1700

    small = cv2.resize(
        both,
        (
            w,
            int(both.shape[0] * w / both.shape[1])
        ),
        interpolation=cv2.INTER_AREA
    )

    bar = np.zeros(
        (30, w, 3),
        np.uint8
    )

    cv2.putText(
        bar,
        "CROPPED good (left) | "
        "CROPPED defective (right)",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return np.vstack([bar, small])


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    ap.add_argument(
        "--root",
        type=Path,
        required=True
    )

    ap.add_argument(
        "--out",
        type=Path,
        required=True
    )

    ap.add_argument(
        "--good",
        default=DEFAULT_GOOD
    )

    ap.add_argument(
        "--defect",
        default=DEFAULT_DEFECT
    )

    ap.add_argument(
        "--show",
        action="store_true"
    )

    ap.add_argument(
        "--margin-left",
        type=int,
        default=240
    )

    ap.add_argument(
        "--margin-top",
        type=int,
        default=210
    )

    ap.add_argument(
        "--margin-right",
        type=int,
        default=136
    )

    ap.add_argument(
        "--margin-bottom",
        type=int,
        default=105
    )

    args = ap.parse_args()

    configure_crop(
        args.margin_left,
        args.margin_top,
        args.margin_right,
        args.margin_bottom
    )

    # Load dataset using the existing Stage 1 validation.
    samples, problems = load_samples(
        args.root,
        args.out
    )

    if problems:
        print(
            f"WARNING: Stage 1 reported "
            f"{len(problems)} problem(s). "
            f"Example: {problems[0]}"
        )

    good = pick(samples, args.good)
    defect = pick(samples, args.defect)

    debug_dir = (
        args.out.resolve() / "_stage2_debug"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dets = {}
    failed = False

    for tag, sample in (
        ("good", good),
        ("defect", defect)
    ):
        try:
            dets[tag] = process_one(sample)

            print(
                f"[ok] {tag} aligned: "
                f"{sample.filename}"
            )

        except AlignmentError as exc:
            failed = True

            print(
                f"[FAIL] {tag}: "
                f"{sample.filename}: {exc}"
            )

    if failed:
        print(
            "\nALIGNMENT FAILED. "
            "No cropped images saved."
        )
        return 2

    # Verify the crop rectangle fits both images.
    try:
        for detection in dets.values():
            validate_crop_bounds(
                detection.aligned
            )

    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    masks = {
        tag: find_outside_objects(d)
        for tag, d in dets.items()
    }

    reports = {
        tag: verify_crop(d, masks[tag])
        for tag, d in dets.items()
    }

    qc = run_qc(reports)

    print(
        f"\nCrop rectangle: "
        f"x={CROP_X0} "
        f"y={CROP_Y0} "
        f"w={CROP_W} "
        f"h={CROP_H}"
    )

    for tag, report in reports.items():
        print(
            f"\n{tag}: "
            f"{len(report.objects)} "
            f"outside-board objects"
        )

        print(
            f"Overhang: {report.overhang}"
        )

        print(
            f"Clearance: {report.clearance}"
        )

        print(
            f"Board margins: "
            f"{report.board_margin}"
        )

    print("\nQUALITY CHECKS")

    for name, passed, detail in qc:
        status = "PASS" if passed else "FAIL"

        print(
            f"  [{status}] "
            f"{name}: {detail}"
        )

    ca = crop(dets["good"].aligned)
    cb = crop(dets["defect"].aligned)

    overview = build_overview(
        dets["good"],
        dets["defect"],
        masks,
        reports,
        qc
    )

    side = build_side_by_side(ca, cb)

    cv2.imwrite(
        str(
            debug_dir
            / "B01_pair_crop_overview.png"
        ),
        overview
    )

    cv2.imwrite(
        str(
            debug_dir
            / "B01_pair_cropped_side_by_side.png"
        ),
        side
    )

    all_passed = all(
        passed
        for _, passed, _ in qc
    )

    if all_passed:
        cv2.imwrite(
            str(debug_dir / "cropped_good.png"),
            ca
        )

        cv2.imwrite(
            str(debug_dir / "cropped_defect.png"),
            cb
        )

        print(
            f"\nCROP QC PASSED. "
            f"Images saved in: {debug_dir}"
        )

    else:
        print(
            "\nCROP QC FAILED. "
            "Cropped images not exported."
        )

        print(
            f"Diagnostic figures: {debug_dir}"
        )

    if args.show:
        cv2.imshow(
            "Stage 2b overview",
            overview
        )

        cv2.imshow(
            "Stage 2b cropped",
            side
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0 if all_passed else 3


if __name__ == "__main__":
    sys.exit(main())

