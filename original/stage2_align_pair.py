"""
Stage 2a - Board detection and alignment of ONE good/defective B01 pair.

What this script does (and nothing more):
  1. Loads two raw images read-only (via Stage 1's load_samples / Sample).
  2. Finds the perforation lattice and the 4 large mounting holes (alignment anchors).
  3. Measures whether perspective correction is needed (keystone / affine residual).
  4. Warps both images into one reference frame with a 4-point homography.
  5. Finds the physical board edges + 4 board corners in that frame.
  6. Runs independent quality checks and saves a side-by-side figure.

Nothing is cropped here: the output canvas is the same size as the camera frame.
Colour is untouched: warping only resamples pixels (bilinear); no colour/contrast ops.
Components (capacitor, LEDs, ICs) are never used as landmarks.

Run from the Scripts folder:
    py stage2_align_pair.py --root .. --out ..\\pcb_processed
Optional: add --show to open the figure in a window.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stage1_dataset import Sample, load_samples  # reuse Stage 1

# ----------------------------------------------------------------------------
# Constants (PROVISIONAL: derived from this one B01 pair; Stage 3 recomputes
# them from all good images).
# ----------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 2048, 1152            # same as the camera frame - no crop yet
TPL_W, TPL_H = 1184, 677                   # mounting-hole rectangle in the reference frame
TPL_X, TPL_Y = 430, 250                    # its top-left corner in the reference frame
TEMPLATE = np.float32([[TPL_X, TPL_Y], [TPL_X + TPL_W, TPL_Y],
                       [TPL_X + TPL_W, TPL_Y + TPL_H], [TPL_X, TPL_Y + TPL_H]])
CORNER_NAMES = ("TL", "TR", "BR", "BL")

HOLE_RADIUS_PX = (20, 36)                  # mounting holes ~25-30 px radius
MAX_HOLE_PRIOR_DIST = 140                  # hole must lie near its lattice corner
EDGE_SEARCH_PX = (25, 105)                 # board edge lies this far OUTSIDE the hole rectangle
MIN_EDGE_STRENGTH = 1.5                    # gradient strength needed to trust an edge
AFFINE_RMS_PERSPECTIVE_PX = 2.0            # affine residual above this => perspective needed

# QC thresholds (PROVISIONAL - to be calibrated on all good images)
LATTICE_INLIER_PX = 8.0                    # a dot with a partner closer than this is an inlier
QC_LATTICE_INLIER_FRAC = 0.70
QC_LATTICE_MEDIAN_PX = 3.0                 # ~7% of the 41px hole pitch
QC_LATTICE_P90_PX = 6.0
QC_EDGE_AGREE_PX = 6.0

DEFAULT_GOOD = "B01_good_S01_20261009_184325_920171.jpg"
DEFAULT_DEFECT = "B01_misaligned_capacitor_S03_20261009_201325_371437.jpg"


class AlignmentError(Exception):
    """Raised when a stage cannot be trusted. Never swallowed silently."""


@dataclass
class Detection:
    sample: Sample
    img: np.ndarray
    dots: np.ndarray            # perforation centres (original coords)
    holes: np.ndarray           # (4,2) TL,TR,BR,BL mounting-hole centres (original coords)
    hole_radii: list
    geometry: dict
    H: np.ndarray               # original -> reference frame
    aligned: np.ndarray
    valid: np.ndarray           # 255 where aligned pixel came from the real image
    edges: dict                 # board edges in reference frame
    edge_strength: dict
    quad_ref: np.ndarray        # board corners in reference frame
    quad_orig: np.ndarray       # same corners mapped back to the original image


# ----------------------------------------------------------------------------
# Step A - perforation lattice (board-only texture; components do not produce it)
# ----------------------------------------------------------------------------
def find_perforations(img: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    _, mask = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    pts = [cents[i] for i in range(1, n)
           if 20 < stats[i, cv2.CC_STAT_AREA] < 400
           and stats[i, cv2.CC_STAT_WIDTH] < 30 and stats[i, cv2.CC_STAT_HEIGHT] < 30]
    if len(pts) < 150:
        raise AlignmentError(f"only {len(pts)} perforations found - board not visible/in focus?")
    pts = np.array(pts, np.float32)
    # keep only dots that have neighbours (removes isolated specks)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    pts = pts[(dist < 70).sum(1) >= 5]
    if len(pts) < 150:
        raise AlignmentError("perforation lattice too sparse after filtering")
    return pts




def find_mount_holes(img: np.ndarray, dots: np.ndarray):
    gray = cv2.medianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=120,
        param2=20,
        minRadius=HOLE_RADIUS_PX[0],
        maxRadius=HOLE_RADIUS_PX[1]
    )

    if circles is None:
        raise AlignmentError(
            "no circular mounting-hole candidates found"
        )

    c = circles[0]

    x0, x1 = np.percentile(dots[:, 0], [1, 99])
    y0, y1 = np.percentile(dots[:, 1], [1, 99])

    priors = [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1)
    ]

    holes = []
    radii = []
    used = set()

    for index, (name, (px, py)) in enumerate(
        zip(CORNER_NAMES, priors)
    ):
        distances = np.hypot(
            c[:, 0] - px,
            c[:, 1] - py
        )

        available = [
            i for i in range(len(c))
            if i not in used
        ]

        if not available:
            raise AlignmentError(
                f"no available candidate for {name}"
            )

        candidate = min(
            available,
            key=lambda i: distances[i]
        )

        # Keep the original limit for TL, TR and BL.
        # BR gets a larger search region.
        limit = 200 if index == 2 else MAX_HOLE_PRIOR_DIST

        if distances[candidate] > limit:
            raise AlignmentError(
                f"mounting hole {name} not found "
                f"(distance={distances[candidate]:.1f}px)"
            )

        used.add(candidate)
        holes.append(c[candidate, :2])
        radii.append(float(c[candidate, 2]))

    return np.array(holes, np.float32), radii


def analyse_geometry(holes: np.ndarray) -> dict:
    """Plausibility checks + 'is perspective correction needed?' measurements."""
    tl, tr, br, bl = holes
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)

    def ang(a, b):
        return float(np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
    corner_angles = [ang(tr - tl, bl - tl), ang(tl - tr, br - tr),
                     ang(tr - br, bl - br), ang(tl - bl, br - bl)]
    if not (1000 < top < 1400 and 1000 < bottom < 1400 and 550 < left < 800 and 550 < right < 800):
        raise AlignmentError(f"hole quad has implausible size (top {top:.0f}, bottom {bottom:.0f}, "
                             f"left {left:.0f}, right {right:.0f})")
    if any(abs(a - 90) > 10 for a in corner_angles):
        raise AlignmentError(f"hole quad corner angles implausible: {np.round(corner_angles, 1)}")

    # best affine map of the 4 holes onto the template rectangle: its residual measures perspective
    A, b = [], []
    for (x, y), (u, v) in zip(holes, TEMPLATE):
        A.append([x, y, 1, 0, 0, 0]); b.append(u)
        A.append([0, 0, 0, x, y, 1]); b.append(v)
    sol, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
    resid = (np.array(A) @ sol - np.array(b)).reshape(4, 2)
    affine_rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    return {
        "top": top, "bottom": bottom, "left": left, "right": right,
        "keystone_w_pct": 100 * (bottom / top - 1), "keystone_h_pct": 100 * (right / left - 1),
        "rot_top_deg": float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))),
        "rot_bottom_deg": float(np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))),
        "corner_angles": corner_angles, "affine_rms_px": affine_rms,
        "needs_perspective": affine_rms > AFFINE_RMS_PERSPECTIVE_PX,
    }


# ----------------------------------------------------------------------------
# Step C - warp into the reference frame
# ----------------------------------------------------------------------------
def warp_to_reference(img: np.ndarray, holes: np.ndarray):
    H = cv2.getPerspectiveTransform(holes, TEMPLATE)
    size = (CANVAS_W, CANVAS_H)
    aligned = cv2.warpPerspective(img, H, size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), H, size,
                                flags=cv2.INTER_NEAREST, borderValue=0)
    return H, aligned, valid


# ----------------------------------------------------------------------------
# Step D - physical board edges, searched OUTSIDE the hole rectangle in the
#          reference frame (axis-aligned there, so a 1-D profile is enough)
# ----------------------------------------------------------------------------
def detect_board_edges(aligned: np.ndarray, valid: np.ndarray):
    g = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
    lo, hi = EDGE_SEARCH_PX
    xs = slice(TPL_X + int(.2 * TPL_W), TPL_X + int(.8 * TPL_W))   # central 60% of top/bottom
    ys = slice(TPL_Y + int(.2 * TPL_H), TPL_Y + int(.8 * TPL_H))   # central 60% of left/right
    base = {"top": TPL_Y, "bottom": TPL_Y + TPL_H, "left": TPL_X, "right": TPL_X + TPL_W}
    edges, strength = {}, {}
    for side in ("top", "bottom", "left", "right"):
        sign = -1 if side in ("top", "left") else 1
        pos = [base[side] + sign * o for o in range(lo, hi + 1)]
        pos = [p for p in pos if 0 <= p < (CANVAS_H if side in ("top", "bottom") else CANVAS_W)]
        if side in ("top", "bottom"):
            prof = np.array([np.median(g[p, xs]) for p in pos]); vf = [valid[p, xs].mean() / 255 for p in pos]
        else:
            prof = np.array([np.median(g[ys, p]) for p in pos]); vf = [valid[ys, p].mean() / 255 for p in pos]
        order = np.argsort(pos)                       # increasing image coordinate
        pos, prof, vf = np.array(pos)[order], prof[order], np.array(vf)[order]
        d = np.gradient(cv2.GaussianBlur(prof[None], (0, 0), 2)[0])
        ok = vf > 0.999
        ok = ok & np.roll(ok, 3) & np.roll(ok, -3)    # stay away from the invalid-pixel border
        if ok.sum() < 10:
            raise AlignmentError(f"board {side} edge: search window lies outside the image")
        d = np.where(ok, d, 0)
        i = int(np.argmax(d) if side in ("top", "left") else np.argmin(d))
        if abs(d[i]) < MIN_EDGE_STRENGTH:
            raise AlignmentError(f"board {side} edge too weak (gradient {abs(d[i]):.2f} < {MIN_EDGE_STRENGTH})")
        if i < 4 or i > len(pos) - 5:
            raise AlignmentError(f"board {side} edge hit the search-window limit - widen EDGE_SEARCH_PX")
        edges[side], strength[side] = int(pos[i]), float(abs(d[i]))
    quad = np.float32([[edges["left"], edges["top"]], [edges["right"], edges["top"]],
                       [edges["right"], edges["bottom"]], [edges["left"], edges["bottom"]]])
    return edges, strength, quad


def process_one(sample: Sample) -> Detection:
    img = cv2.imread(str(sample.raw_path), cv2.IMREAD_COLOR)     # read-only use of the raw file
    if img is None:
        raise AlignmentError(f"cannot read {sample.raw_path}")
    dots = find_perforations(img)
    holes, radii = find_mount_holes(img, dots)
    geom = analyse_geometry(holes)
    H, aligned, valid = warp_to_reference(img, holes)
    edges, strength, quad_ref = detect_board_edges(aligned, valid)
    quad_orig = cv2.perspectiveTransform(quad_ref[None], np.linalg.inv(H))[0]
    return Detection(sample, img, dots, holes, radii, geom, H, aligned, valid,
                     edges, strength, quad_ref, quad_orig)


# ----------------------------------------------------------------------------
# Step E - quality checks that do NOT depend on the 4 holes used for the warp
# ----------------------------------------------------------------------------
def lattice_agreement(a: Detection, b: Detection) -> dict:
    """Warp both perforation lattices into the reference frame and compare them.

    Dots hidden or disturbed by components (which sit in different places in the two
    images) have no partner, so only dots matched within LATTICE_INLIER_PX count as
    inliers; the inlier share is reported separately as its own check.
    """
    def to_ref(d):
        p = cv2.perspectiveTransform(d.dots[None].astype(np.float32), d.H)[0]
        m = 40
        keep = ((p[:, 0] > TPL_X + m) & (p[:, 0] < TPL_X + TPL_W - m) &
                (p[:, 1] > TPL_Y + m) & (p[:, 1] < TPL_Y + TPL_H - m))
        return p[keep]
    pa, pb = to_ref(a), to_ref(b)
    dist = np.linalg.norm(pb[:, None] - pa[None], axis=2)
    nn, j = dist.min(1), dist.argmin(1)
    inl = nn < LATTICE_INLIER_PX
    vec = (pa[j] - pb)[inl]
    return {"n": int(len(pb)), "inlier_frac": float(inl.mean()),
            "median": float(np.median(nn[inl])), "p90": float(np.percentile(nn[inl], 90)),
            "shift_xy": np.median(vec, axis=0)}


def run_qc(a: Detection, b: Detection) -> list[tuple[str, bool, str]]:
    lat = lattice_agreement(a, b)
    edge_diff = {s: abs(a.edges[s] - b.edges[s]) for s in a.edges}
    return [
        ("4/4 mounting holes in both images", True, "found"),
        ("lattice dots visible in both", lat["inlier_frac"] >= QC_LATTICE_INLIER_FRAC,
         f"{100*lat['inlier_frac']:.0f}% of {lat['n']} (min {100*QC_LATTICE_INLIER_FRAC:.0f}%)"),
        ("lattice median offset", lat["median"] <= QC_LATTICE_MEDIAN_PX,
         f"{lat['median']:.2f}px (limit {QC_LATTICE_MEDIAN_PX}); shift dx,dy = "
         f"{lat['shift_xy'][0]:+.1f},{lat['shift_xy'][1]:+.1f}"),
        ("lattice 90th-percentile offset", lat["p90"] <= QC_LATTICE_P90_PX,
         f"{lat['p90']:.2f}px (limit {QC_LATTICE_P90_PX})"),
        ("board edges agree between images", max(edge_diff.values()) <= QC_EDGE_AGREE_PX,
         "max diff %.0fpx %s" % (max(edge_diff.values()), edge_diff)),
    ]


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def _label(img, text, w=680):
    h = int(img.shape[0] * w / img.shape[1])
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(small, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(small, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return small


def overlay_original(d: Detection) -> np.ndarray:
    out = d.img.copy()
    cv2.polylines(out, [d.quad_orig.astype(np.int32)], True, (0, 255, 0), 4)       # board outline
    cv2.polylines(out, [d.holes.astype(np.int32)], True, (255, 200, 0), 2)          # hole quad
    for name, (x, y), r in zip(CORNER_NAMES, d.holes, d.hole_radii):
        cv2.circle(out, (int(x), int(y)), int(r), (255, 0, 0), 4)
        cv2.putText(out, name, (int(x) + 30, int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    for p in d.quad_orig:
        cv2.circle(out, (int(p[0]), int(p[1])), 12, (0, 0, 255), -1)
    return out


def overlay_aligned(d: Detection) -> np.ndarray:
    out = d.aligned.copy()
    cv2.polylines(out, [d.quad_ref.astype(np.int32)], True, (0, 255, 0), 3)
    for p in TEMPLATE:
        cv2.circle(out, (int(p[0]), int(p[1])), 28, (255, 0, 0), 3)
    return out


def metrics_panel(a: Detection, b: Detection, qc) -> np.ndarray:
    lines = []
    for tag, d in (("GOOD", a), ("DEFECT", b)):
        g = d.geometry
        lines += [f"{tag}: {d.sample.filename[:34]}",
                  f"  hole quad top/bottom {g['top']:.0f}/{g['bottom']:.0f}px  left/right {g['left']:.0f}/{g['right']:.0f}px",
                  f"  keystone w {g['keystone_w_pct']:+.1f}%  h {g['keystone_h_pct']:+.1f}%  rot {g['rot_top_deg']:.1f}/{g['rot_bottom_deg']:.1f} deg",
                  f"  affine-only residual {g['affine_rms_px']:.1f}px -> perspective needed: {g['needs_perspective']}",
                  f"  board edges T{d.edges['top']} B{d.edges['bottom']} L{d.edges['left']} R{d.edges['right']}", ""]
    lines.append("QUALITY CHECKS")
    for name, ok, info in qc:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}"[:92])
    panel = np.zeros((382, 680, 3), np.uint8)
    for i, t in enumerate(lines):
        col = (0, 255, 0) if "[PASS]" in t else (0, 0, 255) if "[FAIL]" in t else (230, 230, 230)
        cv2.putText(panel, t, (8, 22 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    return panel


def build_figure(a: Detection, b: Detection, qc) -> np.ndarray:
    blend = cv2.addWeighted(a.aligned, 0.5, b.aligned, 0.5, 0)
    row1 = np.hstack([_label(overlay_original(a), "1. ORIGINAL good (detections)"),
                      _label(overlay_original(b), "2. ORIGINAL defective (detections)"),
                      metrics_panel(a, b, qc)])
    row2 = np.hstack([_label(overlay_aligned(a), "3. ALIGNED good"),
                      _label(overlay_aligned(b), "4. ALIGNED defective"),
                      _label(blend, "5. 50/50 blend (diagnostic only)")])
    return np.vstack([row1, row2])


# ----------------------------------------------------------------------------
def pick(samples, name: str) -> Sample:
    for s in samples:
        if s.filename == name:
            return s
    raise SystemExit(f"ERROR: '{name}' is not listed in metadata.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="NEW output root (never inside raw/)")
    ap.add_argument("--good", default=DEFAULT_GOOD)
    ap.add_argument("--defect", default=DEFAULT_DEFECT)
    ap.add_argument("--show", action="store_true", help="open the figure in a window")
    args = ap.parse_args()

    samples, problems = load_samples(args.root, args.out)
    if problems:
        print(f"WARNING: Stage 1 validation reported {len(problems)} problem(s), e.g. {problems[0]}")
    good, defect = pick(samples, args.good), pick(samples, args.defect)

    debug_dir = args.out.resolve() / "_stage2_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    dets, failed = {}, False
    for tag, s in (("good", good), ("defect", defect)):
        try:
            dets[tag] = process_one(s)
            print(f"[ok]   {tag:6s} {s.filename}: 4 holes, edges {dets[tag].edges}")
        except AlignmentError as e:
            failed = True
            print(f"[FAIL] {tag:6s} {s.filename}: {e}")
    if failed:
        print("\nALIGNMENT FAILED - no aligned images were written. Fix the cause above first.")
        return 2

    a, b = dets["good"], dets["defect"]
    for tag, d in dets.items():
        g = d.geometry
        print(f"\n{tag}: keystone width {g['keystone_w_pct']:+.1f}% height {g['keystone_h_pct']:+.1f}%, "
              f"rotation {g['rot_top_deg']:.2f}/{g['rot_bottom_deg']:.2f} deg, "
              f"affine-only residual {g['affine_rms_px']:.1f}px -> "
              f"perspective correction {'NEEDED' if g['needs_perspective'] else 'not needed'}")
        print(f"  valid-pixel share of canvas: {100*(d.valid>0).mean():.1f}%")

    qc = run_qc(a, b)
    print("\nQUALITY CHECKS")
    for name, ok, info in qc:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")

    fig = build_figure(a, b, qc)
    cv2.imwrite(str(debug_dir / "B01_pair_alignment.png"), fig)
    cv2.imwrite(str(debug_dir / "aligned_good.png"), a.aligned)
    cv2.imwrite(str(debug_dir / "aligned_defect.png"), b.aligned)
    print(f"\nSaved figure + aligned test images in: {debug_dir}")

    if args.show:
        cv2.imshow("Stage 2a - press any key to close", fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if all(ok for _, ok, _ in qc) else 3


if __name__ == "__main__":
    sys.exit(main())