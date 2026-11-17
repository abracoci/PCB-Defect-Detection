"""
pcb_core.py - per-image measurement, GOOD-ONLY calibration, and QC decisions.

Reuses Stage 1 (Sample), Stage 2a (process_one) and Stage 2b (crop + object checks).
No GUI, no printing: this module is meant to be imported later by a Raspberry Pi script.

Calibration rules enforced here
-------------------------------
* calibrate() refuses any image whose condition != "good".
* QC thresholds are derived from the good-image distribution, then FROZEN. They are never
  adjusted using defective images.
* If the good images are not mutually consistent (a metric exceeds its hard ceiling) the
  calibration raises CalibrationError - the algorithm is fixed, not the threshold loosened.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import stage2b_crop as sc
from stage1_dataset import Sample
from stage2_align_pair import AlignmentError, TPL_H, TPL_W, TPL_X, TPL_Y, process_one

THUMB_W = 360
MIN_DOTS = 100
GRID_PITCH_RANGE = (38.0, 44.0)         # perforation pitch search range in the reference frame (px)
INLIER_PX = 8.0

# (floor, ceiling) for each calibrated threshold. Threshold = clamp(1.5 * worst good value, floor, ceiling).
# If the worst GOOD value exceeds the ceiling, calibration fails instead of clamping.
THRESHOLD_BOUNDS = {
    "grid_offset_px": (2.0, 8.0),
    "grid_spread_px": (2.0, 6.0),
    "grid_drift_px": (2.0, 8.0),
    "edge_dev_px": (3.0, 10.0),
}
CROP_MIN_CLEARANCE = sc.QC_OBJECT_CLEARANCE_PX      # fixed by design, not calibrated


class CalibrationError(Exception):
    pass


@dataclass
class Measure:
    sample: Sample
    status: str = "PENDING"             # OK | FAIL_ALIGNMENT | FAIL_QC
    reason: str = ""
    m: dict = field(default_factory=dict)
    dots_ref: np.ndarray | None = None  # perforation centres in the reference frame
    thumb: np.ndarray | None = None


def _thumb(img: np.ndarray) -> np.ndarray:
    h = int(img.shape[0] * THUMB_W / img.shape[1])
    return cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA)


def _fail_thumb(sample: Sample) -> np.ndarray:
    img = cv2.imread(str(sample.raw_path), cv2.IMREAD_COLOR)
    return _thumb(img) if img is not None else np.zeros((203, THUMB_W, 3), np.uint8)


# ----------------------------------------------------------------------------
# 1. Per-image measurement (no thresholds applied here)
# ----------------------------------------------------------------------------
def measure(sample: Sample) -> Measure:
    r = Measure(sample)
    try:
        d = process_one(sample)
    except AlignmentError as e:
        r.status, r.reason, r.thumb = "FAIL_ALIGNMENT", str(e), _fail_thumb(sample)
        return r
    except Exception as e:      # one bad file must neither stop the batch nor pass silently
        r.status, r.reason = "FAIL_ALIGNMENT", f"unexpected {type(e).__name__}: {e}"
        r.thumb = _fail_thumb(sample)
        return r

    g = d.geometry
    r.m.update({
        "hole_top": g["top"], "hole_bottom": g["bottom"], "hole_left": g["left"], "hole_right": g["right"],
        "keystone_w_pct": g["keystone_w_pct"], "keystone_h_pct": g["keystone_h_pct"],
        "rot_top_deg": g["rot_top_deg"], "rot_bottom_deg": g["rot_bottom_deg"],
        "affine_rms_px": g["affine_rms_px"],
        **{f"edge_{k}": v for k, v in d.edges.items()},
    })
    p = cv2.perspectiveTransform(d.dots[None].astype(np.float32), d.H)[0]
    mg = 40
    keep = ((p[:, 0] > TPL_X + mg) & (p[:, 0] < TPL_X + TPL_W - mg) &
            (p[:, 1] > TPL_Y + mg) & (p[:, 1] < TPL_Y + TPL_H - mg))
    r.dots_ref = p[keep]
    r.m["n_dots"] = int(keep.sum())

    mask = sc.find_outside_objects(d)
    rep = sc.verify_crop(d, mask)
    # Medir la cobertura real de la PCB
    pcb_mask = np.zeros(d.valid.shape[:2], dtype=np.uint8)

    quad = np.round(d.quad_ref).astype(np.int32)
    cv2.fillConvexPoly(pcb_mask, quad, 255)

    x0, y0 = sc.CROP_X0, sc.CROP_Y0
    x1, y1 = x0 + sc.CROP_W, y0 + sc.CROP_H

    pcb_crop = pcb_mask[y0:y1, x0:x1] > 0
    valid_crop = d.valid[y0:y1, x0:x1] > 0

    pcb_pixels = int(np.count_nonzero(pcb_crop))

    pcb_valid_fraction = (
        float(np.count_nonzero(pcb_crop & valid_crop) / pcb_pixels)
        if pcb_pixels > 0 else 0.0
    )

    r.m["pcb_valid_fraction"] = pcb_valid_fraction
    clear = [v for v in rep.clearance.values() if v is not None]
    r.m.update({"crop_valid_fraction": rep.valid_fraction, "crop_touching": len(rep.touching),
                "crop_min_clearance": min(clear) if clear else None, "outside_objects": len(rep.objects)})
    r.thumb = _thumb(sc.crop(d.aligned))
    r.status = "MEASURED"
    return r


# ----------------------------------------------------------------------------
# 2. Perforation-grid phase check (independent of the 4 holes used for the warp)
# ----------------------------------------------------------------------------
@dataclass
class Grid:
    px: float
    py: float
    x0: float
    y0: float
    strength_x: float
    strength_y: float


def _fit_axis(c: np.ndarray):
    coarse = np.arange(*GRID_PITCH_RANGE, 0.05)
    R = [abs(np.exp(2j * np.pi * c / p).mean()) for p in coarse]
    p0 = coarse[int(np.argmax(R))]
    fine = np.arange(p0 - 0.05, p0 + 0.05, 0.005)
    R2 = [abs(np.exp(2j * np.pi * c / p).mean()) for p in fine]
    p = float(fine[int(np.argmax(R2))])
    z = np.exp(2j * np.pi * c / p).mean()
    return p, float((np.angle(z) * p / (2 * np.pi)) % p), float(abs(z))


def fit_grid(dots: np.ndarray) -> Grid:
    px, x0, sx = _fit_axis(dots[:, 0])
    py, y0, sy = _fit_axis(dots[:, 1])
    if min(sx, sy) < 0.3:
        raise CalibrationError(f"perforation lattice is not clearly periodic (strength {sx:.2f}/{sy:.2f})")
    return Grid(px, py, x0, y0, sx, sy)


def _robust(res: np.ndarray):
    med = float(np.median(res))
    inl = np.abs(res - med) < INLIER_PX
    med = float(np.median(res[inl]))
    spread = float(1.4826 * np.median(np.abs(res[inl] - med)))
    return med, spread, inl


def grid_offsets(dots: np.ndarray, grid: Grid) -> dict:
    """Where does this image's perforation lattice sit relative to the common grid?"""
    rx = (dots[:, 0] - grid.x0 + grid.px / 2) % grid.px - grid.px / 2
    ry = (dots[:, 1] - grid.y0 + grid.py / 2) % grid.py - grid.py / 2
    ox, sx, ix = _robust(rx)
    oy, sy, iy = _robust(ry)
    cx, cy = TPL_X + TPL_W / 2, TPL_Y + TPL_H / 2

    def half_diff(res, inl, coord, c):
        a, b = res[inl & (coord > c)], res[inl & (coord <= c)]
        return float(np.median(a) - np.median(b)) if len(a) > 5 and len(b) > 5 else float("nan")
    return {"grid_off_x": ox, "grid_off_y": oy, "grid_spread_x": sx, "grid_spread_y": sy,
            "grid_drift_x": half_diff(rx, ix, dots[:, 0], cx), "grid_drift_y": half_diff(ry, iy, dots[:, 1], cy)}


# ----------------------------------------------------------------------------
# 3. Good-only calibration
# ----------------------------------------------------------------------------
@dataclass
class Calibration:
    grid: Grid
    edges: dict
    thresholds: dict
    template_check: dict
    used_files: list


def _clamp(worst_good: float, key: str) -> float:
    lo, hi = THRESHOLD_BOUNDS[key]
    if worst_good > hi:
        raise CalibrationError(f"good images are not mutually consistent: worst {key} = {worst_good:.2f} "
                               f"exceeds the ceiling {hi}. Fix the alignment method; do not loosen the limit.")
    return float(min(max(1.5 * worst_good, lo), hi))


def calibrate(good: list[Measure]) -> Calibration:
    if any(g.sample.condition != "good" for g in good):
        raise ValueError("calibration may only use condition == 'good' images")
    ok = [g for g in good if g.status == "MEASURED" and g.m.get("n_dots", 0) >= MIN_DOTS]
    if len(ok) < 4:
        raise CalibrationError(f"only {len(ok)} usable good images; need at least 4")
    grid = fit_grid(np.vstack([g.dots_ref for g in ok]))
    for g in ok:
        g.m.update(grid_offsets(g.dots_ref, grid))
    edges = {k: float(np.median([g.m[f"edge_{k}"] for g in ok])) for k in ("left", "right", "top", "bottom")}
    for g in ok:
        for k in edges:
            g.m[f"edge_dev_{k}"] = abs(g.m[f"edge_{k}"] - edges[k])

    def worst(keys):
        vals = [abs(g.m[k]) for g in ok for k in keys if not np.isnan(g.m[k])]
        return max(vals) if vals else 0.0
    thr = {
        "grid_offset_px": _clamp(worst(["grid_off_x", "grid_off_y"]), "grid_offset_px"),
        "grid_spread_px": _clamp(worst(["grid_spread_x", "grid_spread_y"]), "grid_spread_px"),
        "grid_drift_px": _clamp(worst(["grid_drift_x", "grid_drift_y"]), "grid_drift_px"),
        "edge_dev_px": _clamp(worst([f"edge_dev_{k}" for k in edges]), "edge_dev_px"),
    }
    w = float(np.median([(g.m["hole_top"] + g.m["hole_bottom"]) / 2 for g in ok]))
    h = float(np.median([(g.m["hole_left"] + g.m["hole_right"]) / 2 for g in ok]))
    tc = {"good_median_hole_width": w, "good_median_hole_height": h,
          "template_width": TPL_W, "template_height": TPL_H,
          "width_dev_pct": 100 * (TPL_W / w - 1), "height_dev_pct": 100 * (TPL_H / h - 1)}
    return Calibration(grid, edges, thr, tc, [g.sample.filename for g in ok])


# ----------------------------------------------------------------------------
# 4. Evaluation with FROZEN thresholds (works for good and defective alike)
# ----------------------------------------------------------------------------
def evaluate(r: Measure, cal: Calibration) -> Measure:
    if r.status == "FAIL_ALIGNMENT":
        return r
    r.m.update(grid_offsets(r.dots_ref, cal.grid))
    for k, v in cal.edges.items():
        r.m[f"edge_dev_{k}"] = abs(r.m[f"edge_{k}"] - v)
    t, m = cal.thresholds, r.m
    nan = float("nan")
    checks = [
        ("enough perforations", m["n_dots"] >= MIN_DOTS, f"{m['n_dots']}"),
        ("grid offset", max(abs(m["grid_off_x"]), abs(m["grid_off_y"])) <= t["grid_offset_px"],
         f"{m['grid_off_x']:+.1f},{m['grid_off_y']:+.1f} (limit {t['grid_offset_px']:.1f})"),
        ("grid spread", max(m["grid_spread_x"], m["grid_spread_y"]) <= t["grid_spread_px"],
         f"{m['grid_spread_x']:.1f},{m['grid_spread_y']:.1f} (limit {t['grid_spread_px']:.1f})"),
        ("grid drift", np.nanmax([abs(m["grid_drift_x"]), abs(m["grid_drift_y"])]) <= t["grid_drift_px"],
         f"{m['grid_drift_x']:+.1f},{m['grid_drift_y']:+.1f} (limit {t['grid_drift_px']:.1f})"),
        ("board edges", max(m[f"edge_dev_{k}"] for k in cal.edges) <= t["edge_dev_px"],
         f"max dev {max(m[f'edge_dev_{k}'] for k in cal.edges):.0f} (limit {t['edge_dev_px']:.0f})"),
        # Cobertura total registrada como advertencia; excepcion visual para fondo invalidado.
        ("no object touches crop edge", m["crop_touching"] == 0, f"{m['crop_touching']} touching"),
        ("object clearance", m["crop_min_clearance"] is None or m["crop_min_clearance"] >= CROP_MIN_CLEARANCE,
         f"min {m['crop_min_clearance']} (limit {CROP_MIN_CLEARANCE})"),
    ]
        # Registrar los pixeles invalidos del recorte como diagnostico.
    # No cambia todavia los criterios de aprobacion.
    m["crop_background_warning"] = (
        m["crop_valid_fraction"] < 1.0
    )
    failed = [f"{n}: {i}" for n, ok, i in checks if not ok]
    r.m["qc_failed"] = "; ".join(failed)
    r.status = "FAIL_QC" if failed else "OK"
    r.reason = r.m["qc_failed"]
    return r


# ----------------------------------------------------------------------------
# 5. Does one shared reference frame suit both boards? (GOOD images only)
# ----------------------------------------------------------------------------
def board_comparison(good: list[Measure]) -> list[tuple]:
    """Compare every board with the first board using GOOD images only.

    Returns (reference_board, other_board, metric, ref_value, other_value,
             difference, tolerance, passed).
    """
    by = {b: [g for g in good if g.sample.board == b
              and g.status in ("OK", "FAIL_QC") and "grid_off_x" in g.m]
          for b in sorted({g.sample.board for g in good})}
    by = {b: gs for b, gs in by.items() if len(gs) >= 2}
    if len(by) < 2:
        return []
    ref_name = next(iter(by))
    ref = by[ref_name]
    def med(gs, f):
        return float(np.median([f(g.m) for g in gs]))
    metrics = [
        ("hole spacing, width (%)", lambda m: (m["hole_top"] + m["hole_bottom"]) / 2, 1.0, True),
        ("hole spacing, height (%)", lambda m: (m["hole_left"] + m["hole_right"]) / 2, 1.0, True),
        ("keystone width (pct-pts)", lambda m: m["keystone_w_pct"], 1.0, False),
        ("rotation (deg)", lambda m: (m["rot_top_deg"] + m["rot_bottom_deg"]) / 2, 1.0, False),
        ("grid offset x (px)", lambda m: m["grid_off_x"], 3.0, False),
        ("grid offset y (px)", lambda m: m["grid_off_y"], 3.0, False),
    ]
    metrics.extend((f"board edge {k} (px)", lambda m, k=k: m[f"edge_{k}"], 4.0, False)
                   for k in ("left", "right", "top", "bottom"))
    rows = []
    for board, gs in by.items():
        if board == ref_name:
            continue
        for name, f, limit, pct in metrics:
            a, b = med(ref, f), med(gs, f)
            diff = 100 * (b / a - 1) if pct and abs(a) > 1e-9 else (b - a if not pct else float("inf"))
            rows.append((ref_name, board, name, a, b, diff, limit, abs(diff) <= limit))
    return rows

