import cv2
import numpy as np
from board_profile import BoardProfile
from stage2_align_pair import (
    AlignmentError,
    find_perforations,
    find_mount_holes,
)


def align_with_profile(img, profile: BoardProfile):
    if img is None or img.size == 0:
        raise AlignmentError("Invalid input image")

    dots = find_perforations(img)
    holes, radii = find_mount_holes(img, dots)

    holes = np.asarray(holes, dtype=np.float32)
    template = profile.template()

    if holes.shape != (4, 2):
        raise AlignmentError("Expected four mounting holes")

    H = cv2.getPerspectiveTransform(holes, template)

    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        raise AlignmentError("Invalid homography")

    # Temporary canvas for alignment testing.
    # Final crop dimensions will be validated separately.
    padding = max(profile.margin, 220)
    canvas_w = int(profile.width + profile.margin + padding)
    canvas_h = int(profile.height + profile.margin + padding)

    aligned = cv2.warpPerspective(
        img, H, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    valid = cv2.warpPerspective(
        np.full(img.shape[:2], 255, dtype=np.uint8),
        H, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )

    return {
        "aligned": aligned,
        "valid": valid,
        "homography": H,
        "holes": holes,
        "radii": radii,
        "dots": dots,
        "template": template,
    }
