"""
Phase 2A — Automatic knee landmark detection (Option D: heuristic, no learned model).

Why heuristic instead of a pretrained model:
  - BoneFinder (bone-finder.com, Univ. Manchester — same lab as Thomson 2016)
    requires a manual licensing-form request, and the knee module specifically
    is *not* included in the default research license (only proximal femur is;
    knee access needs direct email contact with the developers). No guaranteed
    turnaround time, and distribution format for headless/scripted use is
    undocumented.
  - No freely pip-installable / no-registration pretrained knee-landmark
    (keypoint regression) model was found on PyPI or GitHub. Existing public
    repos (e.g. KneeOARSIGrading) do OARSI severity *classification*, not
    landmark coordinates, and gate their weights behind a request server.
  - YOLO-pose fine-tuning (Option C) would need ~50 manually annotated images,
    which defeats the "without manual annotations" goal and adds a blocking
    annotation step.
  => Option D: classical image-processing landmarks, fully automatic, zero
     external dependencies, works immediately on the existing 256x256 images.

Landmark schema (19 points, dict of name -> (x, y) in pixel coords of a
256x256 working image):
    femur_1..femur_5   — femoral condyle margin, sampled across the joint width
    tibia_1..tibia_5   — tibial plateau margin, same x positions as femur_*
    joint_1..joint_5   — joint-space midline (midpoint of femur_i/tibia_i), same x positions
    bone_left_margin   — leftmost bone pixel at the joint row  (lateral/medial extent)
    bone_right_margin  — rightmost bone pixel at the joint row
    femur_shaft        — point on the femoral shaft axis, near top of image
    tibia_shaft        — point on the tibial shaft axis, near bottom of image
"""
from pathlib import Path

import cv2
import numpy as np

WORK_SIZE = 256
X_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)  # sample positions across joint width


def _load_gray(image_path, size=WORK_SIZE):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size))
    return gray


def _find_joint_row(gray):
    """Locate the joint line as the darkest (radiolucent) smoothed row in the
    central 40% width strip, restricted to the vertical middle third of the
    image where a tightly-cropped knee X-ray's joint line anatomically sits.

    Note: baseline/features.py uses argmax (brightest band) over the *full*
    height as a cheap JSW proxy; empirically that heuristic locks onto the
    femoral shaft near the top of the image rather than the joint (verified
    visually on several samples), so it is not reused here. This argmin +
    restricted-window version was checked visually across ~40 sample images
    (all 5 KL grades) and tracks the true joint line in the large majority of
    cases; a handful of images with a monotonic brightness trend still clip to
    the window edge — an accepted limitation of a heuristic, non-learned
    fallback (see module docstring)."""
    h, w = gray.shape
    x0, x1 = int(w * 0.30), int(w * 0.70)
    row_means = gray[:, x0:x1].mean(axis=1).astype(np.float64)
    k = 7
    smooth = np.convolve(row_means, np.ones(k) / k, mode="same")
    lo, hi = int(h * 0.35), int(h * 0.68)
    return lo + int(np.argmin(smooth[lo:hi]))


def _find_boundary(gray, x, joint_row, search_range, above: bool):
    """At column x, find the row of strongest vertical intensity gradient
    within [joint_row-search_range, joint_row) (above=True, femur margin) or
    (joint_row, joint_row+search_range] (above=False, tibia margin)."""
    h = gray.shape[0]
    col = gray[:, x].astype(np.float64)
    grad = np.abs(np.gradient(col))

    if above:
        lo, hi = max(0, joint_row - search_range), joint_row
    else:
        lo, hi = joint_row, min(h, joint_row + search_range)

    if hi <= lo:
        return joint_row
    window = grad[lo:hi]
    return lo + int(np.argmax(window))


def detect_landmarks(image_path, size=WORK_SIZE):
    """Detect ~19 approximate knee landmarks via classical image processing.

    Returns: dict[str, tuple[float, float]] — (x, y) pixel coords in a
    `size`x`size` working image (default 256x256, matching baseline/features.py).
    """
    gray = _load_gray(image_path, size)
    h, w = gray.shape
    joint_row = _find_joint_row(gray)
    search_range = max(4, int(h * 0.15))

    landmarks = {}
    for i, frac in enumerate(X_FRACTIONS, start=1):
        x = int(w * frac)
        femur_row = _find_boundary(gray, x, joint_row, search_range, above=True)
        tibia_row = _find_boundary(gray, x, joint_row, search_range, above=False)
        landmarks[f"femur_{i}"] = (float(x), float(femur_row))
        landmarks[f"tibia_{i}"] = (float(x), float(tibia_row))
        landmarks[f"joint_{i}"] = (float(x), float((femur_row + tibia_row) / 2.0))

    # Bone extent at the joint row (Otsu threshold -> bright/bone pixels)
    band = gray[max(0, joint_row - 2):joint_row + 3, :]
    _, band_bin = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cols_with_bone = np.where(band_bin.max(axis=0) > 0)[0]
    if len(cols_with_bone) > 0:
        x_min, x_max = int(cols_with_bone.min()), int(cols_with_bone.max())
    else:
        x_min, x_max = 0, w - 1
    landmarks["bone_left_margin"] = (float(x_min), float(joint_row))
    landmarks["bone_right_margin"] = (float(x_max), float(joint_row))

    # Shaft axis points (crude alignment reference)
    landmarks["femur_shaft"] = (float(w // 2), float(int(h * 0.08)))
    landmarks["tibia_shaft"] = (float(w // 2), float(int(h * 0.92)))

    return landmarks


def visualize_landmarks(image_path, landmarks, save_path=None, size=WORK_SIZE):
    """Draw landmarks over the (resized) X-ray and save a PNG. Returns save_path."""
    gray = _load_gray(image_path, size)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    colors = {
        "femur": (255, 80, 0),    # blue-ish
        "tibia": (0, 80, 255),    # red-ish
        "joint": (0, 220, 220),   # yellow
        "bone_left_margin": (0, 200, 0),
        "bone_right_margin": (0, 200, 0),
        "femur_shaft": (200, 0, 200),
        "tibia_shaft": (200, 0, 200),
    }

    def color_for(name):
        for prefix, c in colors.items():
            if name.startswith(prefix):
                return c
        return (255, 255, 255)

    for i in range(1, len(X_FRACTIONS) + 1):
        fx, fy = landmarks[f"femur_{i}"]
        tx, ty = landmarks[f"tibia_{i}"]
        cv2.line(canvas, (int(fx), int(fy)), (int(tx), int(ty)), (120, 120, 120), 1)

    for name, (x, y) in landmarks.items():
        cv2.circle(canvas, (int(x), int(y)), 3, color_for(name), -1)

    if save_path is None:
        save_path = Path(image_path).with_name(Path(image_path).stem + "_landmarks.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    return save_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python baseline/landmarks.py <image_path> [output_path]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    lm = detect_landmarks(img_path)
    for k, v in lm.items():
        print(f"  {k:20s} {v}")
    saved = visualize_landmarks(img_path, lm, save_path=out_path)
    print(f"\nSaved visualization to {saved}")
