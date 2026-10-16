"""
Phase 1 vs Phase 1+ comparison: the edge-feature ROIs (Group 4 - Οστεόφυτα)
under the blind fixed-fraction heuristic (baseline/features.py) versus the
landmark-guided ROIs (baseline/landmark_features.py), on the same image.
Same technique (Canny + Sobel + HOG) in both phases -- only the region
changes, exactly like Group 3 (Υφή Οστού).

Usage:
    python results/figures/edge_roi_comparison.py
"""
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from baseline.landmarks import detect_landmarks, X_FRACTIONS

OUT_DIR = Path(__file__).parent
DPI = 300
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
RED = "#e34948"
BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (σοβαρή αλλοίωση)", "data/dataset/test/4/9055836R.png"),
]
N_POS = len(X_FRACTIONS)


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def rois(gray, lm):
    h, w = gray.shape
    femur_y = [lm[f"femur_{i}"][1] for i in range(1, N_POS + 1)]
    tibia_y = [lm[f"tibia_{i}"][1] for i in range(1, N_POS + 1)]
    x0 = int(lm["bone_left_margin"][0])
    x1 = max(x0 + 1, int(lm["bone_right_margin"][0]))
    pad = max(2, int(h * 0.05))

    # Phase 1: blind top/bottom 20%, full width
    p1_top = (0, int(h * 0.20), 0, w)
    p1_bot = (int(h * 0.80), h, 0, w)

    # Phase 1+: landmark-anchored, bone-width only
    p1p_top = (max(0, int(min(femur_y)) - pad), min(h, int(max(femur_y)) + 2), x0, x1)
    p1p_bot = (max(0, int(min(tibia_y)) - 2), min(h, int(max(tibia_y)) + pad), x0, x1)

    return p1_top, p1_bot, p1p_top, p1p_bot


fig, axes = plt.subplots(len(EXAMPLES) * 2, 3, figsize=(12.5, 3.4 * len(EXAMPLES) * 2))

for col_row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    lm = detect_landmarks(path)
    p1_top, p1_bot, p1p_top, p1p_bot = rois(gray, lm)

    for sub, (region_name, p1, p1p) in enumerate([
        ("μηριαίο (πάνω)", p1_top, p1p_top),
        ("κνήμη (κάτω)", p1_bot, p1p_bot),
    ]):
        r = col_row * 2 + sub
        ax0, ax1, ax2 = axes[r]
        y0, y1, x0, x1 = p1
        py0, py1, px0, px1 = p1p

        ax0.imshow(gray, cmap="gray")
        ax0.add_patch(plt.Rectangle((x0, y0), x1 - x0 - 1, y1 - y0 - 1,
                                    fill=False, edgecolor=RED, lw=2, label="Φάση 1 (τυφλό 20%)"))
        ax0.add_patch(plt.Rectangle((px0, py0), px1 - px0, py1 - py0,
                                    fill=False, edgecolor=BLUE, lw=2, label="Φάση 1+ (landmark-guided)"))
        ax0.set_title(f"{label} -- περιοχή {region_name}", fontsize=9.5)
        ax0.axis("off")
        if r == 0:
            handles, labels_ = ax0.get_legend_handles_labels()
            fig.legend(handles, labels_, loc="lower center", bbox_to_anchor=(0.5, -0.01),
                      fontsize=9, ncol=2, frameon=True)

        crop1 = gray[y0:y1, x0:x1]
        ax1.imshow(crop1, cmap="gray", aspect="auto")
        ax1.set_title(f"Φάση 1 crop ({crop1.shape[1]}×{crop1.shape[0]} px)", fontsize=9, color=RED)
        ax1.axis("off")

        crop2 = gray[py0:py1, px0:px1]
        ax2.imshow(crop2, cmap="gray", aspect="auto")
        ax2.set_title(f"Φάση 1+ crop ({crop2.shape[1]}×{crop2.shape[0]} px)", fontsize=9, color=BLUE)
        ax2.axis("off")

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Σύγκριση ROI Ομάδας 4 (Οστεόφυτα): Φάση 1 vs Φάση 1+\n"
             "(ίδια τεχνική: Canny + Sobel + HOG -- μόνο η περιοχή αλλάζει)",
            fontsize=13, y=1.015)
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"edge_roi_comparison.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
