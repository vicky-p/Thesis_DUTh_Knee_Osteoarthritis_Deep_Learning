"""
Phase 1 vs Phase 1+ comparison: the texture-feature ROI (Group 3) under the
blind fixed-fraction heuristic (baseline/features.py) versus the
landmark-guided ROI (baseline/landmark_features.py), on the same image.

Usage:
    python results/figures/texture_roi_comparison.py
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
    # Phase 1: blind bottom 30%, full width
    p1_y0, p1_y1, p1_x0, p1_x1 = int(h * 0.70), h, 0, w

    # Phase 1+: landmark-guided tibial plateau
    tibia_y = [lm[f"tibia_{i}"][1] for i in range(1, N_POS + 1)]
    p1p_y0 = int(min(tibia_y))
    p1p_y1 = min(h, p1p_y0 + max(4, int(h * 0.15)))
    p1p_x0 = int(lm["bone_left_margin"][0])
    p1p_x1 = max(p1p_x0 + 1, int(lm["bone_right_margin"][0]))

    return (p1_y0, p1_y1, p1_x0, p1_x1), (p1p_y0, p1p_y1, p1p_x0, p1p_x1)


fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(12.5, 4.6 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    lm = detect_landmarks(path)
    (p1y0, p1y1, p1x0, p1x1), (p1py0, p1py1, p1px0, p1px1) = rois(gray, lm)

    ax0, ax1, ax2 = axes[row]

    ax0.imshow(gray, cmap="gray")
    ax0.add_patch(plt.Rectangle((p1x0, p1y0), p1x1 - p1x0 - 1, p1y1 - p1y0 - 1,
                                fill=False, edgecolor=RED, lw=2, label="Φάση 1 (τυφλό 30%)"))
    ax0.add_patch(plt.Rectangle((p1px0, p1py0), p1px1 - p1px0, p1py1 - p1py0,
                                fill=False, edgecolor=BLUE, lw=2, label="Φάση 1+ (landmark-guided)"))
    ax0.set_title(f"{label}\nΔύο ROIs πάνω στην ίδια εικόνα", fontsize=10)
    ax0.axis("off")
    if row == 0:
        handles, labels_ = ax0.get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", bbox_to_anchor=(0.5, -0.02),
                  fontsize=9, ncol=2, frameon=True)

    crop1 = gray[p1y0:p1y1, p1x0:p1x1]
    ax1.imshow(crop1, cmap="gray", aspect="auto")
    ax1.set_title(f"Φάση 1 crop\n({crop1.shape[1]}×{crop1.shape[0]} px)", fontsize=9.5, color=RED)
    ax1.axis("off")

    crop2 = gray[p1py0:p1py1, p1px0:p1px1]
    ax2.imshow(crop2, cmap="gray", aspect="auto")
    ax2.set_title(f"Φάση 1+ crop\n({crop2.shape[1]}×{crop2.shape[0]} px)", fontsize=9.5, color=BLUE)
    ax2.axis("off")

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Σύγκριση ROI Ομάδας 3 (Υφή Οστού): Φάση 1 vs Φάση 1+",
            fontsize=13, y=1.02)
fig.tight_layout()

for ext in ("png", "pdf"):
    out = OUT_DIR / f"texture_roi_comparison.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
