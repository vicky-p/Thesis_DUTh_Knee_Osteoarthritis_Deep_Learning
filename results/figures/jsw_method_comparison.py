"""
Phase 1 vs Phase 1+ comparison for the Joint Space Width group: blind
brightest-row heuristic (baseline/features.py) versus direct landmark-to-
landmark vertical gap (baseline/landmark_features.py), on the same image.

Usage:
    python results/figures/jsw_method_comparison.py
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
MUTED = "#898781"
RED = "#e34948"
BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (σοβαρή στένωση)", "data/dataset/test/4/9055836R.png"),
]
N_POS = len(X_FRACTIONS)
TYPICAL_TRUE_JOINT_FRACTION = 0.52  # approximate, from visual inspection of sample images


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def phase1_panel(gray):
    """Mirrors baseline/features.py joint_space_features(): blind brightest row."""
    h, w = gray.shape
    x0, x1 = int(w * 0.30), int(w * 0.70)
    strip = gray[:, x0:x1]
    row_means = strip.mean(axis=1)
    joint_row = int(np.argmax(row_means))
    true_row = int(h * TYPICAL_TRUE_JOINT_FRACTION)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(overlay, (x0, joint_row), (x1, joint_row), (50, 50, 220), 2)
    cv2.line(overlay, (x0, true_row), (x1, true_row), (150, 150, 150), 1)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), joint_row, true_row


def phase1plus_panel(gray, lm):
    """Mirrors baseline/landmark_features.py joint_space_features(): direct gaps."""
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gaps = []
    for i in range(1, N_POS + 1):
        fx, fy = lm[f"femur_{i}"]
        tx, ty = lm[f"tibia_{i}"]
        color = (255, 150, 0) if i in (1, 2) else ((0, 80, 255) if i in (4, 5) else (0, 200, 0))
        cv2.line(overlay, (int(fx), int(fy)), (int(tx), int(ty)), color, 2)
        cv2.circle(overlay, (int(fx), int(fy)), 3, (255, 80, 0), -1)
        cv2.circle(overlay, (int(tx), int(ty)), 3, (0, 80, 255), -1)
        gaps.append(ty - fy)
    gaps = np.array(gaps)
    medial, lateral, central = gaps[:2].mean(), gaps[3:].mean(), gaps[2]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), gaps, medial, lateral, central


fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(9, 5.0 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    lm = detect_landmarks(path)

    p1_img, joint_row, true_row = phase1_panel(gray)
    p1p_img, gaps, medial, lateral, central = phase1plus_panel(gray, lm)

    ax0, ax1 = axes[row]
    ax0.imshow(p1_img)
    gap_pct = abs(joint_row - true_row) / gray.shape[0] * 100
    ax0.set_title(f"{label}\nΦάση 1: πιο φωτεινή γραμμή (τυφλό)", fontsize=9.5, color=RED)
    ax0.axis("off")
    ax0.text(0.5, -0.06,
              f"κόκκινο=joint_row (argmax) · γκρι=τυπική πραγματική θέση\n"
              f"απόκλιση: {gap_pct:.0f}% του ύψους",
              transform=ax0.transAxes, ha="center", va="top", fontsize=7.3, color=INK)

    ax1.imshow(p1p_img)
    ax1.set_title("Φάση 1+: κενό femur→tibia σε 5 θέσεις", fontsize=9.5, color=BLUE)
    ax1.axis("off")
    ax1.text(0.5, -0.06,
              f"έσω={medial:.1f}px (πορτοκαλί) · κεντρικό={central:.1f}px (πράσινο) · "
              f"έξω={lateral:.1f}px (μπλε)",
              transform=ax1.transAxes, ha="center", va="top", fontsize=7.3, color=INK)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Σύγκριση μεθόδου Ομάδας 2 (Αρθρικός Χώρος): Φάση 1 vs Φάση 1+",
            fontsize=13, y=1.02)
fig.tight_layout()
fig.subplots_adjust(hspace=0.45)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"jsw_method_comparison.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
