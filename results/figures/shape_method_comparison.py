"""
Phase 1 vs Phase 1+ comparison for the Shape group: Otsu contour + Hu moments
(baseline/features.py) versus landmark-based curvature + alignment angles
(baseline/landmark_features.py), on the same image.

Usage:
    python results/figures/shape_method_comparison.py
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


def phase1_panel(gray):
    """Otsu threshold + largest contour + bounding box (baseline/features.py)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [largest], -1, (0, 200, 0), 2)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 50, 220), 2)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dx, dy + 1e-6)))


def phase1plus_panel(gray, lm):
    """Landmark curvature points + alignment angle lines (baseline/landmark_features.py)."""
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).copy()
    fs, ts = lm["femur_shaft"], lm["tibia_shaft"]
    jc = lm["joint_3"]
    f3, t3 = lm["femur_3"], lm["tibia_3"]

    def pt(p):
        return (int(p[0]), int(p[1]))

    # varus/valgus reference line: femur_shaft -> joint_3 -> tibia_shaft
    cv2.line(overlay, pt(fs), pt(jc), (0, 165, 255), 2)
    cv2.line(overlay, pt(jc), pt(ts), (0, 165, 255), 2)
    # femur / tibia axis lines
    cv2.line(overlay, pt(fs), pt(f3), (255, 80, 0), 1)
    cv2.line(overlay, pt(t3), pt(ts), (0, 80, 255), 1)

    for i in range(1, N_POS + 1):
        cv2.circle(overlay, pt(lm[f"femur_{i}"]), 3, (255, 80, 0), -1)
        cv2.circle(overlay, pt(lm[f"tibia_{i}"]), 3, (0, 80, 255), -1)
    for p in (fs, jc, ts):
        cv2.circle(overlay, pt(p), 4, (0, 165, 255), -1)

    vv_angle = angle(fs, jc) - angle(jc, ts)
    same_x = fs[0] == jc[0] == ts[0]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), vv_angle, same_x


fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(9, 5.0 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    lm = detect_landmarks(path)

    p1_img = phase1_panel(gray)
    p1p_img, vv, same_x = phase1plus_panel(gray, lm)

    ax0, ax1 = axes[row]
    ax0.imshow(p1_img)
    ax0.set_title(f"{label}\nΦάση 1: Otsu περίγραμμα + Hu moments", fontsize=9.5, color=RED)
    ax0.axis("off")

    ax1.imshow(p1p_img)
    ax1.set_title(f"Φάση 1+: landmarks + καμπυλότητα κονδύλου/πλατώματος\n(5+5 τιμές ανά οστό)", fontsize=9.5, color=BLUE)
    ax1.axis("off")
    if same_x:
        ax1.text(0.5, -0.06,
                  "* η γωνία varus/valgus (πορτοκαλί γραμμή) βγαίνει πάντα 0°:\n"
                  "τα 3 σημεία αναφοράς μοιράζονται το ίδιο x -- γνωστός περιορισμός",
                  transform=ax1.transAxes, ha="center", va="top", fontsize=7.3, color=INK)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Σύγκριση μεθόδου Ομάδας 1 (Σχήμα): Φάση 1 vs Φάση 1+",
            fontsize=13, y=1.02)
fig.tight_layout()
fig.subplots_adjust(hspace=0.45)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"shape_method_comparison.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
