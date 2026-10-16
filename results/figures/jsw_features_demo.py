"""
Visual demo of baseline/features.py's joint_space_features() pipeline
(Group 2 - Στένωση αρθρικού χώρου): center strip -> row-brightness profile ->
joint_row (argmax) -> 17-point width measurement.

Mirrors the exact logic of joint_space_features() for visualization only.

Note: this heuristic (brightest row in the central 40%-width strip, over the
FULL image height) often locks onto the femoral shaft rather than the true
joint line (see baseline/landmarks.py, _find_joint_row). It is reproduced
here deliberately so the figure shows what the Phase 1 pipeline actually
does. An approximate "typical true joint line" (~52% of height, from visual
inspection of sample images) is overlaid for comparison.

Usage:
    python results/figures/jsw_features_demo.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
DPI = 300
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
RED = "#e34948"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (σοβαρή στένωση)", "data/dataset/test/4/9055836R.png"),
]

TYPICAL_TRUE_JOINT_FRACTION = 0.52  # approximate, from visual inspection of sample images


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def jsw_pipeline(gray):
    """Mirrors baseline/features.py joint_space_features()."""
    h, w = gray.shape
    x0, x1 = int(w * 0.30), int(w * 0.70)
    strip = gray[:, x0:x1]
    row_means = strip.mean(axis=1)
    joint_row = int(np.argmax(row_means))

    band_h = max(1, int(h * 0.10))
    band = gray[max(0, joint_row - band_h):min(h, joint_row + band_h), x0:x1]
    band_mean = float(band.mean())

    threshold = band_mean
    sample_rows = np.linspace(max(0, joint_row - band_h), min(h - 1, joint_row + band_h), 17, dtype=int)
    widths = []
    for row_idx in sample_rows:
        row = gray[row_idx, x0:x1]
        widths.append(int((row > threshold).sum()))

    return {
        "x0": x0, "x1": x1, "row_means": row_means, "joint_row": joint_row,
        "band_h": band_h, "sample_rows": sample_rows, "widths": widths,
        "mean_width": float(np.mean(widths)),
    }


fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(13, 4.6 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    r = jsw_pipeline(gray)
    h, w = gray.shape
    true_row = int(h * TYPICAL_TRUE_JOINT_FRACTION)

    ax0, ax1, ax2 = axes[row]

    ax0.imshow(gray, cmap="gray")
    ax0.axvline(r["x0"], color=BLUE, ls="--", lw=1.3)
    ax0.axvline(r["x1"], color=BLUE, ls="--", lw=1.3)
    ax0.set_title(f"{label}\nΚεντρική λωρίδα (40% πλάτους)", fontsize=10)
    ax0.axis("off")

    ax1.plot(r["row_means"], np.arange(h), color=INK, lw=1.5)
    ax1.axhline(r["joint_row"], color=RED, ls="-", lw=1.5,
               label=f"joint_row (argmax) = {r['joint_row']}")
    ax1.axhline(true_row, color=MUTED, ls=":", lw=1.5,
               label=f"τυπική πραγματική θέση (~{int(TYPICAL_TRUE_JOINT_FRACTION*100)}%)")
    ax1.invert_yaxis()
    ax1.set_xlabel("Μέση φωτεινότητα γραμμής", fontsize=8.5)
    ax1.set_ylabel("Γραμμή (pixel row)", fontsize=8.5)
    ax1.set_title("Προφίλ φωτεινότητας ανά γραμμή", fontsize=10)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.tick_params(labelsize=7.5)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for row_idx in r["sample_rows"]:
        for xx in range(r["x0"], r["x1"], 3):
            overlay[row_idx, xx] = (0, 200, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    ax2.imshow(overlay_rgb)
    ax2.axhline(r["joint_row"], color=RED, lw=1.3)
    ax2.axhline(true_row, color=MUTED, ls=":", lw=1.3)
    ax2.set_title("17 σημεία μέτρησης πλάτους", fontsize=10)
    ax2.axis("off")

    gap_pct = abs(r["joint_row"] - true_row) / h * 100
    info = (f"μέσο πλάτος={r['mean_width']:.0f} px   |   "
            f"απόσταση joint_row απο τυπική θέση: {gap_pct:.0f}% του ύψους")
    ax2.text(0.5, -0.08, info, transform=ax2.transAxes, ha="center", va="top",
             fontsize=8.3, color=INK)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Pipeline της Ομάδας 2 (Κενό Αρθρικού Χώρου): joint_space_features()\n"
             "Κόκκινο = ανιχνευμένο joint_row (argmax) · Γκρι διάστικτο = τυπική πραγματική θέση άρθρωσης",
            fontsize=12.5, y=1.03)
fig.tight_layout()
fig.subplots_adjust(hspace=0.75, wspace=0.35)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"jsw_features_demo.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
