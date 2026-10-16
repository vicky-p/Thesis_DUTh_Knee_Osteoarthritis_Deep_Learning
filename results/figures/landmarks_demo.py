"""
Visual demo of baseline/landmarks.py's detect_landmarks() / visualize_landmarks()
pipeline (Phase 2A - heuristic landmark detection): original image side by
side with the 19 detected landmarks overlaid.

Calls the real production functions directly (not reimplemented) - this is
exactly what the pipeline produces, no idealization.

Usage:
    python results/figures/landmarks_demo.py
"""
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from baseline.landmarks import detect_landmarks, visualize_landmarks

OUT_DIR = Path(__file__).parent
DPI = 300
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (σοβαρή αλλοίωση)", "data/dataset/test/4/9055836R.png"),
]

TMP_DIR = Path("/tmp/landmarks_demo")
TMP_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(8, 4.3 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    lm = detect_landmarks(path)
    out_png = TMP_DIR / f"demo_{row}.png"
    visualize_landmarks(path, lm, save_path=out_png)

    orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (256, 256))
    overlay = cv2.cvtColor(cv2.imread(str(out_png)), cv2.COLOR_BGR2RGB)

    ax0, ax1 = axes[row]
    ax0.imshow(orig)
    ax0.set_title(f"{label}\nΑρχική", fontsize=10)
    ax0.axis("off")

    ax1.imshow(overlay)
    ax1.set_title("19 landmarks (detect_landmarks)", fontsize=10)
    ax1.axis("off")

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Phase 2A — Heuristic Landmark Detection\n"
             "Μπλε=femur · Κόκκινο=tibia · Κίτρινο=joint · Πράσινο=πλευρικά όρια · Μωβ=shaft",
            fontsize=12, y=1.02)
fig.tight_layout()

for ext in ("png", "pdf"):
    out = OUT_DIR / f"landmarks_demo.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
