"""
Visual demo of baseline/features.py's shape_features() pipeline (Group 1 —
Παραμόρφωση οστού): original -> Otsu binary -> largest contour + bounding box,
with the extracted numbers annotated. Two examples (KL0 vs KL4) to show how a
more irregular/deformed contour differs from a typical, symmetric one.

Mirrors the exact logic of baseline/features.py's shape_features() /
_load_gray() for visualization only — does not import or modify that module's
internals (it doesn't expose the intermediate binary/contour).

Usage:
    python results/figures/shape_features_demo.py
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
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (σοβαρή παραμόρφωση)", "data/dataset/test/4/9055836R.png"),
]


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def shape_pipeline(gray):
    """Mirrors baseline/features.py shape_features(), but also returns the
    intermediate binary image / contour / bbox for plotting."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / (h + 1e-6)
    bbox_area = w * h
    extent = area / (bbox_area + 1e-6)

    M = cv2.moments(binary)
    hu = cv2.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return {
        "binary": binary, "contour": largest, "bbox": (x, y, w, h),
        "area": area, "perimeter": perimeter, "aspect_ratio": aspect_ratio,
        "extent": extent, "hu": hu_log,
    }


fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(12, 5.0 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    r = shape_pipeline(gray)

    ax0, ax1, ax2 = axes[row]

    ax0.imshow(gray, cmap="gray")
    ax0.set_title(f"{label}\nΑρχική (grayscale)", fontsize=10)
    ax0.axis("off")

    ax1.imshow(r["binary"], cmap="gray")
    ax1.set_title("Otsu threshold (binary)", fontsize=10)
    ax1.axis("off")

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [r["contour"]], -1, (0, 200, 0), 2)
    x, y, w, h = r["bbox"]
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 50, 220), 2)  # BGR -> red
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    ax2.imshow(overlay_rgb)
    ax2.set_title("Περίγραμμα + bounding box", fontsize=10)
    ax2.axis("off")

    hu_line1 = "   ".join(f"hu{i+1}={v:.2f}" for i, v in enumerate(r["hu"][:4]))
    hu_line2 = "   ".join(f"hu{i+1}={v:.2f}" for i, v in enumerate(r["hu"][4:], start=4))
    info = (f"area={r['area']:.0f}   perimeter={r['perimeter']:.0f}   "
            f"aspect_ratio={r['aspect_ratio']:.2f}   extent={r['extent']:.2f}\n"
            f"{hu_line1}\n{hu_line2}")
    ax2.text(0.5, -0.08, info, transform=ax2.transAxes, ha="center", va="top",
             fontsize=8.3, color=INK, linespacing=1.7)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Pipeline της Ομάδας 1 (Χαρακτηριστικά Σχήματος): shape_features()\n"
             "(OpenCV — cv2.HuMoments, cv2.findContours, cv2.threshold)",
            fontsize=13, y=1.03)
fig.tight_layout()
fig.subplots_adjust(hspace=0.8)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"shape_features_demo.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
