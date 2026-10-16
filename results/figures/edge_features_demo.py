"""
Visual demo of baseline/features.py's edge_features() pipeline (Group 4 -
Άκρες/Οστεόφυτα): top/bottom 20% regions -> Canny edges, Sobel gradient
magnitude (full image), HOG visualization.

Mirrors the exact logic of edge_features() for visualization only (HOG is
additionally called with visualize=True here, just to render hog_image - the
real pipeline only keeps the first 15 feature values, no image).

Usage:
    python results/figures/edge_features_demo.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

OUT_DIR = Path(__file__).parent
DPI = 300
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
RED = "#e34948"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (οστεόφυτα)", "data/dataset/test/4/9055836R.png"),
]


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def edge_pipeline(gray):
    """Mirrors baseline/features.py edge_features()."""
    h, w = gray.shape
    top = gray[:int(h * 0.20), :]
    bottom = gray[int(h * 0.80):, :]

    def _canny_counts(region):
        edges = cv2.Canny(region, 50, 150)
        mid = region.shape[1] // 2
        return edges, float(edges[:, :mid].sum() / 255), float(edges[:, mid:].sum() / 255)

    top_edges, top_med, top_lat = _canny_counts(top)
    bot_edges, bot_med, bot_lat = _canny_counts(bottom)

    canny_full = np.zeros_like(gray)
    canny_full[:int(h * 0.20), :] = top_edges
    canny_full[int(h * 0.80):, :] = bot_edges

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    gray_hog = cv2.resize(gray, (128, 128))
    _, hog_image = hog(gray_hog, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm="L2-Hys",
                       feature_vector=True, visualize=True)

    return {"canny_full": canny_full, "mag": mag, "hog_image": hog_image,
            "top_med": top_med, "top_lat": top_lat, "bot_med": bot_med, "bot_lat": bot_lat,
            "sobel_mean": float(mag.mean()), "sobel_std": float(mag.std()), "sobel_max": float(mag.max())}


fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(15, 4.3 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    h, w = gray.shape
    r = edge_pipeline(gray)

    ax0, ax1, ax2, ax3 = axes[row]

    ax0.imshow(gray, cmap="gray")
    ax0.add_patch(plt.Rectangle((0, 0), w - 1, int(h * 0.20) - 1, fill=False, edgecolor=RED, lw=1.5))
    ax0.add_patch(plt.Rectangle((0, int(h * 0.80)), w - 1, int(h * 0.20) - 1, fill=False, edgecolor=RED, lw=1.5))
    ax0.set_title(f"{label}\nΠάνω/κάτω 20%", fontsize=10)
    ax0.axis("off")

    ax1.imshow(r["canny_full"], cmap="gray")
    ax1.set_title("Canny edges (στις 2 περιοχές)", fontsize=10)
    ax1.axis("off")

    ax2.imshow(r["mag"], cmap="inferno")
    ax2.set_title("Sobel gradient magnitude", fontsize=10)
    ax2.axis("off")

    ax3.imshow(r["hog_image"], cmap="gray")
    ax3.set_title("HOG (οπτικοποίηση)", fontsize=10)
    ax3.axis("off")

    info = (f"top: med={r['top_med']:.0f}  lat={r['top_lat']:.0f}   |   "
            f"bottom: med={r['bot_med']:.0f}  lat={r['bot_lat']:.0f}\n"
            f"Sobel: mean={r['sobel_mean']:.1f}  std={r['sobel_std']:.1f}  max={r['sobel_max']:.0f}")
    ax1.text(1.1, -0.12, info, transform=ax1.transAxes, ha="center", va="top",
             fontsize=8.3, color=INK)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Pipeline της Ομάδας 4 (Άκρες/Οστεόφυτα): edge_features()\n"
             "(OpenCV — Canny, Sobel · scikit-image — hog)",
            fontsize=12.5, y=1.03)
fig.tight_layout()
fig.subplots_adjust(hspace=0.9, wspace=0.25)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"edge_features_demo.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
