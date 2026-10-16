"""
Visual demo of baseline/features.py's texture_features() pipeline (Group 3 -
Υφή Οστού / Σκλήρυνση): bottom-30% crop -> LBP transform -> LBP histogram,
plus the 6 GLCM statistics as text.

Mirrors the exact logic of texture_features() for visualization only.

Usage:
    python results/figures/texture_features_demo.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

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
    ("KL4 (σοβαρή σκλήρυνση)", "data/dataset/test/4/9055836R.png"),
]

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def texture_pipeline(gray):
    """Mirrors baseline/features.py texture_features()."""
    h = gray.shape[0]
    sub = gray[int(h * 0.70):, :]

    sub8 = (sub // 16).astype(np.uint8)
    glcm = graycomatrix(sub8, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    glcm_feats = {p: float(graycoprops(glcm, p)[0, 0]) for p in GLCM_PROPS}

    lbp = local_binary_pattern(sub, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 8), density=True)

    return {"sub": sub, "lbp": lbp, "lbp_hist": lbp_hist, "glcm_feats": glcm_feats,
            "mean": float(sub.mean()), "std": float(sub.std())}


fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(15, 4.3 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    h = gray.shape[0]
    r = texture_pipeline(gray)

    ax0, ax1, ax2, ax3 = axes[row]

    ax0.imshow(gray, cmap="gray")
    ax0.axhline(int(h * 0.70), color=RED, ls="--", lw=1.5)
    ax0.add_patch(plt.Rectangle((0, int(h * 0.70)), gray.shape[1] - 1, h - int(h * 0.70) - 1,
                                fill=False, edgecolor=RED, lw=1.5))
    ax0.set_title(f"{label}\nΚάτω 30% (κνήμη)", fontsize=10)
    ax0.axis("off")

    ax1.imshow(r["sub"], cmap="gray")
    ax1.set_title("Υποπεριοχή (zoom)", fontsize=10)
    ax1.axis("off")

    im = ax2.imshow(r["lbp"], cmap="viridis")
    ax2.set_title("LBP transform", fontsize=10)
    ax2.axis("off")

    ax3.bar(range(8), r["lbp_hist"], color="#1baf7a", edgecolor=SURFACE)
    ax3.set_title("Ιστόγραμμα LBP (8 bins)", fontsize=10)
    ax3.set_xlabel("LBP bin", fontsize=8.5)
    ax3.set_ylabel("Συχνότητα", fontsize=8.5)
    ax3.tick_params(labelsize=7.5)

    glcm_line1 = "   ".join(f"{p}={v:.2f}" for p, v in list(r["glcm_feats"].items())[:3])
    glcm_line2 = "   ".join(f"{p}={v:.2f}" for p, v in list(r["glcm_feats"].items())[3:])
    info = f"GLCM: {glcm_line1}\n{glcm_line2}\nmean={r['mean']:.1f}   std={r['std']:.1f}"
    fig.text(0.5, 0.52 - row * 0.5, info, ha="center", va="top", fontsize=8.3, color=INK)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Pipeline της Ομάδας 3 (Υφή Οστού): texture_features()\n"
             "(scikit-image — graycomatrix/graycoprops, local_binary_pattern)",
            fontsize=12.5, y=1.03)
fig.tight_layout()
fig.subplots_adjust(hspace=0.95, wspace=0.35)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"texture_features_demo.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
