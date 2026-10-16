"""
Visual demo of baseline/features.py's global_stats() pipeline (Group 5 -
Γενικά Στατιστικά): top/bottom half split -> whole-image brightness
histogram -> top-half vs bottom-half histograms.

Mirrors the exact logic of global_stats() for visualization only.

Usage:
    python results/figures/global_stats_demo.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

OUT_DIR = Path(__file__).parent
DPI = 300
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
RED = "#e34948"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

EXAMPLES = [
    ("KL0 (υγιές)", "data/dataset/test/0/9003175L.png"),
    ("KL4 (σοβαρή αλλοίωση)", "data/dataset/test/4/9055836R.png"),
]


def load_gray(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256))


def global_pipeline(gray):
    """Mirrors baseline/features.py global_stats()."""
    flat = gray.flatten().astype(np.float64)
    half = gray.shape[0] // 2
    top_half = gray[:half, :].flatten().astype(np.float64)
    bot_half = gray[half:, :].flatten().astype(np.float64)
    return {
        "half": half, "flat": flat, "top_half": top_half, "bot_half": bot_half,
        "mean": flat.mean(), "std": flat.std(), "skew": skew(flat), "kurt": kurtosis(flat),
        "top_mean": top_half.mean(), "top_std": top_half.std(),
        "bot_mean": bot_half.mean(), "bot_std": bot_half.std(),
    }


fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(13, 4.3 * len(EXAMPLES)))

for row, (label, path) in enumerate(EXAMPLES):
    gray = load_gray(path)
    h, w = gray.shape
    r = global_pipeline(gray)

    ax0, ax1, ax2 = axes[row]

    ax0.imshow(gray, cmap="gray")
    ax0.axhline(r["half"], color=RED, lw=1.8)
    ax0.text(5, r["half"] - 5, "πάνω μισό", color=RED, fontsize=8, va="bottom")
    ax0.text(5, r["half"] + 5, "κάτω μισό", color=RED, fontsize=8, va="top")
    ax0.set_title(f"{label}\nΔιαχωρισμός σε μισά", fontsize=10)
    ax0.axis("off")

    ax1.hist(r["flat"], bins=40, color=INK, alpha=0.75)
    ax1.axvline(r["mean"], color=RED, ls="--", lw=1.3, label=f"mean={r['mean']:.0f}")
    ax1.set_title("Ιστόγραμμα όλης της εικόνας", fontsize=10)
    ax1.set_xlabel("Φωτεινότητα", fontsize=8.5)
    ax1.set_ylabel("Πλήθος pixels", fontsize=8.5)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.tick_params(labelsize=7.5)

    ax2.hist(r["top_half"], bins=40, color=BLUE, alpha=0.6, label=f"πάνω (mean={r['top_mean']:.0f})")
    ax2.hist(r["bot_half"], bins=40, color=ORANGE, alpha=0.6, label=f"κάτω (mean={r['bot_mean']:.0f})")
    ax2.set_title("Πάνω vs κάτω μισό", fontsize=10)
    ax2.set_xlabel("Φωτεινότητα", fontsize=8.5)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.tick_params(labelsize=7.5)

    info = (f"mean={r['mean']:.1f}  std={r['std']:.1f}  skew={r['skew']:.2f}  kurtosis={r['kurt']:.2f}   |   "
            f"Δ(πάνω-κάτω)={r['top_mean']-r['bot_mean']:.1f}")
    ax1.text(1.1, -0.17, info, transform=ax1.transAxes, ha="center", va="top",
             fontsize=8.3, color=INK)

fig.patch.set_facecolor(SURFACE)
fig.suptitle("Pipeline της Ομάδας 5 (Γενικά Στατιστικά): global_stats()\n"
             "(NumPy — mean/std · SciPy — skew, kurtosis)",
            fontsize=12.5, y=1.03)
fig.tight_layout()
fig.subplots_adjust(hspace=0.8, wspace=0.3)

for ext in ("png", "pdf"):
    out = OUT_DIR / f"global_stats_demo.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
