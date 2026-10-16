"""
Standalone diagram: 3 boxes showing the data pipeline from raw OAI/MOST
through the Kaggle dataset's own preprocessing to our own preprocessing
(Section 3.1.3 of the thesis). Not part of plot_all.py's model-comparison
figures — a one-off methodology diagram, kept consistent in style (serif
font, 300 DPI, PNG+PDF).

Usage:
    python results/figures/preprocessing_pipeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).parent
DPI = 300

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
BOX_EDGE = "#2a78d6"
BOX_FACE = "#eaf2fc"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

BOXES = [
    {
        "title": "OAI / MOST\n(πρωτότυπα δεδομένα)",
        "lines": [
            "Πολυκεντρική κλινική μελέτη (NIH)",
            "Αμφοτερόπλευρες ακτινογραφίες + MRI",
            "Μεταβλητό μέγεθος/format ανά συνεδρία",
            "Κλινικά δεδομένα, WOMAC pain scores,",
            "παρακολούθηση σε βάθος χρόνου",
        ],
    },
    {
        "title": "Προεπεξεργασία\nKaggle Dataset",
        "lines": [
            "Crop γύρω από την άρθρωση γόνατος",
            "Μετατροπή σε grayscale",
            "Ενιαίο μέγεθος 224×224 px, μορφή PNG",
            "Ετικέτα KL grade (0–4) ανά φάκελο",
            "Διαχωρισμός σε train / val / test",
        ],
    },
    {
        "title": "Δική μας\nΠροεπεξεργασία",
        "lines": [
            "Resize 224×224 (ασφάλεια εισόδου)",
            "Grayscale → 3-channel RGB",
            "Κανονικοποίηση ImageNet (mean/std)",
            "Augmentation στο training split",
            "(RF: 256×256 grayscale + χειρ. χαρακτ.)",
        ],
    },
]

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

box_w, box_h = 3.6, 4.2
gap = 0.9
x0 = 0.3
y0 = 0.7

centers = []
for i, box in enumerate(BOXES):
    x = x0 + i * (box_w + gap)
    centers.append((x, x + box_w))

    rect = FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8, edgecolor=BOX_EDGE, facecolor=BOX_FACE,
    )
    ax.add_patch(rect)

    ax.text(x + box_w / 2, y0 + box_h - 0.45, box["title"],
            ha="center", va="top", fontsize=13, fontweight="bold", color=INK)

    ax.plot([x + 0.25, x + box_w - 0.25], [y0 + box_h - 1.05] * 2,
           color=BOX_EDGE, lw=1, alpha=0.6)

    ty = y0 + box_h - 1.35
    for line in box["lines"]:
        ax.text(x + 0.25, ty, f"• {line}", ha="left", va="top",
                fontsize=9.3, color=MUTED)
        ty -= 0.42

for i in range(len(BOXES) - 1):
    x_start = centers[i][1]
    x_end = centers[i + 1][0]
    arrow = FancyArrowPatch(
        (x_start + 0.05, y0 + box_h / 2), (x_end - 0.05, y0 + box_h / 2),
        arrowstyle="-|>", mutation_scale=22, linewidth=2, color=INK,
    )
    ax.add_patch(arrow)

ax.text(7, 5.25, "Από τα πρωτότυπα δεδομένα στην είσοδο των μοντέλων",
        ha="center", va="top", fontsize=14, color=INK)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = OUT_DIR / f"preprocessing_pipeline.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
