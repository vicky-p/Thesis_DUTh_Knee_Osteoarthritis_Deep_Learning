"""
Snake/zigzag version of the preprocessing pipeline diagram: 3 rows (one per
stage), each box = one bullet point, connected by arrows that wind back and
forth (row 1 L->R, row 2 R->L, row 3 L->R), like Mermaid's layout engine was
asked for but wouldn't reliably render across subgraphs.

Usage:
    python results/figures/preprocessing_pipeline_snake.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).parent
DPI = 300
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
})

STAGES = [
    {
        "title": "OAI / MOST  (πρωτότυπα δεδομένα)",
        "edge": "#2a78d6", "face": "#eaf2fc",
        "items": [
            "Πολυκεντρική κλινική\nμελέτη (NIH)",
            "Αμφοτερόπλευρες\nακτινογραφίες + MRI",
            "Μεταβλητό μέγεθος/\nformat ανά συνεδρία",
            "Κλινικά δεδομένα,\nWOMAC pain scores",
            "Παρακολούθηση\nσε βάθος χρόνου",
        ],
        "direction": "ltr",
    },
    {
        "title": "Προεπεξεργασία Kaggle Dataset",
        "edge": "#eb6834", "face": "#fef1ea",
        "items": [
            "Crop γύρω από\nτην άρθρωση γόνατος",
            "Μετατροπή σε\ngrayscale",
            "Ενιαίο μέγεθος\n224×224 px, PNG",
            "Ετικέτα KL grade\n(0-4) ανά φάκελο",
            "Διαχωρισμός σε\ntrain / val / test",
        ],
        "direction": "rtl",
    },
    {
        "title": "Δική μας Προεπεξεργασία",
        "edge": "#1baf7a", "face": "#e8f8f0",
        "items": [
            "Resize 224×224\n(ασφάλεια εισόδου)",
            "Grayscale →\n3-channel RGB",
            "Κανονικοποίηση\nImageNet (mean/std)",
            "Augmentation στο\ntraining split",
            "RF: 256×256 grayscale\n+ χειρ. χαρακτηριστικά",
        ],
        "direction": "ltr",
    },
]

N_COLS = 5
BOX_W, BOX_H = 2.5, 1.5
COL_GAP = 0.55
ROW_GAP = 1.6
X0, Y_TOP = 0.6, 8.6

col_x = [X0 + c * (BOX_W + COL_GAP) for c in range(N_COLS)]
fig_w = col_x[-1] + BOX_W + 0.6
n_rows = len(STAGES)
y_bottom_last_row = Y_TOP - (n_rows - 1) * (BOX_H + ROW_GAP) - BOX_H
ylim_bottom = y_bottom_last_row - 0.4
ylim_top = Y_TOP + 1.9
fig_h = ylim_top - ylim_bottom

fig, ax = plt.subplots(figsize=(fig_w * 0.78, fig_h * 0.62))
ax.set_xlim(0, fig_w)
ax.set_ylim(ylim_bottom, ylim_top)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

row_centers = []  # (y_center, [box_centers_x in display order left..right by col index])

for r, stage in enumerate(STAGES):
    y_top = Y_TOP - r * (BOX_H + ROW_GAP)
    y_center = y_top - BOX_H / 2
    row_centers.append(y_center)

    # column index -> item index, depending on direction
    cols = list(range(N_COLS))
    items_in_col_order = stage["items"] if stage["direction"] == "ltr" else list(reversed(stage["items"]))

    ax.text(X0, y_top + 0.55, stage["title"], ha="left", va="bottom",
            fontsize=13, fontweight="bold", color=stage["edge"])

    box_centers_x = []
    for c in cols:
        x = col_x[c]
        box_centers_x.append(x + BOX_W / 2)
        rect = FancyBboxPatch(
            (x, y_top - BOX_H), BOX_W, BOX_H,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            linewidth=1.6, edgecolor=stage["edge"], facecolor=stage["face"],
        )
        ax.add_patch(rect)
        ax.text(x + BOX_W / 2, y_top - BOX_H / 2, items_in_col_order[c],
                ha="center", va="center", fontsize=8.7, color=INK, linespacing=1.4)

    # within-row arrows, following reading direction (ltr: left->right; rtl: right->left)
    seq = cols if stage["direction"] == "ltr" else list(reversed(cols))
    for i in range(len(seq) - 1):
        xa = col_x[seq[i]] + (BOX_W if stage["direction"] == "ltr" else 0)
        xb = col_x[seq[i + 1]] + (0 if stage["direction"] == "ltr" else BOX_W)
        arrow = FancyArrowPatch((xa + (0.04 if stage["direction"] == "ltr" else -0.04), y_center),
                                (xb + (-0.04 if stage["direction"] == "ltr" else 0.04), y_center),
                                arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=INK)
        ax.add_patch(arrow)

    stage["_last_x_center"] = box_centers_x[-1] if stage["direction"] == "ltr" else box_centers_x[0]
    stage["_first_x_center"] = box_centers_x[0] if stage["direction"] == "ltr" else box_centers_x[-1]
    stage["_y_top"] = y_top
    stage["_y_bottom"] = y_top - BOX_H

# Connect row i's last box (end of its reading direction) down to row i+1's first box
for i in range(len(STAGES) - 1):
    s_from, s_to = STAGES[i], STAGES[i + 1]
    x_from = s_from["_last_x_center"]
    x_to = s_to["_first_x_center"]
    y_from = s_from["_y_bottom"]
    y_to = s_to["_y_top"]
    arrow = FancyArrowPatch((x_from, y_from - 0.04), (x_to, y_to + 0.04),
                            arrowstyle="-|>", mutation_scale=18, linewidth=1.8,
                            color=INK, connectionstyle="arc3,rad=0.0")
    ax.add_patch(arrow)

ax.text(fig_w / 2, ylim_top - 0.2, "Από τα πρωτότυπα δεδομένα στην είσοδο των μοντέλων",
        ha="center", va="top", fontsize=14, color=INK)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = OUT_DIR / f"preprocessing_pipeline_snake.{ext}"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {out}")
plt.close(fig)
