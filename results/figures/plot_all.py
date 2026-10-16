"""
Publication-quality figures for the diploma thesis.

Generates (to results/figures/, PNG + PDF, 300 DPI, serif font):
  1. training_curves_vit.png   — loss + kappa per epoch, frozen->finetune
                                  transition and best checkpoint marked
  2. comparison_bar.png        — grouped bars, all methods x 6 metrics,
                                  Thomson 2016 as a dashed reference line
  3. confusion_matrices.png    — Phase 1 RF | Phase 2 ViT | best model,
                                  normalized, Blues colormap
  4. class_distribution.png    — KL0..KL4 stacked counts, train/val/test
  5. per_class_comparison.png  — per-class accuracy, one line per method

Usage:
    python results/figures/plot_all.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from data.loader import DATA_ROOT

RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ── Style: validated 8-slot categorical palette (references/palette.md of the
# dataviz skill) — fixed order, never re-cycled, colorblind-safe on adjacent
# pairs (the comparison a grouped bar / multi-line chart actually needs). ──
SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRID        = "#e1e0d9"
BASELINE    = "#c3c2b7"

METHODS = [
    # display name,        results json,                    palette color
    ("RF",              "phase_1_rf_results.json",          "#2a78d6"),
    ("RF+LM",           "phase1plus_results.json",           "#eb6834"),
    ("ViT-B/16",        "phase_2_vit_v2_results.json",       "#1baf7a"),
    ("ResNet-50",       "resnet_results.json",               "#eda100"),
    ("EfficientNet-B4", "efficientnet_results.json",         "#e87ba4"),
    ("Hybrid",          "hybrid_results.json",               "#008300"),
    ("Hybrid-FT",       "hybridfinetuned_results.json",      "#4a3aa7"),
    ("ViT+ResNet Ens.", "ensemble_vit_resnet_results.json",  "#e34948"),
]

# Ordinal KL-severity ramp (clinical green-healthy -> red-severe convention;
# not an arbitrary categorical identity, so the ordered ramp is correct here)
KL_COLORS = ["#0ca30c", "#8bc34a", "#fab219", "#ec835a", "#d03b3b"]

THOMSON = {
    "Binary OA AUC": 0.904,
    "KL kappa (kw)": 0.490,
    # Thomson (2016) PhD thesis, Table 6.2 (combined model, 8875 OAI images)
    "KL1 accuracy": 0.036,
    "KL2 accuracy": 0.436,
    "KL3 accuracy": 0.499,
    "KL4 accuracy": 0.392,
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


def save(fig, name):
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved {name}.png + {name}.pdf")


def load_json(filename):
    p = RESULTS_DIR / filename
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_predictions(name):
    path = PRED_DIR / f"{name}.csv"
    if not path.exists():
        return None
    y_true, y_pred = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return np.array(y_true), np.array(y_pred)


def ensemble_predictions(names):
    """Recompute an ensemble's y_pred by averaging softmax probabilities
    from each member's predictions CSV (same method as experiments/ensemble.py)."""
    probas, y_true = [], None
    for name in names:
        path = PRED_DIR / f"{name}.csv"
        if not path.exists():
            return None
        rows = list(csv.DictReader(open(path)))
        proba = np.array([[float(r[f"p{i}"]) for i in range(5)] for r in rows])
        probas.append(proba)
        if y_true is None:
            y_true = np.array([int(r["y_true"]) for r in rows])
    avg = np.mean(probas, axis=0)
    return y_true, avg.argmax(axis=1)


# ── Figure 1: ViT training curves ────────────────────────────────────────────

def fig1_training_curves():
    log_path = RESULTS_DIR / "training_log_v2.csv"
    if not log_path.exists():
        print("  Skipping Figure 1 (training_curves_vit) — training_log_v2.csv not found")
        return

    epochs, tr_loss, va_loss, tr_kw, va_kw, lr = [], [], [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))
            tr_kw.append(float(row["train_kappa"]))
            va_kw.append(float(row["val_kappa"]))
            lr.append(float(row["lr"]))

    # Frozen -> finetune transition: LR resets to a fresh (much lower) value,
    # a far bigger single-step drop than the smooth cosine decay ever produces.
    transition_epoch = None
    for i in range(1, len(lr)):
        if lr[i] < lr[i - 1] / 3:
            transition_epoch = epochs[i]
            break

    best_idx = int(np.argmax(va_kw))
    best_epoch, best_kw = epochs[best_idx], va_kw[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, tr, va, ylabel, title in [
        (ax1, tr_loss, va_loss, "Cross-entropy loss", "Loss"),
        (ax2, tr_kw, va_kw, "Weighted kappa (kw)", "Weighted Kappa"),
    ]:
        ax.plot(epochs, tr, label="Train", color=METHODS[2][2], lw=2)
        ax.plot(epochs, va, label="Val", color=METHODS[7][2], lw=2)
        if transition_epoch:
            ax.axvline(transition_epoch, color=INK_MUTED, ls="--", lw=1.5,
                       label=f"Unfreeze (ep {transition_epoch})")
        ax.axvline(best_epoch, color=KL_COLORS[0], ls="--", lw=1.5,
                   label=f"Best checkpoint (ep {best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    ax2.scatter([best_epoch], [best_kw], color=KL_COLORS[0], zorder=5, s=60)
    ax2.annotate(f"kw={best_kw:.3f}", (best_epoch, best_kw),
                fontsize=9, color=INK_PRIMARY, xytext=(6, -10), textcoords="offset points")
    ax2.axhline(THOMSON["KL kappa (kw)"], color=INK_MUTED, ls=":", lw=1.2, alpha=0.8)
    ax2.annotate("Thomson 2016", (epochs[0], THOMSON["KL kappa (kw)"]),
                fontsize=8, color=INK_MUTED, xytext=(4, 4), textcoords="offset points")

    fig.suptitle("ViT-B/16 Training (partially frozen -> full fine-tune)", fontsize=13)
    fig.tight_layout()
    save(fig, "training_curves_vit")


# ── Figure 2: grouped comparison bar chart ───────────────────────────────────

def fig2_comparison_bar():
    metrics = ["Binary OA AUC", "KL kappa (kw)", "KL1 accuracy",
               "KL2 accuracy", "KL3 accuracy", "KL4 accuracy"]

    def get(result, metric):
        if result is None:
            return float("nan")
        m = {
            "Binary OA AUC": result.get("binary_oa_auc"),
            "KL kappa (kw)": result.get("weighted_kappa"),
            "KL1 accuracy": result.get("per_class_accuracy", {}).get("KL1"),
            "KL2 accuracy": result.get("per_class_accuracy", {}).get("KL2"),
            "KL3 accuracy": result.get("per_class_accuracy", {}).get("KL3"),
            "KL4 accuracy": result.get("per_class_accuracy", {}).get("KL4"),
        }
        return m.get(metric, float("nan"))

    results = [(name, load_json(fname), color) for name, fname, color in METHODS]
    results = [(n, r, c) for n, r, c in results if r is not None]
    if not results:
        print("  Skipping Figure 2 (comparison_bar) — no result JSONs found")
        return

    n_bars = len(results)
    x = np.arange(len(metrics))
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(3 + 2.6 * n_bars, 7))

    for i, (name, result, color) in enumerate(results):
        vals = [get(result, m) for m in metrics]
        offset = (i - (n_bars - 1) / 2) * width
        rects = ax.bar(x + offset, vals, width, label=name, color=color,
                       edgecolor=SURFACE, linewidth=0.5)
        for rect, val in zip(rects, vals):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color=INK_PRIMARY)

    for i, m in enumerate(metrics):
        thomson_val = THOMSON.get(m)
        if thomson_val is not None:
            ax.plot([x[i] - 0.42, x[i] + 0.42], [thomson_val, thomson_val],
                    color=INK_PRIMARY, ls="--", lw=1.5, zorder=4,
                    label="Thomson 2016" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score")
    ax.set_title("All models vs Thomson 2016 — every headline metric", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "comparison_bar")


# ── Figure 3: confusion matrices ─────────────────────────────────────────────

def _normalized_cm(y_true, y_pred, n_classes=5):
    cm = np.zeros((n_classes, n_classes))
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100


def fig3_confusion_matrices():
    # Pick the best model by weighted kappa among everything we have.
    kw_by_method = {}
    for name, fname, _ in METHODS:
        r = load_json(fname)
        if r is not None:
            kw_by_method[name] = r.get("weighted_kappa", -1)
    if not kw_by_method:
        print("  Skipping Figure 3 (confusion_matrices) — no results found")
        return
    best_name = max(kw_by_method, key=kw_by_method.get)
    best_kw = kw_by_method[best_name]

    panels = []
    rf = load_predictions("rf")
    if rf is not None:
        panels.append(("Phase 1 — Random Forest", rf, load_json("phase_1_rf_results.json")))
    vit = load_predictions("vit")
    if vit is not None:
        panels.append(("Phase 2 — ViT-B/16", vit, load_json("phase_2_vit_v2_results.json")))

    if best_name == "ViT+ResNet Ens.":
        best_pred = ensemble_predictions(["vit", "resnet"])
    else:
        key = {"RF": "rf", "RF+LM": "rf_landmarks", "ViT-B/16": "vit", "ResNet-50": "resnet",
               "EfficientNet-B4": "efficientnet", "Hybrid": "hybrid",
               "Hybrid-FT": "hybrid_finetuned"}.get(best_name)
        best_pred = load_predictions(key) if key else None
    if best_pred is not None:
        panels.append((f"Best model — {best_name}", best_pred, {"weighted_kappa": best_kw}))

    if not panels:
        print("  Skipping Figure 3 (confusion_matrices) — no per-image predictions found")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5))
    if len(panels) == 1:
        axes = [axes]

    labels = [f"KL{i}" for i in range(5)]
    for ax, (title, (y_true, y_pred), result) in zip(axes, panels):
        cm_pct = _normalized_cm(y_true, y_pred)
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                val = cm_pct[i, j]
                color = "white" if val > 55 else INK_PRIMARY
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(5)); ax.set_xticklabels(labels)
        ax.set_yticks(range(5)); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        kw = result.get("weighted_kappa") if result else None
        kw_str = f"  (kw={kw:.3f})" if kw is not None else ""
        ax.set_title(f"{title}{kw_str}", fontsize=10)
        ax.grid(False)

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="% of true class")
    fig.suptitle("Confusion matrices (row-normalized)", fontsize=13)
    save(fig, "confusion_matrices")


# ── Figure 4: class distribution ─────────────────────────────────────────────

def fig4_class_distribution():
    splits = ["train", "val", "test"]
    counts = {s: [0] * 5 for s in splits}
    for s in splits:
        for g in range(5):
            class_dir = DATA_ROOT / s / str(g)
            if class_dir.exists():
                counts[s][g] = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))

    if all(sum(c) == 0 for c in counts.values()):
        print("  Skipping Figure 4 (class_distribution) — dataset not found locally")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(splits))
    bottoms = np.zeros(len(splits))
    for g in range(5):
        vals = np.array([counts[s][g] for s in splits])
        rects = ax.bar(x, vals, bottom=bottoms, color=KL_COLORS[g],
                       label=f"KL{g}", edgecolor=SURFACE, linewidth=1.5)
        for rect, val, bottom in zip(rects, vals, bottoms):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, bottom + val / 2,
                        str(int(val)), ha="center", va="center", fontsize=9,
                        color="white" if g in (0, 4) else INK_PRIMARY)
        bottoms += vals

    for i, s in enumerate(splits):
        ax.text(i, bottoms[i] + max(bottoms) * 0.02, f"n={int(bottoms[i])}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution by split (KL0=healthy -> KL4=severe)", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_ylim(0, max(bottoms) * 1.12)
    fig.tight_layout()
    save(fig, "class_distribution")


# ── Figure 5: per-class accuracy comparison ──────────────────────────────────

def fig5_per_class_comparison():
    grades = [f"KL{i}" for i in range(5)]
    fig, ax = plt.subplots(figsize=(9, 6))

    any_plotted = False
    for name, fname, color in METHODS:
        r = load_json(fname)
        if r is None:
            continue
        pc = r.get("per_class_accuracy", {})
        vals = [pc.get(g) for g in grades]
        if all(v is None for v in vals):
            continue
        any_plotted = True
        ax.plot(grades, vals, marker="o", markersize=7, lw=2, color=color,
               label=name, markeredgecolor=SURFACE, markeredgewidth=1)

    if not any_plotted:
        print("  Skipping Figure 5 (per_class_comparison) — no per-class results found")
        plt.close(fig)
        return

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("KL grade")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy by method", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "per_class_comparison")


def main():
    print(f"\n  Generating figures -> {FIGURES_DIR}\n")
    set_style()
    fig1_training_curves()
    fig2_comparison_bar()
    fig3_confusion_matrices()
    fig4_class_distribution()
    fig5_per_class_comparison()
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
