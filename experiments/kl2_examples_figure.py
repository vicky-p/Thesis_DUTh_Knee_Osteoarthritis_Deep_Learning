"""
Qualitative companion to experiments/kl2_analysis.py's headline finding: 126
of 447 true-KL2 test images (28.2%) are misclassified by all 7 models. This
picks a handful of those and renders them with what each model predicted, as
evidence for the "likely label noise" argument (both DL and RF disagree with
the ground truth, and mostly agree with EACH OTHER on a different grade).

Usage:
    python experiments/kl2_examples_figure.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import DATA_ROOT

RESULTS_DIR = Path(__file__).parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

MODELS = ["rf", "rf_landmarks", "vit", "resnet", "efficientnet", "hybrid", "hybrid_finetuned"]
DISPLAY_NAMES = {
    "rf": "RF", "rf_landmarks": "RF+LM", "vit": "ViT", "resnet": "ResNet",
    "efficientnet": "EffNet", "hybrid": "Hybrid", "hybrid_finetuned": "Hybrid-FT",
}


def load(name):
    rows = {}
    with open(PRED_DIR / f"{name}.csv") as f:
        for row in csv.DictReader(f):
            rows[row["path"]] = int(row["y_pred"])
    return rows


def main():
    preds = {n: load(n) for n in MODELS}

    # ground truth from any file's y_true column
    y_true = {}
    with open(PRED_DIR / "vit.csv") as f:
        for row in csv.DictReader(f):
            y_true[row["path"]] = int(row["y_true"])

    always_wrong = []
    for p, yt in y_true.items():
        if yt != 2:
            continue
        model_preds = {n: preds[n][p] for n in MODELS}
        if all(v != 2 for v in model_preds.values()):
            always_wrong.append((p, model_preds))

    print(f"  {len(always_wrong)} true-KL2 images misclassified by all {len(MODELS)} models")

    n_show = 6
    chosen = always_wrong[:n_show]

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for ax, (rel_path, model_preds) in zip(axes.flat, chosen):
        img_path = DATA_ROOT / rel_path
        img = Image.open(img_path).convert("L")
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        dl_preds = [model_preds[n] for n in ("vit", "resnet", "efficientnet", "hybrid", "hybrid_finetuned")]
        dl_consensus = max(set(dl_preds), key=dl_preds.count)
        pred_str = "  ".join(f"{DISPLAY_NAMES[n]}=KL{model_preds[n]}" for n in MODELS)
        title = (f"True: KL2  |  DL consensus: KL{dl_consensus}\n"
                 f"{pred_str}")
        ax.set_title(title, fontsize=8)

    for ax in axes.flat[len(chosen):]:
        ax.axis("off")

    fig.suptitle(
        f"KL2 test images misclassified by ALL 7 models "
        f"({len(always_wrong)}/447 = {100*len(always_wrong)/447:.1f}% of true-KL2 images)\n"
        f"Every model — hand-crafted RF, 3 CNN/ViT architectures, 2 hybrid variants — "
        f"disagrees with the ground truth here, evidence of label noise rather than a fixable model gap.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "kl2_label_noise_examples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
