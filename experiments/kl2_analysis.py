"""
KL2 is the hardest class for every deep-learning model in this project.
This checks whether it's the *same* images failing across all models
(pointing to label noise / genuinely ambiguous cases) or *different* images
per model (pointing to model-specific weaknesses that a better model could fix).

Usage:
    python experiments/kl2_analysis.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"

MODELS = ["rf", "rf_landmarks", "vit", "resnet", "efficientnet", "hybrid", "hybrid_finetuned"]


def load_predictions(name):
    path = PRED_DIR / f"{name}.csv"
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["path"]] = (int(row["y_true"]), int(row["y_pred"]))
    return rows


def main():
    preds = {name: load_predictions(name) for name in MODELS if (PRED_DIR / f"{name}.csv").exists()}
    names = list(preds.keys())

    # All KL2 ground-truth images (from any one model's file — labels agree)
    kl2_paths = [p for p, (yt, _) in preds[names[0]].items() if yt == 2]
    n_kl2 = len(kl2_paths)
    print(f"\n{'='*70}")
    print(f"  KL2 error analysis  ({n_kl2} true-KL2 test images, {len(names)} models)")
    print(f"{'='*70}")

    # Per-image: how many models got it right?
    n_correct_per_image = Counter()
    misclass_target = Counter()
    for p in kl2_paths:
        n_right = sum(1 for name in names if preds[name][p][1] == 2)
        n_correct_per_image[n_right] += 1
        for name in names:
            pred = preds[name][p][1]
            if pred != 2:
                misclass_target[pred] += 1

    print(f"\n  Distribution of 'how many of {len(names)} models got each KL2 image right':")
    for k in range(len(names) + 1):
        n = n_correct_per_image.get(k, 0)
        bar = "#" * (n * 40 // max(n_kl2, 1))
        print(f"    {k}/{len(names)} models correct: {n:4d}  {bar}")

    always_wrong = n_correct_per_image.get(0, 0)
    always_right = n_correct_per_image.get(len(names), 0)
    print(f"\n  Always wrong (0/{len(names)} models): {always_wrong} ({100*always_wrong/n_kl2:.1f}%)"
          f"  <- candidates for label noise / genuinely ambiguous KL1-2-3 boundary")
    print(f"  Always right ({len(names)}/{len(names)} models): {always_right} ({100*always_right/n_kl2:.1f}%)")

    print(f"\n  When a model misses KL2, what does it predict instead?")
    total_misclass = sum(misclass_target.values())
    for grade in sorted(misclass_target):
        n = misclass_target[grade]
        print(f"    Predicted KL{grade}: {n:4d}  ({100*n/total_misclass:.1f}%)")
    adjacent = misclass_target.get(1, 0) + misclass_target.get(3, 0)
    print(f"\n  {100*adjacent/total_misclass:.1f}% of KL2 misclassifications land on an ADJACENT grade"
          f" (KL1 or KL3) — blended across all {len(names)} models, including the much")
    print(f"  weaker RF ones, so this blended number is misleading. Split by model quality:")

    groups = {
        "DL models (vit/resnet/efficientnet/hybrid)": [n for n in names if n not in ("rf", "rf_landmarks")],
        "RF models (rf/rf_landmarks)": [n for n in names if n in ("rf", "rf_landmarks")],
    }
    for group_name, group in groups.items():
        if not group:
            continue
        gt = Counter()
        for p in kl2_paths:
            for name in group:
                pred = preds[name][p][1]
                if pred != 2:
                    gt[pred] += 1
        gtotal = sum(gt.values())
        if gtotal == 0:
            continue
        adj = gt.get(1, 0) + gt.get(3, 0)
        dist = gt.get(0, 0) + gt.get(4, 0)
        print(f"\n    {group_name} (n_miss={gtotal}):")
        for g in sorted(gt):
            print(f"      -> KL{g}: {gt[g]:4d} ({100*gt[g]/gtotal:.1f}%)")
        print(f"      Adjacent (KL1/KL3): {100*adj/gtotal:.1f}%   Distant (KL0/KL4): {100*dist/gtotal:.1f}%")

    print(f"\n  Interpretation: the accurate DL models fail almost entirely on ADJACENT")
    print(f"  grades (mild<->moderate boundary ambiguity, a genuine ordinal-labeling")
    print(f"  problem) — the weak RF models instead collapse toward the majority class")
    print(f"  KL0, a feature-quality problem, not a labeling-ambiguity one.\n")

    # Pairwise agreement on KL2 specifically
    print(f"  Pairwise agreement on KL2 images (do models fail on the same images?):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            both_right = sum(1 for p in kl2_paths if preds[a][p][1] == 2 and preds[b][p][1] == 2)
            both_wrong = sum(1 for p in kl2_paths if preds[a][p][1] != 2 and preds[b][p][1] != 2)
            agree = both_right + both_wrong
            print(f"    {a:<14} & {b:<14} agree on {agree}/{n_kl2} ({100*agree/n_kl2:.0f}%)"
                  f"  [{both_right} both-right, {both_wrong} both-wrong]")


if __name__ == "__main__":
    main()
