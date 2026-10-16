"""
Bootstrap confidence intervals and pairwise significance testing on the test
set, using the per-image predictions from experiments/collect_predictions.py.

All aggregate metrics reported so far (compare.py) are point estimates on a
single 1656-image test set — this quantifies how much they could plausibly
move under resampling, and whether "close" results (e.g. ViT vs ResNet-50)
are actually distinguishable or within noise.

Usage:
    python experiments/statistical_significance.py
    python experiments/statistical_significance.py --n_boot 2000
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"

SEED = 42


def load_predictions(name):
    path = PRED_DIR / f"{name}.csv"
    y_true, y_pred, proba = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
            proba.append([float(row[f"p{i}"]) for i in range(5)])
    return np.array(y_true), np.array(y_pred), np.array(proba)


def weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(5)))


def binary_oa_auc(y_true, proba):
    binary_true = (y_true >= 2).astype(int)
    oa_score = proba[:, 2:].sum(axis=1)
    if len(np.unique(binary_true)) < 2:
        return float("nan")
    return roc_auc_score(binary_true, oa_score)


def bootstrap_metric(y_true, y_pred, proba, metric_fn, n_boot, rng):
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = metric_fn(y_true[idx], y_pred[idx], proba[idx])
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--models", type=str,
                        default="rf,rf_landmarks,vit,resnet,efficientnet,hybrid,hybrid_finetuned")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    rng = np.random.default_rng(SEED)
    data = {}
    for name in models:
        path = PRED_DIR / f"{name}.csv"
        if not path.exists():
            print(f"  Skipping '{name}': {path} not found")
            continue
        data[name] = load_predictions(name)

    print(f"\n{'='*70}")
    print(f"  Bootstrap 95% CI  (n_boot={args.n_boot}, resampling the 1656-image test set)")
    print(f"{'='*70}")
    print(f"  {'Model':<16}{'AUC (95% CI)':<26}{'kw (95% CI)':<26}")

    boot_cache = {}
    for name, (y_true, y_pred, proba) in data.items():
        auc_boot = bootstrap_metric(y_true, y_pred, proba,
                                    lambda yt, yp, pr: binary_oa_auc(yt, pr),
                                    args.n_boot, rng)
        kw_boot = bootstrap_metric(y_true, y_pred, proba,
                                   lambda yt, yp, pr: weighted_kappa(yt, yp),
                                   args.n_boot, rng)
        boot_cache[name] = (auc_boot, kw_boot)

        auc_lo, auc_hi = np.percentile(auc_boot, [2.5, 97.5])
        kw_lo, kw_hi = np.percentile(kw_boot, [2.5, 97.5])
        auc_point = binary_oa_auc(y_true, proba)
        kw_point = weighted_kappa(y_true, y_pred)
        print(f"  {name:<16}{auc_point:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]      "
              f"{kw_point:.3f} [{kw_lo:.3f}, {kw_hi:.3f}]")

    # Paired bootstrap significance: for each pair, use the SAME resampled
    # indices for both models (paired), so the comparison isn't confounded by
    # independent resampling noise.
    print(f"\n{'='*70}")
    print(f"  Paired bootstrap: P(AUC_A <= AUC_B) — is A really ahead of B?")
    print(f"{'='*70}")

    names = list(data.keys())
    n = len(data[names[0]][0])
    for i in range(len(names)):
        for j in range(len(names)):
            if i >= j:
                continue
            a, b = names[i], names[j]
            ya, yb = data[a], data[b]
            diffs = np.empty(args.n_boot)
            for k in range(args.n_boot):
                idx = rng.integers(0, n, n)
                diffs[k] = binary_oa_auc(ya[0][idx], ya[2][idx]) - binary_oa_auc(yb[0][idx], yb[2][idx])
            p_a_ahead = (diffs > 0).mean()
            direction = a if p_a_ahead >= 0.5 else b
            confidence = max(p_a_ahead, 1 - p_a_ahead)
            flag = "  <- not distinguishable (< 95%)" if confidence < 0.95 else "  <- distinguishable"
            print(f"  {a:<14} vs {b:<14}  P({direction} ahead on AUC) = {confidence:.3f}{flag}")

    print()


if __name__ == "__main__":
    main()
