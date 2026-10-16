"""
Ensemble: average the softmax probabilities of two (or more) already-trained
models and re-evaluate. No retraining — pure inference-time combination on
the per-image predictions from experiments/collect_predictions.py.

ViT and ResNet-50 are statistically indistinguishable individually
(experiments/statistical_significance.py) but may make different mistakes;
averaging tests whether they're complementary.

Usage:
    python experiments/ensemble.py --models vit,resnet
    python experiments/ensemble.py --models vit,resnet,hybrid
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import full_evaluation_report

RESULTS_DIR = Path(__file__).parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"


def load_predictions(name):
    path = PRED_DIR / f"{name}.csv"
    paths, y_true, proba = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            y_true.append(int(row["y_true"]))
            proba.append([float(row[f"p{i}"]) for i in range(5)])
    return paths, np.array(y_true), np.array(proba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="vit,resnet")
    args = parser.parse_args()
    names = [m.strip() for m in args.models.split(",")]

    loaded = [load_predictions(n) for n in names]
    paths0, y_true0 = loaded[0][0], loaded[0][1]
    for paths, y_true, _ in loaded[1:]:
        assert paths == paths0, "prediction files cover different/misordered image sets"
        assert (y_true == y_true0).all(), "label mismatch between prediction files"

    avg_proba = np.mean([proba for _, _, proba in loaded], axis=0)
    y_pred = avg_proba.argmax(axis=1)

    phase_name = "Ensemble_" + "_".join(names)
    print(f"\n  Ensembling: {', '.join(names)}  (simple probability averaging)")
    full_evaluation_report(
        y_true0, y_pred, avg_proba[:, 2:].sum(axis=1),
        phase_name=phase_name, save_dir=str(RESULTS_DIR),
    )


if __name__ == "__main__":
    main()
