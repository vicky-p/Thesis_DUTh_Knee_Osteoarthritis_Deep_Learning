"""
Phase 1 — Evaluate Random Forest on the test split.

Usage:
    python baseline/evaluate.py
    python baseline/evaluate.py --quick
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline.features import extract_features
from baseline.train import load_split, print_class_distribution
from evaluation.metrics import full_evaluation_report

DATA_ROOT = Path(__file__).parent.parent / "data" / "dataset"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    model_path = RESULTS_DIR / "baseline_rf_model.pkl"
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("  Run baseline/train.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    rf = bundle["model"]
    scaler = bundle["scaler"]

    test_dir = DATA_ROOT / "test"
    if not test_dir.exists():
        print(f"ERROR: test split not found at {test_dir}")
        sys.exit(1)

    test_paths, test_labels = load_split(test_dir, quick=args.quick)
    print_class_distribution(test_labels, "Test")

    # Feature extraction
    features, failed = [], 0
    valid_labels = []
    for p, lbl in tqdm(zip(test_paths, test_labels), total=len(test_paths), desc="Test features"):
        try:
            features.append(extract_features(p))
            valid_labels.append(lbl)
        except Exception as e:
            print(f"\n  Warning: skipping {p.name} — {e}")
            failed += 1

    if not features:
        print("ERROR: no features extracted from test set.")
        sys.exit(1)

    X_raw = np.vstack(features)
    y_true = np.array(valid_labels)
    X = scaler.transform(X_raw)

    y_pred = rf.predict(X)
    y_proba = rf.predict_proba(X)

    # Binary OA score: probability that KL >= 2
    # Classes are ordered 0,1,2,3,4
    classes = list(rf.classes_)
    oa_score = np.array([
        sum(y_proba[i, classes.index(c)] for c in [2, 3, 4] if c in classes)
        for i in range(len(y_pred))
    ])

    full_evaluation_report(
        y_true, y_pred, oa_score,
        phase_name="Phase 1 RF",
        save_dir=str(RESULTS_DIR),
    )


if __name__ == "__main__":
    main()
