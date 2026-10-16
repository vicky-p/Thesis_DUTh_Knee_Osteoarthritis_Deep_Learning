"""
Phase 2B — Random Forest on landmark-guided features (Phase 1+).

Usage:
    python baseline/train_with_landmarks.py
    python baseline/train_with_landmarks.py --quick      # 20% of data, fast sanity-check
"""
import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline.landmark_features import extract_landmark_features
from baseline.train import load_split, print_class_distribution
from evaluation.metrics import full_evaluation_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = Path(__file__).parent.parent / "data" / "dataset"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def extract_all(paths, desc="Extracting features"):
    features, failed = [], 0
    for p in tqdm(paths, desc=desc, unit="img"):
        try:
            features.append(extract_landmark_features(p))
        except Exception as e:
            print(f"\n  Warning: skipping {p.name} — {e}")
            failed += 1
            features.append(None)
    valid = [(f, i) for i, f in enumerate(features) if f is not None]
    if not valid:
        raise RuntimeError("No features extracted.")
    X = np.vstack([v[0] for v in valid])
    keep = [v[1] for v in valid]
    return X, keep, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 20%% of data")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Phase 2B — Random Forest on Landmark-Guided Features")
    print(f"  Data root : {DATA_ROOT}")
    print(f"  Quick mode: {args.quick}")
    print(f"{'='*55}")

    train_dir = DATA_ROOT / "train"
    if not train_dir.exists():
        print(f"\nERROR: Dataset not found at {DATA_ROOT}")
        sys.exit(1)

    train_paths, train_labels = load_split(train_dir, quick=args.quick)
    print_class_distribution(train_labels, "Train")

    t0 = time.time()
    X_raw, keep_idx, n_fail = extract_all(train_paths, desc="Train landmark features")
    y = np.array([train_labels[i] for i in keep_idx])
    print(f"\n  Features extracted: {X_raw.shape[0]} images x {X_raw.shape[1]} features  "
          f"({time.time()-t0:.1f}s, {n_fail} failed)")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("\n  Running 5-fold stratified cross-validation ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("\n  Fitting final model on full training set ...")
    rf.fit(X, y)

    model_path = RESULTS_DIR / "rf_with_landmarks.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf, "scaler": scaler}, f)
    print(f"  Model saved to {model_path}")

    # ── Test-set evaluation (self-contained, matches Phase 1C run pattern) ──
    test_dir = DATA_ROOT / "test"
    test_paths, test_labels = load_split(test_dir, quick=args.quick)
    print_class_distribution(test_labels, "Test")

    X_test_raw, keep_idx, n_fail = extract_all(test_paths, desc="Test landmark features")
    y_true = np.array([test_labels[i] for i in keep_idx])
    X_test = scaler.transform(X_test_raw)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    classes = list(rf.classes_)
    oa_score = np.array([
        sum(y_proba[i, classes.index(c)] for c in [2, 3, 4] if c in classes)
        for i in range(len(y_pred))
    ])

    full_evaluation_report(
        y_true, y_pred, oa_score,
        phase_name="Phase1plus",
        save_dir=str(RESULTS_DIR),
    )


if __name__ == "__main__":
    main()
