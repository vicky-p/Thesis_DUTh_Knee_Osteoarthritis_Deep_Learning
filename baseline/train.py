"""
Phase 1 — Random Forest training on hand-crafted features.

Usage:
    python baseline/train.py
    python baseline/train.py --quick      # 20% of data, fast sanity-check
"""
import argparse
import os
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

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline.features import extract_features

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = Path(__file__).parent.parent / "data" / "dataset"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_split(split_dir, quick=False):
    """Load all images from a split directory. Returns (X, y, paths)."""
    split_dir = Path(split_dir)
    paths, labels = [], []
    for grade in range(5):
        class_dir = split_dir / str(grade)
        if not class_dir.exists():
            continue
        imgs = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        paths.extend(imgs)
        labels.extend([grade] * len(imgs))

    if quick:
        # deterministic 20% subset, stratified
        rng = np.random.default_rng(SEED)
        idx = []
        for g in range(5):
            cls_idx = [i for i, l in enumerate(labels) if l == g]
            k = max(1, len(cls_idx) // 5)
            idx.extend(rng.choice(cls_idx, k, replace=False).tolist())
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx]

    return paths, labels


def extract_all(paths, desc="Extracting features"):
    features = []
    failed = 0
    for p in tqdm(paths, desc=desc, unit="img"):
        try:
            features.append(extract_features(p))
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


def print_class_distribution(labels, split_name):
    from collections import Counter
    dist = Counter(labels)
    total = len(labels)
    print(f"\n  {split_name} class distribution ({total} images):")
    for g in range(5):
        n = dist.get(g, 0)
        bar = "█" * (n * 30 // max(dist.values(), default=1))
        print(f"    KL{g}: {n:5d}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 20%% of data")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Phase 1 — Baseline Random Forest Training")
    print(f"  Data root : {DATA_ROOT}")
    print(f"  Quick mode: {args.quick}")
    print(f"{'='*55}")

    train_dir = DATA_ROOT / "train"
    if not train_dir.exists():
        print(f"\nERROR: Dataset not found at {DATA_ROOT}")
        print("  Place the Kaggle dataset so that data/dataset/train/0,1,2,3,4 exist.")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    train_paths, train_labels = load_split(train_dir, quick=args.quick)
    print_class_distribution(train_labels, "Train")

    # ── Feature extraction ───────────────────────────────────────────────────
    t0 = time.time()
    X_raw, keep_idx, n_fail = extract_all(train_paths, desc="Train features")
    y = np.array([train_labels[i] for i in keep_idx])
    print(f"\n  Features extracted: {X_raw.shape[0]} images × {X_raw.shape[1]} features  "
          f"({time.time()-t0:.1f}s, {n_fail} failed)")

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── Cross-validation ─────────────────────────────────────────────────────
    print("\n  Running 5-fold stratified cross-validation …")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Final fit on full train set ───────────────────────────────────────────
    print("\n  Fitting final model on full training set …")
    rf.fit(X, y)

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = RESULTS_DIR / "baseline_rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf, "scaler": scaler}, f)
    print(f"  Model saved to {model_path}")
    print("\n  Done. Run baseline/evaluate.py next.\n")


if __name__ == "__main__":
    main()
