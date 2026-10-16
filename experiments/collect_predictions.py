"""
Collect per-image test-set predictions for every trained model.

The saved *_results.json files only hold aggregate metrics — bootstrap
confidence intervals, ensembling, and cross-model error analysis all need
per-image predictions, which this script dumps to results/predictions/*.csv.

vit/resnet/efficientnet/hybrid need the trained .pth checkpoints;
rf/rf_landmarks need the trained .pkl models.

Usage:
    python experiments/collect_predictions.py --models vit,resnet,efficientnet,hybrid
    python experiments/collect_predictions.py --models rf,rf_landmarks
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import DATA_ROOT, get_transforms
from modern.model import build_vit
from modern.resnet import build_resnet50
from modern.efficientnet import build_efficientnet_b4
from modern.hybrid import HybridModel, HybridDataset
from modern.train import get_device
from baseline.train import load_split
from baseline.features import extract_features
from baseline.landmark_features import extract_landmark_features

RESULTS_DIR = Path(__file__).parent.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "vit":          RESULTS_DIR / "vit_best_v2.pth",
    "resnet":       RESULTS_DIR / "resnet_best.pth",
    "efficientnet": RESULTS_DIR / "efficientnet_best.pth",
    "hybrid":       RESULTS_DIR / "hybrid_best.pth",
    "hybrid_finetuned": RESULTS_DIR / "hybrid_finetuned_best.pth",
    "rf":           RESULTS_DIR / "baseline_rf_model.pkl",
    "rf_landmarks": RESULTS_DIR / "rf_with_landmarks.pkl",
}


def _relative_key(p):
    """Portable join key across machines: 'test/<grade>/<filename>' instead of
    a machine-specific absolute path."""
    parts = Path(p).parts
    idx = parts.index("dataset")
    return "/".join(parts[idx + 1:])


def _save(paths, y_true, y_pred, y_proba, name):
    out = PRED_DIR / f"{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_pred", "p0", "p1", "p2", "p3", "p4"])
        for p, yt, yp, proba in zip(paths, y_true, y_pred, y_proba):
            w.writerow([_relative_key(p), int(yt), int(yp), *[float(x) for x in proba]])
    print(f"  Saved {len(paths)} predictions to {out}")


def collect_torch_model(name, checkpoint_path, build_fn):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_fn(num_classes=5)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ds = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=get_transforms("test"))
    paths = [p for p, _ in ds.samples]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"  {name}"):
            logits = model(imgs.to(device))
            y_proba.append(F.softmax(logits, dim=1).cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    _save(paths, y_true, y_pred, np.vstack(y_proba), name)


def collect_hybrid(checkpoint_path, name="hybrid", unfreeze_vit=False):
    device = get_device()
    model = HybridModel(str(RESULTS_DIR / "vit_best_v2.pth"), unfreeze_vit=unfreeze_vit).to(device)
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_trainable_state_dict(best["trainable_state"])
    model.eval()

    test_ds = HybridDataset("test", quick=False)
    paths = [test_ds.base.samples[i][0] for i in test_ds.indices]
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for imgs, lm, labels in tqdm(loader, desc=f"  {name}"):
            logits = model(imgs.to(device), lm.to(device))
            y_proba.append(F.softmax(logits, dim=1).cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    _save(paths, y_true, y_pred, np.vstack(y_proba), name)


def collect_rf(name, model_path, feature_fn):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    rf, scaler = bundle["model"], bundle["scaler"]

    paths, labels = load_split(DATA_ROOT / "test", quick=False)
    X = np.vstack([feature_fn(p) for p in tqdm(paths, desc=f"  {name} features")])
    X = scaler.transform(X)
    proba_raw = rf.predict_proba(X)
    y_pred = rf.predict(X)

    full_proba = np.zeros((len(paths), 5))
    for i, c in enumerate(rf.classes_):
        full_proba[:, c] = proba_raw[:, i]
    _save(paths, labels, y_pred, full_proba, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str,
                        default="vit,resnet,efficientnet,hybrid,hybrid_finetuned,rf,rf_landmarks")
    args = parser.parse_args()

    for name in [n.strip() for n in args.models.split(",")]:
        ckpt = CHECKPOINTS.get(name)
        if ckpt is None:
            print(f"  Skipping unknown model '{name}'")
            continue
        if not ckpt.exists():
            print(f"  Skipping '{name}': checkpoint not found at {ckpt}")
            continue

        print(f"\n  Collecting predictions: {name}")
        if name == "vit":
            collect_torch_model("vit", ckpt, build_vit)
        elif name == "resnet":
            collect_torch_model("resnet", ckpt, build_resnet50)
        elif name == "efficientnet":
            collect_torch_model("efficientnet", ckpt, build_efficientnet_b4)
        elif name == "hybrid":
            collect_hybrid(ckpt, name="hybrid", unfreeze_vit=False)
        elif name == "hybrid_finetuned":
            collect_hybrid(ckpt, name="hybrid_finetuned", unfreeze_vit=True)
        elif name == "rf":
            collect_rf("rf", ckpt, extract_features)
        elif name == "rf_landmarks":
            collect_rf("rf_landmarks", ckpt, extract_landmark_features)


if __name__ == "__main__":
    main()
