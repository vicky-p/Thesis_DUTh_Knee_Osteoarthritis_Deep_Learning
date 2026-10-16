"""
Phase 2 — Evaluate ViT checkpoint on the test split.

Usage:
    python modern/evaluate.py
    python modern/evaluate.py --quick
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import get_loader, DATA_ROOT
from modern.model import build_vit
from evaluation.metrics import full_evaluation_report

RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default=str(RESULTS_DIR / "vit_best.pth"))
    parser.add_argument("--phase_name", type=str, default="Phase 2 ViT")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("  Run modern/train.py first.")
        sys.exit(1)

    device = get_device()
    print(f"\n  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_vit(num_classes=5)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Checkpoint epoch: {ckpt.get('epoch','?')}  "
          f"val_kappa: {ckpt.get('val_kappa', '?'):.4f}")

    num_workers = 0 if device.type == "mps" else 4
    test_loader = get_loader("test", batch_size=32, num_workers=num_workers,
                             quick=args.quick)

    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            proba = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_proba.append(proba)

    all_proba = np.vstack(all_proba)
    oa_score = all_proba[:, 2:].sum(axis=1)  # P(KL>=2)

    full_evaluation_report(
        all_labels, all_preds, oa_score,
        phase_name=args.phase_name,
        save_dir=str(RESULTS_DIR),
    )


if __name__ == "__main__":
    main()
