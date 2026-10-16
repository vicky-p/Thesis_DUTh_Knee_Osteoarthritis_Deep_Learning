"""
Phase 3 — EfficientNet-B4 ablation: fine-tune all layers from epoch 1 (no frozen phase).

Usage:
    python modern/efficientnet.py
    python modern/efficientnet.py --quick      # 20% of data, fast sanity-check
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import get_loader, print_dataset_stats, DATA_ROOT
from modern.model import count_trainable
from modern.train import get_device, class_weights_from_loader, run_epoch
from evaluation.metrics import full_evaluation_report

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_efficientnet_b4(num_classes: int = 5) -> nn.Module:
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1792, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def _plot_training_curves(log_path, out_name):
    import matplotlib.pyplot as plt
    epochs, tr_kw, va_kw, tr_loss, va_loss = [], [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_kw.append(float(row["train_kappa"]))
            va_kw.append(float(row["val_kappa"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, tr_loss, label="train"); ax1.plot(epochs, va_loss, label="val")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="CE Loss"); ax1.legend()
    ax2.plot(epochs, tr_kw, label="train"); ax2.plot(epochs, va_kw, label="val")
    ax2.set(title="Weighted Kappa", xlabel="Epoch", ylabel="kw"); ax2.legend()
    fig.tight_layout()
    out = Path(log_path).parent / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {out}")


def evaluate_on_test(model, device, quick: bool):
    num_workers = 0 if device.type == "mps" else 4
    test_loader = get_loader("test", batch_size=32, num_workers=num_workers, quick=quick)

    model.eval()
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
    return full_evaluation_report(
        all_labels, all_preds, oa_score,
        phase_name="Efficientnet",
        save_dir=str(RESULTS_DIR),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_name", type=str, default="efficientnet_best.pth")
    parser.add_argument("--log_name", type=str, default="training_log_efficientnet.csv")
    args = parser.parse_args()

    device = get_device()
    print(f"\n{'='*55}")
    print("  Phase 3 — EfficientNet-B4 Training (fine-tune all layers)")
    print(f"  Device      : {device}")
    print(f"  Quick mode  : {args.quick}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Batch size  : {args.batch_size}  |  Patience: {args.patience}")
    print(f"  Checkpoint  : {args.checkpoint_name}")
    print(f"{'='*55}")
    print_dataset_stats(DATA_ROOT)

    num_workers = 0 if device.type == "mps" else 4
    train_loader = get_loader("train", batch_size=args.batch_size,
                              num_workers=num_workers, quick=args.quick)
    val_loader   = get_loader("val",   batch_size=args.batch_size,
                              num_workers=num_workers, quick=args.quick)

    model = build_efficientnet_b4(num_classes=5).to(device)
    print(f"\n  Trainable params : {count_trainable(model):,}")

    cw = class_weights_from_loader(train_loader, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    log_path = RESULTS_DIR / args.log_name
    best_kappa, best_epoch, patience_counter = -1.0, 0, 0
    checkpoint_path = RESULTS_DIR / args.checkpoint_name

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_kappa", "train_acc",
                         "val_loss", "val_kappa", "val_acc", "lr"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_kappa, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer, scaler, device, train=True
            )
            va_loss, va_kappa, va_acc = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, train=False
            )
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(f"  Ep {epoch:3d}/{args.epochs}  "
                  f"tr_loss={tr_loss:.4f}  tr_kw={tr_kappa:.4f}  "
                  f"va_loss={va_loss:.4f}  va_kw={va_kappa:.4f}  "
                  f"va_acc={va_acc:.3f}  lr={lr:.2e}  ({elapsed:.0f}s)")

            writer.writerow([epoch, tr_loss, tr_kappa, tr_acc,
                             va_loss, va_kappa, va_acc, lr])
            f.flush()

            if va_kappa > best_kappa:
                best_kappa = va_kappa
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_kappa": va_kappa,
                    "val_acc": va_acc,
                }, checkpoint_path)
                print(f"    ✓ New best val kappa={best_kappa:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(best kappa {best_kappa:.4f} at epoch {best_epoch})")
                    break

    print(f"\n  Training complete. Best val kappa: {best_kappa:.4f} (epoch {best_epoch})")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Log        : {log_path}")

    _plot_training_curves(log_path, "training_curves_efficientnet.png")

    print("\n  Loading best checkpoint for test-set evaluation...")
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.to(device)
    evaluate_on_test(model, device, quick=args.quick)


if __name__ == "__main__":
    main()
