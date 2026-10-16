"""
Phase 2 — ViT-B/16 training loop.

Usage:
    python modern/train.py
    python modern/train.py --quick      # 20% of data, fast sanity-check
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import get_loader, print_dataset_stats, DATA_ROOT
from modern.model import build_vit, set_frozen, count_trainable

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights_from_loader(loader, n_classes=5, device=None):
    """Compute inverse-frequency class weights from a DataLoader."""
    from collections import Counter
    # For plain ImageFolder datasets, use .dataset.targets
    ds = loader.dataset
    if hasattr(ds, "targets"):
        counts = Counter(ds.targets)
    elif hasattr(ds, "dataset") and hasattr(ds.dataset, "targets"):
        # Subset wrapper
        counts = Counter(ds.dataset.targets[i] for i in ds.indices)
    else:
        return None
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)],
        dtype=torch.float,
    )
    if device:
        weights = weights.to(device)
    return weights


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="  train" if train else "  val  ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            use_amp = device.type == "cuda"
            with autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic",
                              labels=list(range(5)))
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return avg_loss, kappa, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_frozen", type=float, default=1e-4)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)
    parser.add_argument("--checkpoint_name", type=str, default="vit_best.pth")
    parser.add_argument("--log_name", type=str, default="training_log.csv")
    args = parser.parse_args()

    device = get_device()
    print(f"\n{'='*55}")
    print("  Phase 2 — ViT-B/16 Training")
    print(f"  Device      : {device}")
    print(f"  Quick mode  : {args.quick}")
    print(f"  Epochs      : {args.epochs}  (freeze={args.freeze_epochs})")
    print(f"  LR frozen   : {args.lr_frozen}  →  LR finetune: {args.lr_finetune}")
    print(f"  Batch size  : {args.batch_size}  |  Patience: {args.patience}")
    print(f"  Checkpoint  : {args.checkpoint_name}")
    print(f"{'='*55}")
    print_dataset_stats(DATA_ROOT)

    num_workers = 0 if device.type == "mps" else 4
    train_loader = get_loader("train", batch_size=args.batch_size,
                              num_workers=num_workers, quick=args.quick)
    val_loader   = get_loader("val",   batch_size=args.batch_size,
                              num_workers=num_workers, quick=args.quick)

    model = build_vit(num_classes=5).to(device)
    set_frozen(model, frozen_epoch=True)
    print(f"\n  Trainable params (frozen phase) : {count_trainable(model):,}")

    cw = class_weights_from_loader(train_loader, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_frozen, weight_decay=0.01,
    )
    finetune_epochs = max(1, args.epochs - args.freeze_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    log_path = RESULTS_DIR / args.log_name
    best_kappa, best_epoch, patience_counter = -1.0, 0, 0
    checkpoint_path = RESULTS_DIR / args.checkpoint_name

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_kappa", "train_acc",
                         "val_loss", "val_kappa", "val_acc", "lr"])

        for epoch in range(1, args.epochs + 1):
            # Switch from frozen to fine-tuning
            if epoch == args.freeze_epochs + 1:
                set_frozen(model, frozen_epoch=False)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr_finetune, weight_decay=0.01
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=finetune_epochs
                )
                print(f"\n  Epoch {epoch}: unfreezing all layers — lr={args.lr_finetune} "
                      f"({count_trainable(model):,} trainable params)")

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
    print("\n  Run modern/evaluate.py to get test-set results.\n")

    _plot_training_curves(log_path)


def _plot_training_curves(log_path):
    import csv
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
    out = Path(log_path).parent / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {out}")


if __name__ == "__main__":
    main()
