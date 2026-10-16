"""
Phase 2C — Hybrid model: frozen ViT-B/16 backbone (Phase 1A weights) +
landmark-guided features (Phase 2A/2B), fused through a small trainable head.

    X-ray image                         landmark-guided features (80-dim)
         |                                          |
   ViT-B/16 backbone (frozen)              Linear(80->64) -> ReLU
   768-dim CLS token                                |
         |                                          |
   Linear(768->256) -> ReLU                         |
          \\_______________  concat [256+64=320]  __/
                            |
                      Dropout(0.3)
                      Linear(320->128) -> ReLU
                      Linear(128->5)

Usage:
    python modern/hybrid.py
    python modern/hybrid.py --quick      # 20% of data, fast sanity-check
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import get_transforms, DATA_ROOT
from modern.model import build_vit
from modern.train import get_device
from baseline.landmark_features import extract_landmark_features
from evaluation.metrics import full_evaluation_report

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_LANDMARK_FEATS = 80  # baseline/landmark_features.py output size


# ── Dataset: image + precomputed landmark-guided feature vector ─────────────

class HybridDataset(Dataset):
    def __init__(self, split: str, quick: bool = False):
        split_dir = DATA_ROOT / split
        self.base = datasets.ImageFolder(str(split_dir), transform=get_transforms(split))

        if quick:
            import random
            random.seed(SEED)
            by_class = {}
            for idx, (_, label) in enumerate(self.base.samples):
                by_class.setdefault(label, []).append(idx)
            keep = []
            for idxs in by_class.values():
                k = max(1, len(idxs) // 5)
                keep.extend(random.sample(idxs, k))
            self.indices = sorted(keep)
        else:
            self.indices = list(range(len(self.base.samples)))

        self.landmark_feats = []
        for i in tqdm(self.indices, desc=f"  {split} landmark features"):
            path, _ = self.base.samples[i]
            try:
                self.landmark_feats.append(extract_landmark_features(path))
            except Exception as e:
                print(f"\n  Warning: landmark extraction failed for {path} — {e}")
                self.landmark_feats.append(np.zeros(N_LANDMARK_FEATS, dtype=np.float32))

        self.targets = [self.base.samples[i][1] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        lm = torch.from_numpy(self.landmark_feats[i]).float()
        return img, lm, label


def make_weighted_sampler(dataset: HybridDataset):
    counts = Counter(dataset.targets)
    weights = [1.0 / counts[l] for l in dataset.targets]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.float),
                                 num_samples=len(weights), replacement=True)


def class_weights_from_targets(targets, n_classes=5, device=None):
    counts = Counter(targets)
    total = sum(counts.values())
    w = torch.tensor([total / (n_classes * counts.get(c, 1)) for c in range(n_classes)],
                     dtype=torch.float)
    return w.to(device) if device else w


# ── Model ─────────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):
    def __init__(self, vit_checkpoint, n_landmark_feats=N_LANDMARK_FEATS, num_classes=5,
                 unfreeze_vit=False):
        super().__init__()
        self.unfreeze_vit = unfreeze_vit
        self.vit = build_vit(num_classes=5)
        ckpt = torch.load(vit_checkpoint, map_location="cpu", weights_only=False)
        self.vit.load_state_dict(ckpt["model_state"])
        self.vit.heads = nn.Identity()  # -> 768-dim CLS token feature
        for p in self.vit.parameters():
            p.requires_grad = unfreeze_vit

        self.vit_proj = nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        self.landmark_branch = nn.Sequential(nn.Linear(n_landmark_feats, 64), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.unfreeze_vit:
            self.vit.eval()  # frozen backbone always stays in eval mode
        return self

    def forward(self, img, landmark_feat):
        vit_feat = self.vit_proj(self.vit(img))
        lm_feat = self.landmark_branch(landmark_feat)
        return self.fusion(torch.cat([vit_feat, lm_feat], dim=1))

    def trainable_state_dict(self):
        state = {
            "vit_proj": self.vit_proj.state_dict(),
            "landmark_branch": self.landmark_branch.state_dict(),
            "fusion": self.fusion.state_dict(),
        }
        if self.unfreeze_vit:
            state["vit"] = self.vit.state_dict()  # backbone weights changed, must persist too
        return state

    def load_trainable_state_dict(self, state):
        self.vit_proj.load_state_dict(state["vit_proj"])
        self.landmark_branch.load_state_dict(state["landmark_branch"])
        self.fusion.load_state_dict(state["fusion"])
        if "vit" in state:
            self.vit.load_state_dict(state["vit"])


# ── Train / eval epoch ───────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for imgs, lm_feats, labels in tqdm(loader, desc="  train" if train else "  val  ", leave=False):
            imgs, lm_feats, labels = imgs.to(device), lm_feats.to(device), labels.to(device)

            use_amp = device.type == "cuda"
            with autocast("cuda", enabled=use_amp):
                logits = model(imgs, lm_feats)
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
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic", labels=list(range(5)))
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss / n, kappa, acc


def evaluate_on_test(model, device, quick: bool, phase_name: str = "Hybrid"):
    test_ds = HybridDataset("test", quick=quick)
    num_workers = 0 if device.type == "mps" else 4
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)

    model.eval()
    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for imgs, lm_feats, labels in tqdm(test_loader, desc="  Evaluating"):
            imgs, lm_feats = imgs.to(device), lm_feats.to(device)
            logits = model(imgs, lm_feats)
            proba = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_proba.append(proba)

    all_proba = np.vstack(all_proba)
    oa_score = all_proba[:, 2:].sum(axis=1)
    return full_evaluation_report(
        all_labels, all_preds, oa_score, phase_name=phase_name, save_dir=str(RESULTS_DIR)
    )


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


# ── Grad-CAM for the ViT branch of the hybrid model ──────────────────────────

def hybrid_gradcam_overlay(image_path, model, device):
    """Grad-CAM on model.vit's last block, w.r.t. the *fused* hybrid logits —
    shows what the ViT branch contributes to the final hybrid decision."""
    img_orig = Image.open(image_path).convert("RGB").resize((224, 224))
    img_tensor = get_transforms("test")(img_orig).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)  # backbone weights are frozen; the input
                                       # tensor must require grad so autograd
                                       # still builds a graph through it.
    lm = extract_landmark_features(image_path)
    lm_tensor = torch.from_numpy(lm).float().unsqueeze(0).to(device)

    target_layer = model.vit.encoder.layers[-1].ln_1
    gradients, activations = [], []
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    model.eval()
    logits = model(img_tensor, lm_tensor)
    proba = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred = int(logits.argmax(1).item())

    model.zero_grad()
    logits[0, pred].backward()
    fh.remove()
    bh.remove()

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    weights = grad.mean(dim=-1)
    cam = (weights.unsqueeze(-1) * act).sum(dim=-1)[1:]  # drop CLS token
    cam = F.relu(cam)
    side = int(cam.shape[0] ** 0.5)
    cam = cam.reshape(side, side).detach().cpu().float().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    heatmap = (cm.jet(np.array(cam_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.6 * np.array(img_orig) + 0.4 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay), pred, proba


def save_hybrid_gradcam_samples(model, device, n_per_grade=2, max_scan=60, out_dirname="hybrid_gradcam"):
    out_dir = RESULTS_DIR / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = DATA_ROOT / "test"
    print(f"\n  Saving hybrid Grad-CAM samples (ViT branch) to {out_dir} ...")
    for grade in range(5):
        class_dir = test_dir / str(grade)
        if not class_dir.exists():
            continue
        imgs = (sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")))[:max_scan]
        saved = 0
        for img_path in imgs:
            if saved >= n_per_grade:
                break
            try:
                overlay, pred, _ = hybrid_gradcam_overlay(img_path, model, device)
                if pred == grade:
                    orig = Image.open(img_path).convert("RGB").resize((224, 224))
                    canvas = Image.new("RGB", (448, 224))
                    canvas.paste(orig, (0, 0))
                    canvas.paste(overlay, (224, 0))
                    canvas.save(out_dir / f"KL{grade}_{saved}_{img_path.stem}.png")
                    saved += 1
            except Exception as e:
                print(f"    Warning: {img_path.name} — {e}")
    print("  Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="LR for the ViT backbone when --unfreeze_vit is set "
                             "(kept lower than --lr since it starts pretrained, not from scratch)")
    parser.add_argument("--unfreeze_vit", action="store_true",
                        help="Fine-tune the ViT backbone jointly instead of keeping it frozen "
                             "(ablation: does joint training beat the frozen-backbone hybrid?)")
    parser.add_argument("--vit_checkpoint", type=str, default=str(RESULTS_DIR / "vit_best_v2.pth"))
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--log_name", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_name is None:
        args.checkpoint_name = "hybrid_finetuned_best.pth" if args.unfreeze_vit else "hybrid_best.pth"
    if args.log_name is None:
        args.log_name = "training_log_hybrid_finetuned.csv" if args.unfreeze_vit else "training_log_hybrid.csv"

    if not Path(args.vit_checkpoint).exists():
        print(f"ERROR: ViT checkpoint not found at {args.vit_checkpoint}")
        print("  Run modern/train.py first (Phase 1A).")
        sys.exit(1)

    device = get_device()
    print(f"\n{'='*55}")
    print("  Phase 2C — Hybrid (frozen ViT + landmark features)")
    print(f"  Device        : {device}")
    print(f"  Quick mode    : {args.quick}")
    print(f"  Epochs        : {args.epochs}  |  Patience: {args.patience}")
    print(f"  LR            : {args.lr}"
          + (f"  |  LR backbone: {args.lr_backbone}" if args.unfreeze_vit else ""))
    print(f"  Unfreeze ViT  : {args.unfreeze_vit}")
    print(f"  ViT checkpoint: {args.vit_checkpoint}")
    print(f"{'='*55}")

    train_ds = HybridDataset("train", quick=args.quick)
    val_ds = HybridDataset("val", quick=args.quick)

    num_workers = 0 if device.type == "mps" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=make_weighted_sampler(train_ds),
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = HybridModel(args.vit_checkpoint, unfreeze_vit=args.unfreeze_vit).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  Trainable params: {trainable:,}  (frozen: {frozen:,})")

    cw = class_weights_from_targets(train_ds.targets, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    if args.unfreeze_vit:
        optimizer = torch.optim.AdamW([
            {"params": model.vit.parameters(), "lr": args.lr_backbone},
            {"params": list(model.vit_proj.parameters()) + list(model.landmark_branch.parameters())
                       + list(model.fusion.parameters()), "lr": args.lr},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.lr, weight_decay=0.01)
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

            writer.writerow([epoch, tr_loss, tr_kappa, tr_acc, va_loss, va_kappa, va_acc, lr])
            f.flush()

            if va_kappa > best_kappa:
                best_kappa, best_epoch, patience_counter = va_kappa, epoch, 0
                torch.save({
                    "epoch": epoch,
                    "trainable_state": model.trainable_state_dict(),
                    "vit_checkpoint": args.vit_checkpoint,
                    "val_kappa": va_kappa,
                    "val_acc": va_acc,
                }, checkpoint_path)
                print(f"    New best val kappa={best_kappa:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(best kappa {best_kappa:.4f} at epoch {best_epoch})")
                    break

    print(f"\n  Training complete. Best val kappa: {best_kappa:.4f} (epoch {best_epoch})")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Log        : {log_path}")

    curves_name = "training_curves_hybrid_finetuned.png" if args.unfreeze_vit else "training_curves_hybrid.png"
    _plot_training_curves(log_path, curves_name)

    print("\n  Loading best checkpoint for test-set evaluation...")
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_trainable_state_dict(best["trainable_state"])
    model.to(device)
    result_phase_name = "HybridFinetuned" if args.unfreeze_vit else "Hybrid"
    evaluate_on_test(model, device, quick=args.quick, phase_name=result_phase_name)

    gradcam_dirname = "hybrid_finetuned_gradcam" if args.unfreeze_vit else "hybrid_gradcam"
    save_hybrid_gradcam_samples(model, device, out_dirname=gradcam_dirname)


if __name__ == "__main__":
    main()
