"""
Grad-CAM visualization for ResNet-50 — classic spatial Grad-CAM (Selvaraju et
al.) on the last conv block, target_layer = model.layer4[-1]. Complements
modern/gradcam.py (ViT, attention-token-based) so the two architectures'
Grad-CAMs can be compared directly for the Phase 1C ablation study.

For each KL grade (0-4):
  - 3 correctly classified images + 1 misclassified image
  - Each saved as side-by-side: original | heatmap overlay
  - One combined 5x4 summary grid -> results/resnet_gradcam_summary.png

Usage:
    python modern/resnet_gradcam.py
    python modern/resnet_gradcam.py --checkpoint results/resnet_best.pth
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD
from modern.resnet import build_resnet50
from modern.train import get_device

RESULTS_DIR = Path(__file__).parent.parent / "results"
GRADCAM_DIR = RESULTS_DIR / "resnet_gradcam"

CELL_SIZE = 112

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _gradcam_overlay(image_path: Path, model, device) -> tuple:
    img_orig = Image.open(image_path).convert("RGB").resize((224, 224))
    img_tensor = PREPROCESS(img_orig).unsqueeze(0).to(device)

    target_layer = model.layer4[-1]  # last spatial feature map before avgpool
    activations, gradients = [], []
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    model.eval()
    logits = model(img_tensor)
    proba = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred = int(logits.argmax(1).item())

    model.zero_grad()
    logits[0, pred].backward()
    fh.remove()
    bh.remove()

    act = activations[0].squeeze(0)   # [C, H, W]
    grad = gradients[0].squeeze(0)    # [C, H, W]
    weights = grad.mean(dim=(1, 2))   # global-average-pool gradients -> per-channel weight
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = F.relu(cam)
    cam = cam.detach().cpu().float().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    import matplotlib.cm as cm
    heatmap = (cm.jet(np.array(cam_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    orig_np = np.array(img_orig)
    overlay = (0.6 * orig_np + 0.4 * heatmap).astype(np.uint8)

    return Image.fromarray(overlay), pred, proba


def make_sidebyside(orig_path: Path, overlay: Image.Image, size=224) -> Image.Image:
    orig = Image.open(orig_path).convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size * 2, size))
    canvas.paste(orig, (0, 0))
    canvas.paste(overlay, (size, 0))
    return canvas


def collect_samples(model, device, test_dir: Path, n_correct=3, n_wrong=1, max_scan=200):
    samples = {g: {"correct": [], "wrong": []} for g in range(5)}
    for grade in range(5):
        class_dir = test_dir / str(grade)
        if not class_dir.exists():
            continue
        imgs = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))[:max_scan]
        c_buf, w_buf = [], []
        for img_path in tqdm(imgs, desc=f"  KL{grade}", leave=False):
            if len(c_buf) >= n_correct and len(w_buf) >= n_wrong:
                break
            try:
                overlay, pred, proba = _gradcam_overlay(img_path, model, device)
                entry = (img_path, overlay, pred, proba)
                if pred == grade and len(c_buf) < n_correct:
                    c_buf.append(entry)
                elif pred != grade and len(w_buf) < n_wrong:
                    w_buf.append(entry)
            except Exception as e:
                print(f"    Warning: {img_path.name} — {e}")
        samples[grade]["correct"] = c_buf
        samples[grade]["wrong"] = w_buf
        print(f"  KL{grade}: {len(c_buf)} correct, {len(w_buf)} misclassified "
              f"(scanned {min(len(imgs), max_scan)} images)")
    return samples


def save_individual(samples: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for grade, buckets in samples.items():
        for tag, entries in buckets.items():
            for idx, (path, overlay, pred, _) in enumerate(entries):
                panel = make_sidebyside(path, overlay, size=224)
                label = "correct" if tag == "correct" else f"pred{pred}"
                panel.save(out_dir / f"KL{grade}_{label}_{idx}_{path.stem}.png")


def build_summary_grid(samples: dict, save_path: Path, cell=CELL_SIZE):
    n_cols, n_rows = 4, 5
    panel_w, panel_h = cell * 2, cell
    canvas = np.ones((n_rows * panel_h, n_cols * panel_w, 3), dtype=np.uint8) * 240

    for row, grade in enumerate(range(5)):
        entries = samples[grade]["correct"] + samples[grade]["wrong"]
        for col in range(n_cols):
            y0, y1 = row * panel_h, (row + 1) * panel_h
            x0, x1 = col * panel_w, (col + 1) * panel_w
            if col < len(entries):
                path, overlay, pred, _ = entries[col]
                panel = make_sidebyside(path, overlay, size=cell)
                canvas[y0:y1, x0:x1] = np.array(panel)
                if col == n_cols - 1 and samples[grade]["wrong"]:
                    canvas[y0:y0+3, x0:x1] = [220, 50, 50]
                    canvas[y1-3:y1, x0:x1] = [220, 50, 50]
                    canvas[y0:y1, x0:x0+3] = [220, 50, 50]
                    canvas[y0:y1, x1-3:x1] = [220, 50, 50]

    fig, ax = plt.subplots(figsize=(n_cols * panel_w / 80, n_rows * panel_h / 80 + 1.2))
    ax.imshow(canvas)
    ax.axis("off")
    for row in range(n_rows):
        ax.text(-10, row * panel_h + panel_h // 2, f"KL{row}", va="center", ha="right",
                fontsize=10, fontweight="bold", transform=ax.transData)
    for col in range(n_cols):
        label = f"Correct {col+1}" if col < 3 else "Misclassified"
        color = "black" if col < 3 else "#dc3232"
        ax.text(col * panel_w + panel_w // 2, -8, label, ha="center", va="bottom",
                fontsize=9, color=color, transform=ax.transData)
    for col in range(n_cols):
        for sub, lbl in enumerate(["orig", "Grad-CAM"]):
            ax.text(col * panel_w + sub * cell + cell // 2, -2, lbl, ha="center", va="bottom",
                    fontsize=7, color="#555", transform=ax.transData)

    correct_patch = mpatches.Patch(color="none", label="White border = correct")
    wrong_patch = mpatches.Patch(color="#dc3232", label="Red border = misclassified")
    ax.legend(handles=[correct_patch, wrong_patch], loc="lower right", fontsize=8,
              framealpha=0.7, bbox_to_anchor=(1, -0.02))
    ax.set_title("Grad-CAM: ResNet-50 (layer4) — 3 correct + 1 misclassified per KL grade",
                 fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary grid saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(RESULTS_DIR / "resnet_best.pth"))
    parser.add_argument("--n_correct", type=int, default=3)
    parser.add_argument("--n_wrong", type=int, default=1)
    parser.add_argument("--max_scan", type=int, default=200)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_resnet50(num_classes=5)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"val_kappa={ckpt.get('val_kappa',0):.4f})")

    test_dir = DATA_ROOT / "test"
    print(f"\n  Collecting {args.n_correct} correct + {args.n_wrong} misclassified per grade...")
    samples = collect_samples(model, device, test_dir,
                              n_correct=args.n_correct, n_wrong=args.n_wrong,
                              max_scan=args.max_scan)

    print(f"\n  Saving individual side-by-side images to {GRADCAM_DIR} ...")
    save_individual(samples, GRADCAM_DIR)

    summary_path = RESULTS_DIR / "resnet_gradcam_summary.png"
    print(f"\n  Building 5x4 summary grid ...")
    build_summary_grid(samples, summary_path)

    total = sum(len(v["correct"]) + len(v["wrong"]) for v in samples.values())
    print(f"\n  Done — {total} individual images + summary grid saved.\n")


if __name__ == "__main__":
    main()
