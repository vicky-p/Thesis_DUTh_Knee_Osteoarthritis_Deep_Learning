"""
Test-time augmentation: average predictions over a few augmented views of
each test image (original, horizontal flip, +-5deg rotation) instead of a
single forward pass. Pure inference-time change, no retraining.

Usage:
    python experiments/tta_eval.py --model vit
    python experiments/tta_eval.py --model resnet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD
from modern.model import build_vit
from modern.resnet import build_resnet50
from modern.efficientnet import build_efficientnet_b4
from modern.train import get_device
from evaluation.metrics import full_evaluation_report

RESULTS_DIR = Path(__file__).parent.parent / "results"

BUILDERS = {
    "vit": (build_vit, RESULTS_DIR / "vit_best_v2.pth"),
    "resnet": (build_resnet50, RESULTS_DIR / "resnet_best.pth"),
    "efficientnet": (build_efficientnet_b4, RESULTS_DIR / "efficientnet_best.pth"),
}

TTA_VIEWS = [
    ("original", transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])),
    ("hflip", transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])),
    ("rot+5", transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((5, 5)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])),
    ("rot-5", transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((-5, -5)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(BUILDERS.keys()))
    args = parser.parse_args()

    build_fn, ckpt_path = BUILDERS[args.model]
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_fn(num_classes=5)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Loaded {args.model} checkpoint (epoch {ckpt.get('epoch','?')})")

    test_dir = DATA_ROOT / "test"
    base_ds = datasets.ImageFolder(str(test_dir))  # no transform: raw PIL images
    y_true = [label for _, label in base_ds.samples]

    all_view_proba = []
    for view_name, tf in TTA_VIEWS:
        print(f"\n  View: {view_name}")
        proba_list = []
        with torch.no_grad():
            batch_imgs = []
            for path, _ in tqdm(base_ds.samples, desc=f"  {view_name}"):
                from PIL import Image
                img = Image.open(path).convert("RGB")
                batch_imgs.append(tf(img))
                if len(batch_imgs) == 32:
                    x = torch.stack(batch_imgs).to(device)
                    proba_list.append(F.softmax(model(x), dim=1).cpu().numpy())
                    batch_imgs = []
            if batch_imgs:
                x = torch.stack(batch_imgs).to(device)
                proba_list.append(F.softmax(model(x), dim=1).cpu().numpy())
        all_view_proba.append(np.vstack(proba_list))

    avg_proba = np.mean(all_view_proba, axis=0)
    y_pred = avg_proba.argmax(axis=1)
    oa_score = avg_proba[:, 2:].sum(axis=1)

    phase_name = f"{args.model.capitalize()}_TTA"
    full_evaluation_report(y_true, y_pred, oa_score, phase_name=phase_name, save_dir=str(RESULTS_DIR))


if __name__ == "__main__":
    main()
