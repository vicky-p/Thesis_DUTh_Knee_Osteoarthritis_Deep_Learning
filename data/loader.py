"""
PyTorch DataLoader for the Knee OA dataset.
Handles augmentation, normalization, and class-imbalance via WeightedRandomSampler.
"""
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DATA_ROOT = Path(__file__).parent / "dataset"


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def make_weighted_sampler(dataset):
    labels = [dataset.targets[i] for i in range(len(dataset))]
    counts = Counter(labels)
    n_classes = len(counts)
    class_weights = {c: 1.0 / counts[c] for c in range(n_classes)}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_loader(split: str, batch_size: int = 32, num_workers: int = 4,
               quick: bool = False, data_root: Path = DATA_ROOT) -> DataLoader:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dataset = datasets.ImageFolder(str(split_dir), transform=get_transforms(split))

    if quick:
        # Deterministic 20% stratified subset
        import random
        random.seed(42)
        indices_by_class = {}
        for idx, (_, label) in enumerate(dataset.samples):
            indices_by_class.setdefault(label, []).append(idx)
        keep = []
        for cls_indices in indices_by_class.values():
            k = max(1, len(cls_indices) // 5)
            keep.extend(random.sample(cls_indices, k))
        from torch.utils.data import Subset
        dataset = Subset(dataset, keep)
        # Rebuild targets for sampler on subset
        full_targets = [dataset.dataset.targets[i] for i in keep]
        counts = Counter(full_targets)
        class_weights = {c: 1.0 / counts[c] for c in counts}
        sample_weights = [class_weights[l] for l in full_targets]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(sample_weights),
            replacement=True,
        ) if split == "train" else None
    else:
        sampler = make_weighted_sampler(dataset) if split == "train" else None

    shuffle = (split == "train") and (sampler is None)
    pin = torch.cuda.is_available()  # pin_memory unsupported on MPS
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )


def print_dataset_stats(data_root: Path = DATA_ROOT):
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        ds = datasets.ImageFolder(str(split_dir))
        counts = Counter(ds.targets)
        total = len(ds)
        print(f"\n  {split} ({total} images):")
        for g in range(5):
            n = counts.get(g, 0)
            bar = "█" * (n * 30 // max(counts.values(), default=1))
            print(f"    KL{g}: {n:5d}  {bar}")
