"""
ViT-B/16 pretrained on ImageNet with a custom 5-class classification head.
"""
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit(num_classes: int = 5, dropout_head: float = 0.3) -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Sequential(
        nn.Dropout(dropout_head),
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def set_frozen(model: nn.Module, frozen_epoch: bool):
    """
    frozen_epoch=True  → freeze all encoder layers except last 4 transformer blocks + head
    frozen_epoch=False → unfreeze everything
    """
    for param in model.parameters():
        param.requires_grad = not frozen_epoch

    if frozen_epoch:
        # Always train the head
        for param in model.heads.parameters():
            param.requires_grad = True
        # Unfreeze last 4 transformer blocks
        n_blocks = len(model.encoder.layers)
        for block in model.encoder.layers[n_blocks - 4:]:
            for param in block.parameters():
                param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
