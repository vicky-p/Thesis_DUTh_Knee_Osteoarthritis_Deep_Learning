import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix
)


def compute_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_auc_binary(y_true, y_score):
    """Binary OA detection: KL>=2 vs KL<2. y_score is probability of KL>=2."""
    binary_true = (np.array(y_true) >= 2).astype(int)
    return roc_auc_score(binary_true, y_score)


def compute_per_class_accuracy(y_true, y_pred, n_classes=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accs = {}
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = (y_pred[mask] == c).mean()
    return accs


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"KL{i}" for i in range(5)],
        yticklabels=[f"KL{i}" for i in range(5)],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def full_evaluation_report(y_true, y_pred, y_score, phase_name, save_dir="results"):
    """Compute all metrics, print them, and save a JSON report."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    kw = compute_weighted_kappa(y_true, y_pred)
    auc = compute_auc_binary(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    per_class = compute_per_class_accuracy(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=[f"KL{i}" for i in range(5)])

    print(f"\n{'='*50}")
    print(f"  {phase_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Overall accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Weighted kappa    : {kw:.4f}")
    print(f"  Binary OA AUC     : {auc:.4f}")
    print(f"  Per-class accuracy:")
    for c, a in per_class.items():
        print(f"    KL{c}: {a:.4f}  ({a*100:.1f}%)")
    print(f"\n{report}")

    results = {
        "phase": phase_name,
        "overall_accuracy": acc,
        "weighted_kappa": kw,
        "binary_oa_auc": auc,
        "per_class_accuracy": {f"KL{k}": v for k, v in per_class.items()},
    }
    json_path = os.path.join(save_dir, f"{phase_name.lower().replace(' ', '_')}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(save_dir, f"{phase_name.lower().replace(' ', '_')}_confusion.png"),
    )
    return results
