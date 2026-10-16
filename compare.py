"""
Load saved result JSON files and print a side-by-side comparison table.

Usage:
    python compare.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

THOMSON = {
    "Binary OA AUC":       0.904,
    "KL grading (kw)":     0.49,
    "Overall accuracy":    0.556,
    "KL grade 2 accuracy": 0.436,
}

PHASE_FILES = {
    "Phase 1 (RF)":     "phase_1_rf_results.json",
    "Phase 1+ (RF+LM)": "phase1plus_results.json",
    "Phase 2 (ViT)":    "phase_2_vit_v2_results.json",
    "ResNet-50":        "resnet_results.json",
    "EfficientNet-B4":  "efficientnet_results.json",
    "Hybrid (ViT+LM)":  "hybrid_results.json",
    "Hybrid finetuned": "hybridfinetuned_results.json",
    "ViT+ResNet ensemble": "ensemble_vit_resnet_results.json",
}


def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val):
    if val is None:
        return "?.???"
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        if val > 1:
            return f"{val:.1f}%"
        return f"{val:.3f}"
    return str(val)


def main():
    results = {name: load_results(fname) for name, fname in PHASE_FILES.items()}

    def get(r, key):
        if r is None:
            return None
        m = {
            "Binary OA AUC":       r.get("binary_oa_auc"),
            "KL grading (kw)":     r.get("weighted_kappa"),
            "Overall accuracy":    r.get("overall_accuracy"),
            "KL grade 2 accuracy": r.get("per_class_accuracy", {}).get("KL2"),
        }
        return m.get(key)

    metrics = list(THOMSON.keys())
    col_w = [31, 14] + [16] * len(results)
    sep = "┼".join("─" * w for w in col_w)
    hdr = "│".join(
        s.center(w) for s, w in zip(
            ["Metric", "Thomson 2016"] + list(results.keys()), col_w
        )
    )

    print(f"\n┌{'┬'.join('─'*w for w in col_w)}┐")
    print(f"│{hdr}│")
    print(f"├{sep}┤")
    for m in metrics:
        row = [m.ljust(col_w[0] - 2)]
        row.append(fmt(THOMSON[m]).center(col_w[1] - 2))
        for i, name in enumerate(results):
            val = get(results[name], m)
            if m in ("Overall accuracy", "KL grade 2 accuracy") and val is not None:
                cell = fmt(val * 100)
            else:
                cell = fmt(val)
            row.append(cell.center(col_w[2 + i] - 2))
        line = "│ " + " │ ".join(row) + " │"
        print(line)
    print(f"└{'┴'.join('─'*w for w in col_w)}┘\n")


if __name__ == "__main__":
    main()
