# Knee OA Classification (Kellgren–Lawrence grading)

Reproduces the hand-crafted-feature + Random Forest pipeline of Thomson (2016)
and compares it with modern deep-learning models (ViT-B/16, ResNet-50,
EfficientNet-B4, hybrid ViT + landmark models and ensembles) on the Kaggle
"Knee Osteoarthritis Dataset with Severity".

Full results and analysis: [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download from Kaggle:
<https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity>

Place the unzipped contents so the layout is:

```
data/dataset/
    train/  0/  1/  2/  3/  4/
    val/    0/  1/  2/  3/  4/
    test/   0/  1/  2/  3/  4/
```

## Project structure

| Folder | Contents |
|---|---|
| `baseline/` | Phase 1: hand-crafted features + Random Forest; Phase 1+: heuristic landmark detection and landmark-guided features |
| `modern/` | ViT-B/16, ResNet-50, EfficientNet-B4, hybrid models, Grad-CAM |
| `experiments/` | Per-image predictions, bootstrap significance, ensembles, test-time augmentation, KL2 error analysis |
| `evaluation/` | Shared metrics (binary OA AUC, weighted kappa, per-class accuracy) |
| `data/` | Data loading, transforms, weighted sampler |
| `results/` | Result JSON/CSV/TXT files and figures |

## Running

```bash
# Phase 1 — Random Forest (fixed image regions)
python baseline/train.py
python baseline/evaluate.py

# Phase 1+ — Random Forest with heuristic landmarks
python baseline/train_with_landmarks.py

# Phase 2 — ViT-B/16
python modern/train.py
python modern/evaluate.py

# Other architectures and hybrid models
python modern/resnet.py
python modern/efficientnet.py
python modern/hybrid.py

# Comparison table and figures
python compare.py
python results/figures/plot_all.py
```

Add `--help` to any script to see its options.

## Main results (test set, 1,656 images)

| Model | Binary OA AUC | Weighted kappa | Accuracy |
|---|---|---|---|
| Thomson 2016 (published) | 0.904 | 0.49 | 55.6% |
| Phase 1 — Random Forest | 0.700 | 0.226 | 42.7% |
| Phase 1+ — RF + landmarks | 0.690 | 0.192 | 41.5% |
| ViT-B/16 | 0.925 | 0.816 | 61.4% |
| ResNet-50 | 0.924 | 0.813 | 60.1% |
| ViT + ResNet-50 ensemble | **0.932** | **0.826** | 61.8% |

Thomson's results are from a different dataset (8,875 OAI images), so the
comparison is indicative rather than direct.
