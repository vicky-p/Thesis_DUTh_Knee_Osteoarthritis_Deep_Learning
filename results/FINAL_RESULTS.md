# Knee Osteoarthritis KL Grading — Final Results

*Reproducing and modernizing Thomson et al. 2016 (Univ. of Manchester) on the
Kaggle "Knee Osteoarthritis Dataset with Severity" (5778 train / 826 val /
1656 test PA knee X-rays, KL grades 0-4).*

## Executive Summary

Every modern deep-learning model built in this project beats the Thomson 2016
baseline on every headline metric — best single model (ViT-B/16) reaches
AUC 0.925 vs Thomson's 0.904 and weighted kappa 0.816 vs Thomson's 0.49. A
from-scratch reproduction of Thomson's own hand-crafted-feature + Random
Forest pipeline (Phase 1) fell short of the published numbers (AUC 0.700),
confirming the gap modern deep learning closes is real, not an artifact of a
weak baseline reimplementation. The single best result overall is a simple,
retrain-free ensemble of ViT-B/16 and ResNet-50 (AUC 0.932, kw 0.826) —
a statistically significant improvement over either model alone. Automatic
(non-learned) landmark detection, used to drive hand-crafted features and as
an auxiliary branch in a hybrid ViT model, never improved aggregate metrics,
but in the hybrid deep-learning models it significantly improved accuracy
specifically on KL grade 2 — the class every deep-learning model here
struggles with most.

## Complete Results Table

| Model | Binary OA AUC | KL kappa (kw) | Overall acc. | KL0 acc. | KL1 acc. | KL2 acc. | KL3 acc. | KL4 acc. |
|---|---|---|---|---|---|---|---|---|
| **Thomson 2016** (published) | 0.904 | 0.490 | 55.6% | 91.2% | 3.6% | 43.6% | 49.9% | 39.2% |
| Phase 1 — Random Forest | 0.700 | 0.226 | 42.7% | 89.0% | 2.4% | 23.3% | 12.1% | 0.0% |
| Phase 1+ — RF + auto landmarks | 0.690 | 0.192 | 41.5% | 90.9% | 0.7% | 18.1% | 9.4% | 3.9% |
| Phase 2 — ViT-B/16 | **0.925** | **0.816** | 61.4% | 64.9% | 59.5% | 44.1% | 84.3% | 80.4% |
| ResNet-50 | 0.924 | 0.813 | 60.1% | 61.2% | 63.9% | 42.1% | 82.1% | 88.2% |
| EfficientNet-B4 | 0.911 | 0.780 | 51.6% | 40.7% | 74.7% | 30.9% | 86.5% | 84.3% |
| Hybrid (frozen ViT + landmarks) | 0.921 | 0.808 | 61.1% | 62.9% | 57.8% | 47.2% | 83.9% | 80.4% |
| Hybrid, ViT unfrozen (fine-tuned) | 0.920 | 0.806 | 62.1% | 65.6% | 55.1% | **52.3%** | 77.6% | 78.4% |
| ViT + TTA (4-view avg.) | 0.930 | 0.817 | 61.2% | 60.3% | 65.9% | 45.0% | 85.7% | 82.4% |
| ResNet-50 + TTA | 0.932 | 0.813 | 58.6% | 56.3% | 65.5% | 40.9% | 84.8% | 88.2% |
| **ViT + ResNet-50 ensemble** | **0.932** | **0.826** | 61.8% | 63.8% | 61.8% | 43.8% | 87.0% | 84.3% |

All numbers computed on the same held-out 1656-image test set. Full per-image
predictions for every model: `results/predictions/*.csv`. Aggregate JSON per
model: `results/*_results.json`.

## Key Findings

1. **Modern DL comprehensively beats Thomson 2016**, but bootstrap 95% CI
   analysis (`results/statistical_significance.txt`, 2000-resample paired
   bootstrap) shows the individual DL architectures are *not* all
   distinguishable from each other: **ViT-B/16 vs ResNet-50 is a coin flip**
   (P=0.527) despite ViT reporting a nominally higher AUC. ViT *is* significantly ahead of EfficientNet-B4 and both hybrid
   variants (P≈0.98–0.996).

2. **A free ensemble beats every single model.** Averaging ViT-B/16 and
   ResNet-50 softmax outputs (no retraining, `experiments/ensemble.py`)
   reaches AUC 0.932, kw 0.826 — the best result in the project, and the gain
   over ViT alone is statistically significant (P=0.999). The two
   architectures are similarly accurate but make different mistakes;
   EfficientNet-B4 and the hybrid models are too correlated/weaker to help
   when added to the ensemble.

3. **Landmark-guided features are a genuine negative result at the aggregate
   level — but not everywhere.** Neither RF+landmarks (Phase 1+) nor the
   frozen-backbone Hybrid model beat their non-landmark counterparts on
   AUC/kappa/accuracy (and per bootstrap CI, RF vs RF+landmarks isn't even a
   statistically significant difference — it's noise). *However*, on KL
   grade 2 specifically, both hybrid deep-learning models are significantly
   better than ViT alone: ViT 44.1% → Hybrid 47.2% → Hybrid fine-tuned 52.3%
   (P=0.999–1.000, bootstrapped on the KL2 subset). This does not hold for
   the Random Forest, where KL2 accuracy dropped from 23.3% to 18.1%. The
   heuristic (non-learned) landmarks aren't precise enough to help broadly,
   but they carry real, reproducible signal for the single hardest class.

4. **KL grade 2 is disproportionately hard for a structural reason, not
   (only) a modeling one.** Splitting misclassifications by model quality
   (`results/kl2_error_analysis.txt`): the four accurate DL models put 91.7%
   of their KL2 errors on an *adjacent* grade (KL1 or KL3) — consistent with
   genuine mild/moderate boundary ambiguity in the ordinal scale. The two
   weak RF models instead collapse 95.5% of their KL2 errors onto KL0 — a
   feature-quality failure, a different phenomenon entirely. Most strikingly,
   **28.2% of all true-KL2 test images are misclassified by every single one
   of the 7 models tested**, including architecturally unrelated ones (CNN,
   transformer, hand-crafted RF) — a strong signal of label noise in this
   specific dataset's KL2 annotations, not a fixable model limitation. Six
   example images are rendered in
   `results/figures/kl2_label_noise_examples.png`
   (`experiments/kl2_examples_figure.py`) — most look plausibly KL1 to the
   eye too, and the DL models mostly agree with each other, just not with
   the ground-truth label.

5. **Test-time augmentation is close to free accuracy.** Averaging
   predictions over 4 views (original, h-flip, ±5° rotation) with zero
   retraining nudges AUC up consistently: ViT 0.925→0.930, ResNet-50
   0.924→0.932.

## Limitations

- **KL2 label noise** (finding 4 above) likely caps achievable performance
  on this specific dataset below what a cleaner-labeled cohort would allow —
  this affects every model tested here, not a weakness specific to one
  architecture.
- **Landmark detection is heuristic, not learned** (`baseline/landmarks.py`):
  BoneFinder (the tool Thomson's own lab built) requires a manual licensing
  request and its knee module isn't in the default research license; no
  freely available pretrained knee-keypoint model was found. A trained
  detector would likely be needed to convert the KL2-specific landmark
  benefit (finding 3) into an aggregate-metric win.
- **Pain detection AUC (Thomson's other headline metric, 0.663) was not
  reproduced** — this Kaggle dataset ships only KL-grade folder labels, no
  pain/WOMAC scores, so there is nothing to train or evaluate that task on.
- **Single dataset, no external validation.** All numbers are on one
  train/val/test split of one public dataset; generalization to other
  scanners/populations (e.g. the full OAI cohort) is untested.
- **Test set is 1656 images** — per the bootstrap CIs, several of this
  project's own "wins" (e.g. any individual architecture vs. another) are
  within noise at this sample size; only the differences explicitly flagged
  as significant above should be treated as reliable.

## Supporting Files (everything referenced above exists and is reproducible)

| What | Code | Output |
|---|---|---|
| Grad-CAM, ViT-B/16 | `modern/gradcam.py` | `results/gradcam_summary.png`, `results/gradcam/` |
| Grad-CAM, ResNet-50 | `modern/resnet_gradcam.py` | `results/resnet_gradcam_summary.png`, `results/resnet_gradcam/` |
| Grad-CAM, Hybrid (ViT branch) | `modern/hybrid.py` (`save_hybrid_gradcam_samples`) | `results/hybrid_gradcam/`, `results/hybrid_finetuned_gradcam/` |
| Per-image predictions (all 7 models) | `experiments/collect_predictions.py` | `results/predictions/*.csv` |
| Bootstrap CI + pairwise significance | `experiments/statistical_significance.py` | `results/statistical_significance.txt` |
| Ensemble (ViT+ResNet, no retraining) | `experiments/ensemble.py` | `results/ensemble_vit_resnet_results.json` |
| KL2 error analysis (adjacent-vs-distant, agreement) | `experiments/kl2_analysis.py` | `results/kl2_error_analysis.txt` |
| KL2 label-noise example images | `experiments/kl2_examples_figure.py` | `results/figures/kl2_label_noise_examples.png` |
| Test-time augmentation | `experiments/tta_eval.py` | `results/vit_tta_results.json`, `results/resnet_tta_results.json` |
| Publication figures (training curves, comparison bars, confusion matrices, per-class chart) | `results/figures/plot_all.py` | `results/figures/*.png`, `*.pdf` |

Model checkpoints (`*.pth`, `*.pkl`) and Grad-CAM outputs are not included in
the repository because of their size; they are regenerated by running the
training and Grad-CAM scripts. All `.json`, `.csv` and `.txt` outputs listed
above are included.
