# CNN Transfer Learning on Oxford-IIIT Pet Dataset (MobileNetV2)

**CS3807 – Deep Learning Laboratory | Experiment 5**
Shiv Nadar University Chennai · B.Tech AI & Data Science · Semester V

## Overview

This experiment performs a systematic study of how weight initialization,
regularization, optimization algorithms, hyperparameter choices, and transfer
learning affect the performance of a CNN image classifier. A single
architecture — **MobileNetV2**, pretrained on ImageNet — is used throughout
so that every comparison isolates one design choice at a time.

## Dataset

**Oxford-IIIT Pet Dataset** — 37 cat and dog breeds, RGB images of varying
size, resized to 224×224×3 for this experiment.

| Split | Images |
|---|---|
| Train-val pool | 3,680 |
| Test (held out, untouched until final evaluation) | 3,669 |
| → Train | 3,128 |
| → Validation | 552 |

Inputs are normalized to ImageNet statistics. The test set was kept
completely separate from all training/tuning and only used once, at the very
end, to evaluate the final selected model.

## Method / What was tested

1. **Weight initialization** — Zero, Random, Xavier/Glorot, He
2. **Regularization** — None, L2, Dropout, Batch Normalization
3. **Batch Normalization** — worked numerical example + with-vs-without comparison
4. **Optimizers** — SGD, Momentum, RMSProp, Adam
5. **Hyperparameter tuning** (one factor at a time) — learning rate (0.001 / 0.0001), batch size (16 / 32 / 64), dropout rate (0 / 0.25 / 0.5), optimizer (SGD / Adam), fine-tuning LR (1e-4 / 1e-5), frozen vs. partially-unfrozen backbone
6. **Transfer learning** — Case A: feature extraction (frozen backbone) vs. Case B: fine-tuning (unfrozen upper layers, small LR)
7. **Model selection** — 5-fold cross-validation over 4 candidate configurations (C1–C4)
8. **Final evaluation** — best configuration retrained on full training data, evaluated once on the untouched test set

## Key Results

**Initialization** — all four methods converge to similar loss/accuracy by
epoch 10 (the backbone is pretrained, so only a shallow head is randomly
initialized). **He initialization** gave the best final validation accuracy
(89.13%).

**Regularization** — "None" and "L2" clearly overfit (train accuracy ~96%
vs. validation ~87–88%, gap of 7.9–8.8 points). **Dropout** and **Batch
Normalization** kept the train/val gap under 1 point, with Dropout the most
effective at closing the gap and Batch Normalization showing no detectable
overfitting onset within 10 epochs.

**Optimizers** — plain **SGD** converged much slower than the others (still
improving at epoch 10). **Momentum, RMSProp, and Adam** all reached
85–89% validation accuracy within 2–3 epochs; Momentum and RMSProp were the
fastest and strongest overall.

**Hyperparameters** — best settings found: learning rate **0.001** (0.0001
was far too slow in the 5-epoch budget, only reaching 70%), batch size
**64**, dropout rate **0.25** (a sweet spot — 0.5 was worse), Adam over SGD,
and fine-tuning LR **1e-4** over 1e-5.

**Transfer learning** — **fine-tuning** (unfreezing upper backbone layers
with a small LR) beat pure feature extraction: 89.67% vs. 87.86% final
validation accuracy.

**Cross-validation (C1–C4)** — the fine-tuned configuration **C4** had the
best mean CV accuracy among the four candidates, with moderate fold-to-fold
variability.

**Final model (test set)** — the selected configuration was retrained and
evaluated once on the untouched 3,669-image test set:

| Metric | Value |
|---|---|
| Test Accuracy | 87.52% |
| Precision | 87.63% |
| Recall | 87.43% |
| F1-score | 87.30% |
| Trainable / Total Parameters | 47,397 / 2,271,269 |
| Training Time | 170.48 sec |

**Confusion matrix** — best-classified breeds included Keeshond (100%),
Samoyed, Leonberger, and Yorkshire Terrier (98% each); the hardest breeds
were visually similar same-species pairs — **Staffordshire Bull Terrier vs.
American Pit Bull Terrier** and **Birman vs. Ragdoll** — reflecting genuine
fine-grained visual similarity rather than a model-specific weakness.

## Files

- `Experiment5_Report.pdf` / `.tex` — full write-up with all plots, tables, inferences, and answers to the discussion questions
- `Experiment5_Plots.zip` — all 15 plots as PNGs

## Note on figures

CV Accuracy values in the Overall Results summary (Section 13 of the report)
and the fold-by-fold mean computed for C4 in Section 11 currently differ
(86.49% vs. 89.46%) — worth reconciling against your actual run logs before
final submission.
  run — no training logs exist for them in this submission.
