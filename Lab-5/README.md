# Experiment 5 — CNN Training, Regularization, Optimization, Transfer Learning & Cross-Validation

CS3807 – Deep Learning Laboratory | Shiv Nadar University Chennai

## Contents

| File | Description |
|---|---|
| `Experiment5_Report.tex` | Full LaTeX source of the completed lab report |
| `Experiment5_Report.pdf` | Compiled 20-page PDF report |
| `Experiment5_Plots.zip` | All 15 plots (Plot 1–15) referenced in the report, as PNG images |
| `README.md` | This file |

## What's in the report

The report follows the official Experiment 5 template section-by-section, with all
placeholder tables and figures filled in using the actual training logs and plots
generated for this run:

1. Objective, Learning Outcomes, Dataset/Setup, MobileNetV2 architecture
2. Weight Initialization (Zero / Random / Xavier / He) — Plots 1–2
3. Regularization & Overfitting (None / L2 / Dropout / BatchNorm) — Plots 3–4
4. Batch Normalization (worked numerical example + with/without BN) — Plot 5
5. Optimizers (SGD / Momentum / RMSProp / Adam) — Plots 6–7
6. CNN Hyperparameter Tuning (learning rate, batch size, dropout rate) — Plots 8–10
7. Transfer Learning vs. Fine-Tuning (Case A vs. Case B) — Plots 11–12
8. 5-Fold Cross-Validation over configurations C1–C4 — Plot 13
9. Final Model Evaluation (confusion matrix, misclassified samples) — Plots 14–15
10. Overall Results summary table
11. All 23 discussion questions, answered
12. Additional Exercise (proposed follow-up configurations C5/C6)

Every plot is followed by a short **Inference** paragraph (what the plot shows, the
trend, and why), as required by Section 14 of the template.

## How to recompile the PDF

The report is a single-file LaTeX document with no external `.bib` file. From a
TeX Live (or similar) installation:

```bash
pdflatex Experiment5_Report.tex
pdflatex Experiment5_Report.tex   # run twice to resolve section numbering/refs
```

Requires the images from `Experiment5_Plots.zip` to be unzipped into an `img/`
subfolder alongside the `.tex` file, using these filenames (already matched to the
`\includegraphics` calls in the source):

```
img/plot1_init_loss.png
img/plot2_init_valacc.png
img/plot3_reg_acc.png
img/plot4_reg_loss.png
img/plot5_bn.png
img/plot6_opt_loss.png
img/plot7_opt_valacc.png
img/plot8_lr.png
img/plot9_batch.png
img/plot10_dropout.png
img/plot11_fe_ft.png
img/plot12_loss_beforeafter.png
img/plot13_cv.png
img/plot14_confmat.png
img/plot15_misclassified.png
```

Packages used (all standard, included in most TeX distributions):
`geometry, graphicx, amsmath, amssymb, booktabs, array, longtable, float,
hyperref, enumitem, fancyhdr, titlesec, xcolor`.

## Known data note — please check before submitting

Section 11 (K-Fold Cross-Validation) computes the final configuration C4's mean CV
accuracy directly from the 5 fold values you provided (84.78, 86.55, 88.45, 84.78,
87.91 → **86.49% ± 1.53**).

Section 13 (Overall Results) currently shows C4's CV Accuracy as **89.46%**, per
values you gave separately for that summary table. These two numbers for the same
configuration don't match. If 89.46% is the correct final figure, let me know and
I'll update the Section 11 fold table/mean (and the related inference text) so the
whole report is internally consistent — right now the two sections tell slightly
different stories about C4's performance.

## Editable placeholders worth reviewing

- **Section 11**, C1–C3 configuration *descriptions* (exact hyperparameters) were
  inferred from context since only accuracy numbers were supplied — replace with
  your actual settings if different.
- **Section 8** (Optimizer table), per-optimizer training *time* was not in the
  logs, so it's marked "Not individually timed."
- **Section 16** (Additional Exercise), configurations C5/C6 are proposed, not
  run — no training logs exist for them in this submission.
