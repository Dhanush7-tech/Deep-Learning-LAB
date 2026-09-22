# CS3807 – Deep Learning Laboratory, Experiment 6
### End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

## What's in this folder

```
experiment6_report.tex   -> LaTeX source of the full report
experiment6_report.pdf   -> compiled report (17 pages)
figs/                    -> all 13 figures referenced by the .tex file
```

## Compiling the report yourself

The `.tex` file expects the `figs/` folder to sit right next to it (same
directory), since every figure is included with a relative path like
`figs/plot1_sensor.png`. If you move `experiment6_report.tex` somewhere
else, bring `figs/` along with it.

To rebuild the PDF:

```bash
pdflatex experiment6_report.tex
pdflatex experiment6_report.tex   # run twice for correct cross-references
```

Any standard LaTeX install (TeX Live, MacTeX, MiKTeX) or Overleaf will
compile this — it only uses `graphicx`, `amsmath`, `booktabs`, `hyperref`,
`float` and `enumitem`, which are all part of a normal LaTeX distribution.

## What the report covers

The report walks through the full lab: preprocessing the UCI HAR dataset
into `(N, 128, 9)` windows, training and comparing SimpleRNN, LSTM and GRU
classifiers on it, a CNN+GRU pipeline for video action recognition on a
small UCF101 subset, and a synthetic sequence-to-sequence reversal task
with an LSTM encoder–decoder. Every required plot has a written inference
underneath it, and the consolidated results tables from the lab manual are
filled in with actual numbers from the runs.

## A note on the numbers

Two separate full training runs of the RNN/LSTM/GRU classifiers are
referenced in the report — they gave slightly different results (this is
normal, especially for SimpleRNN, which trains less consistently than the
gated models). The main results table uses the first run, since that's the
one with matching confusion matrices and cross-checked precision/recall/F1
for all three models; the second run is mentioned separately as a
repeatability check. The training-curve figures come from the second run.
If your instructor wants everything from a single run, rerun the
RNN/LSTM/GRU training cell once, take fresh screenshots of all four plot
types, and swap the numbers/images into the relevant sections — the
report's structure won't need to change.

The CNN–GRU video result (100% test accuracy) is based on only 32 test
videos, which is flagged with a footnote in the report — treat it as a
demonstration of the pipeline rather than a robust benchmark.
