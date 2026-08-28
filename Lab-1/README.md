# Single Layer Perceptron – Banknote Authentication

Implementation of a Single Layer Perceptron **from scratch** (no ML frameworks for the model itself) for binary classification of banknotes as authentic or forged.

> Deep Learning Laboratory – Experiment 1
> Shiv Nadar University Chennai | B.Tech AI & Data Science

## Overview

The perceptron is trained using the classic weight update rule with a step activation function, on the UCI **Banknote Authentication** dataset.

- **Dataset:** 1372 samples, 4 numerical features, binary target
- **Features:** Variance, Skewness, Curtosis, Entropy
- **Target:** 0 = Authentic, 1 = Forged

## Dataset Description

**Banknote Authentication Dataset** ([UCI ML Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication))

Features were extracted from wavelet-transformed images of genuine and forged banknote specimens.

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 (numerical) |
| Classes | 2 |
| Missing values | None |
| Task | Binary Classification |

**Feature columns:**
- `Variance` – variance of wavelet-transformed image
- `Skewness` – skewness of wavelet-transformed image
- `Curtosis` – curtosis of wavelet-transformed image
- `Entropy` – entropy of the image

**Target column (`Class`):**
- `0` – Authentic banknote
- `1` – Forged banknote

## Workflow

1. **EDA** – summary stats, histograms, correlation heatmap, scatter plot, boxplots
2. **Preprocessing** – `StandardScaler` normalization, 80/20 train-test split
3. **Model** – Perceptron implemented from scratch (NumPy), trained with learning rate 0.01 for up to 30 epochs
4. **Evaluation** – Accuracy, Precision, Recall, F1-score, Confusion Matrix
5. **Analysis** – training error curve, weight/bias evolution, learning rate comparison (0.001 / 0.01 / 0.1)

## Results

| Metric | Score |
|---|---|
| Accuracy | 97.82% |
| Precision | 98.40% |
| Recall | 96.85% |
| F1-score | 97.62% |

The model converges quickly, since the two classes are close to linearly separable — validated visually via the Variance vs. Skewness scatter plot.

## Tech Stack

- Python, NumPy
- pandas
- matplotlib, seaborn
- scikit-learn (only for preprocessing & metrics, not the model)

## References

- F. Rosenblatt, *The Perceptron*, Psychological Review, 1958
- UCI ML Repository – Banknote Authentication Dataset
uthentication Dataset
