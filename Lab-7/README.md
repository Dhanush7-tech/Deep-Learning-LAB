# Experiment 7 — Autoencoders, CAE, Denoising CAE and VAE on MNIST

CS3807 Deep Learning Laboratory · Shiv Nadar University Chennai
Dhanush Vellachi

This README walks through the whole experiment in the order it was actually
built and run, from raw data to the final consolidated report. It's meant as
a map of the project — what each file does, why each step exists, and how
everything connects — not a restatement of the lab manual.

---

## 1. What this experiment is about

The goal is to build and compare four related ways of squeezing an image
down to a small latent code and then rebuilding it:

1. **Fully Connected Autoencoder (FC AE)** — the baseline.
2. **Convolutional Autoencoder (CAE)** — same idea, but respects the 2D
   structure of an image instead of flattening it.
3. **Denoising Convolutional Autoencoder** — same architecture as the CAE,
   but trained to strip noise back out.
4. **Variational Autoencoder (VAE)** — instead of one fixed latent vector
   per image, it learns a *distribution*, which lets you sample brand-new
   digits that were never in the dataset.

Every model is trained and evaluated on **the same 10,000 train / 2,000 test
MNIST subset**, seeded (`seed = 42`) so all comparisons in the report are on
identical data.

---

## 2. Files in this project

| File | What it is |
|---|---|
| `fc_autoencoder_mnist.py` | Study 1 — FC autoencoder only: training, Plot 1, Plot 2, MSE/MAE/SSIM (Sections 7–9) |
| `cnn_autoencoder_comparison.py` | Study 2 — rebuilds the FC AE + builds the CAE on the same split, Plot 3, comparison table (Sections 10–11) |
| `denoising_autoencoder.py` | Study 3 — Gaussian + salt-and-pepper noise, denoising CAE, Plot 4a/4b, Plot 5 (Sections 12–14) |
| `vae_mnist.py` | Study 4 — VAE (encoder/sampling/decoder), Plot 6 (latent space), Plot 7 (generated samples), Plot 8 (interpolation) (Sections 15–20) |
| `final_comparison_all_models.py` | Retrains all four models fresh for a from-scratch consolidated comparison (Section 21 tables) |
| `latent_dimension_study.py` | Repeats the FC AE at $d_z \in \{2, 8, 16, 32\}$, Plot 10 (Section 27) |
| `Experiment_7.tex` | The full written lab report — every section, table and figure, with all required inferences written out |
| `Experiment_7.pdf` | Compiled version of the report above |
| `Experiment_7_figures.zip` | All 15 plot images used in the report, named to match the `.tex` source |

Each script is self-contained and runnable on its own in Google Colab — you
don't need to run them in a strict order except where noted below, but the
order below is the order they were actually developed and run in.

---

## 3. Step-by-step walkthrough

### Step 1 — Data preparation (common to every script)

- Load MNIST via `keras.datasets.mnist.load_data()`.
- Randomly sample 10,000 training images and 2,000 test images
  (`np.random.seed(42)` before sampling, so the same subset is reused
  everywhere).
- Rescale pixels from `[0, 255]` to `[0, 1]`.
- Keep two versions of the data: a **flattened** `(N, 784)` version for the
  FC autoencoder, and a **spatial** `(N, 28, 28, 1)` version for every
  convolutional model.
- Labels are loaded but never used as a training target anywhere — the only
  place a label is used at all is to colour points in the VAE latent-space
  plot (Plot 6), purely for visualization.

### Step 2 — Fully Connected Autoencoder (`fc_autoencoder_mnist.py`)

Architecture: `784 → Dense(128, relu) → Dense(32, relu) → Latent(16, relu)
→ Dense(32, relu) → Dense(128, relu) → Dense(784, sigmoid)`.

Trained with Adam (lr = 1e-3), batch size 128, 20 epochs, binary
cross-entropy loss — exactly the suggested configuration.

What comes out of this step:
- **Plot 1** — 10 test digits, original vs. reconstruction, each labelled
  with its own MSE.
- **Plot 2** — training vs. validation loss curve over 20 epochs.
- **Test MSE / MAE / Mean SSIM** — computed with `skimage.metrics.
  structural_similarity` for SSIM, plain NumPy for MSE/MAE.

This step establishes the baseline: the model preserves overall digit shape
but loses fine stroke detail, and the loss curves show clean convergence
with no overfitting (train and validation loss track each other closely the
whole way).

### Step 3 — Convolutional Autoencoder + FC vs. CAE comparison
(`cnn_autoencoder_comparison.py`)

This script **re-trains the FC AE from scratch on the identical seeded
split**, then builds the CAE using the exact architecture given in the
brief:

```
Input(28,28,1)
Conv2D(32,3,relu,same) → MaxPooling2D(2,same)
Conv2D(64,3,relu,same) → MaxPooling2D(2,same)
Conv2D(64,3,relu,same)                              # latent feature map
UpSampling2D(2) → Conv2D(32,3,relu,same)
UpSampling2D(2) → Conv2D(1,3,sigmoid,same)
```

Both models are evaluated on the same test set so the comparison is
apples-to-apples. Output:
- **Plot 3** — Original → FC-AE reconstruction → CAE reconstruction,
  stacked per digit.
- A side-by-side table of MSE, MAE, SSIM and parameter count for both
  models.

Result: the CAE wins on every metric at once, using roughly a third of the
parameters — direct evidence that preserving spatial locality (rather than
flattening the image) is what actually drives reconstruction quality here.

### Step 4 — Denoising Convolutional Autoencoder (`denoising_autoencoder.py`)

Two corruption types are implemented, as required:
- **Gaussian noise**: `x̃ = clip(x + N(0, σ²), 0, 1)`, σ ∈ {0.1, 0.2, 0.3}.
- **Salt-and-pepper noise**: a random proportion `p` of pixels forced to 0
  or 1, `p` ∈ {0.05, 0.10, 0.20}.

The model is trained on a **mix** of both noise types at random severities
per image (rather than one fixed setting), so it generalizes across
corruption types instead of overfitting to a single noise level. The target
is always the clean image — the noisy image is only ever the input, never
what the model is trained to reproduce.

Output:
- **Plot 4a / 4b** — Clean / Noisy / Denoised, shown separately for
  Gaussian (σ = 0.2) and salt-and-pepper (p = 0.10).
- **Plot 5** — MSE, MAE and SSIM plotted against Gaussian noise level, as
  three separate subplots (kept apart deliberately, since the three metrics
  live on incompatible scales).
- The same table repeated for salt-and-pepper severities, for a fuller
  picture of robustness.

Result: quality degrades gradually as noise increases (no sudden collapse),
and the model recovers the correct digit shape reliably even under fairly
heavy corruption — what's lost at high noise is fine texture, not identity.

### Step 5 — Variational Autoencoder (`vae_mnist.py`)

Latent dimension kept at **2**, specifically so it can be plotted directly.

- **Encoder**: `Conv2D(32,3,relu,stride2) → Conv2D(64,3,relu,stride2) →
  Flatten → Dense(32,relu) → (μ, log σ²)` each in ℝ².
- **Sampling layer**: implements the reparameterization trick,
  `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`.
- **Decoder**: `Dense → Reshape → Conv2DTranspose ×2 → Conv2D(sigmoid)`
  back to 28×28×1.
- **Loss**: a custom `train_step`/`test_step` computing
  `L = L_rec (binary cross-entropy) + L_KL` (closed-form KL divergence
  against N(0, I)), with both terms tracked separately.

Three things come out of this step:
- **Plot 6** — every test image encoded to (μ₁, μ₂) and scatter-plotted,
  coloured by digit label (labels used here only for the colour, never fed
  into training).
- **Plot 7** — 25 images generated by sampling `z ~ N(0, I)` directly and
  decoding, displayed as a 5×5 grid.
- **Plot 8** — latent-space interpolation: pick two encoded means (one
  digit-0 image, one digit-1 image), walk a straight line between them in
  11 steps (`α = 0, 0.1, …, 1`), decode every step, and display the
  sequence.
- The VAE's own loss curve (training vs. validation total loss).

Result: digits that are visually distinctive (0, 1, 7) form their own
separate regions in the latent plane; everything else overlaps heavily in
the middle. Generated samples are mostly plausible but skew toward an
ambiguous "loop" shape, which traces directly back to that overlapping
central region. Interpolation between two points is smooth and continuous
the whole way, with no abrupt jumps — a property a plain (non-variational)
autoencoder's latent space isn't guaranteed to have.

### Step 6 — Consolidated comparison (`final_comparison_all_models.py`)

Retrains **all four models from scratch** on the identical split, purely so
the final numbers in the report all come from one clean, reproducible run
rather than being stitched together from separate sessions. Produces:
- Table 1 — MSE / MAE / SSIM / Parameters for FC AE, CAE, Denoising CAE.
- Table 2 — VAE reported separately (Reconstruction Loss, KL Loss, Total
  Loss, Test MSE/MAE/SSIM), since its objective isn't directly comparable
  to the other three (it optimizes for a structured latent space, not pure
  reconstruction accuracy).

This is also where the **reconstruction-error histogram across all four
models** (Plot 10 in the report, Section 23) and the **top-5 highest-error
FC AE images** (Section 24) come from — both computed on the per-image MSE
distribution over the test set.

### Step 7 — Latent-dimension study (`latent_dimension_study.py`)

Repeats *only* the FC autoencoder, keeping everything (architecture shape,
optimizer, batch size, epochs, loss) fixed except the bottleneck width,
at `d_z ∈ {2, 8, 16, 32}`. Reports MSE, SSIM and parameter count for each,
plus the required MSE-vs-latent-dimension plot.

Result: quality improves steadily as the bottleneck widens, but with
diminishing returns — the jump from `d_z = 2 → 8` buys far more than
`d_z = 16 → 32` — while parameter count barely moves across the whole
range, since the bottleneck layer itself is tiny next to the surrounding
128-/32-unit hidden layers.

### Step 8 — Writing up the report (`Experiment_7.tex` / `.pdf`)

Everything above is pulled together into one LaTeX report following the
lab's exact section numbering (Objective → Learning Outcomes → Dataset →
... → Discussion Questions → Additional Exercises → Expected Outcome →
References). All tables carry the actual numbers from the runs above (never
placeholder values), all ten required plots are embedded, and every
required inference is written out in plain, first-person analysis rather
than templated "the plot shows X" phrasing.

---

## 4. How to reproduce it yourself

1. Open a fresh Google Colab notebook (a free-tier CPU/GPU runtime is
   plenty — nothing here needs more than a couple of minutes per model).
2. Run the scripts in the order listed in Section 2 above, in separate
   cells (or separate notebooks) — each one is self-contained and only
   needs `tensorflow`, `numpy`, `matplotlib`, and `scikit-image`
   (`pip install scikit-image` if it isn't already available).
3. Each script prints its own metrics table and saves its plots as `.png`
   files in the working directory.
4. To rebuild the PDF report: put `Experiment_7.tex` in the same folder as
   the 15 PNGs from `Experiment_7_figures.zip` (they're already named to
   match what the `.tex` file references) and run `pdflatex` on it twice
   (once to generate cross-references, once to resolve them).

---

## 5. Key results at a glance

| Model | MSE | MAE | SSIM | Parameters |
|---|---|---|---|---|
| Fully Connected AE | 0.0215 | 0.0577 | 0.7582 | 211,040 |
| Convolutional AE | 0.0025 | 0.0152 | 0.9728 | 74,497 |
| Denoising CAE | 0.0034 | 0.0175 | 0.9631 | 74,497 |
| VAE (recon-only) | 0.0474 | 0.1101 | 0.4790 | 184,421 |

The headline takeaway: convolution beats flattening decisively for pure
reconstruction (better on every metric, with a third of the parameters),
denoising costs almost nothing in accuracy while buying real robustness to
corruption, and the VAE trades reconstruction sharpness for something none
of the other three models can do at all — a smooth, sampleable latent space
that can generate entirely new handwritten digits.
