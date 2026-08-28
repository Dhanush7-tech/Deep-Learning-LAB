# Comparative Study of Deep CNN Architectures Using Transfer Learning

**CS3807 – Deep Learning Laboratory | Experiment 4**
Shiv Nadar University Chennai — B.Tech AI & Data Science, Semester V

## Overview

This project implements and evaluates transfer learning on CIFAR-10 using a pretrained **MobileNetV2** backbone, and contextualizes the results against the historical evolution of CNN architectures (LeNet-5 → AlexNet → VGG16 → GoogleNet → ResNet).

## Objectives

- Study the evolution of deep CNN architectures
- Compare LeNet-5, AlexNet, VGG16, GoogleNet, and ResNet
- Implement and fine-tune transfer learning on a pretrained CNN
- Evaluate classification performance using standard metrics
- Conduct a hyperparameter sensitivity study

## Dataset

**CIFAR-10**
| Property | Value |
|---|---|
| Training images | 50,000 |
| Testing images | 10,000 |
| Classes | 10 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck) |
| Image size | 32 × 32 × 3 |

## Methodology

1. **Data preparation** — load CIFAR-10, normalize pixels to [0,1], inspect sample images and shapes.
2. **Transfer learning setup** — load MobileNetV2 with ImageNet weights, remove the classifier head, freeze the convolutional base, add GlobalAveragePooling → Dense(ReLU) → Dense(Softmax).
3. **Initial training** — train the new classifier head (frozen base) for ~15 epochs using Adam (lr = 0.001), batch size 32, categorical cross-entropy loss.
4. **Fine-tuning** — unfreeze the last convolutional block and continue training for additional epochs at a lower effective learning rate to adapt features to CIFAR-10.
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, and classification report on the test set.
6. **Hyperparameter study (OFAT)** — sweep learning rate, batch size, epochs, optimizer, dense units, and frozen-layer strategy against a fixed baseline configuration.

## Results Summary

| Metric | Value |
|---|---|
| Training Accuracy (post fine-tune) | 37.34% |
| Testing Accuracy | 39.35% |
| Precision (weighted avg) | 38.95% |
| Recall (weighted avg) | 39.35% |
| F1-score (weighted avg) | 38.75% |
| Total Parameters | 2,423,242 |
| Training Time | ≈20.6 min (10.1 min frozen + 10.5 min fine-tune, 23 epochs) |

**Key hyperparameter finding:** Partial layer unfreezing gave by far the largest single improvement, raising test accuracy from a baseline of 0.344 to **0.501**, by letting the last convolutional block adapt from generic ImageNet filters to CIFAR-10-specific features. A lower learning rate (0.0001) and SGD both underperformed the Adam/0.001 baseline.

**Architectural comparison (literature, ImageNet top-1):**
| Model | Params | Accuracy | Empirically Run? |
|---|---|---|---|
| LeNet-5 | 60K | N/A (MNIST/OCR) | No |
| AlexNet | 61M | ~57.1% | No |
| VGG16 | 138M | ~71.5% | No |
| GoogleNet | 6.8M | ~69.8% | No |
| ResNet50 | 25.6M | ~76.0% | No |
| **MobileNetV2** | **2.42M** | **39.35% (CIFAR-10)** | **Yes** |

> Note: Only MobileNetV2 was empirically trained here; other architectures' figures are cited literature values on ImageNet, included for context only and not directly comparable to the CIFAR-10 result.

## Repository Structure

```
.
├── README.md                  # This file
├── notebook/                  # Training & evaluation notebook (Colab/Jupyter)
├── data/                      # CIFAR-10 (auto-downloaded via tf.keras.datasets)
├── models/                    # Saved model checkpoints (frozen-base, fine-tuned)
├── results/
│   ├── figures/                # Sample images, accuracy/loss curves, confusion matrix
│   └── hyperparameter_sweep/   # OFAT sweep results and plots
└── report/                    # Lab report (PDF/LaTeX)
```

## Requirements

```
tensorflow>=2.x
numpy
matplotlib
scikit-learn
seaborn
```

Install with:
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Run the data preparation cells to load and normalize CIFAR-10.
3. Run the transfer learning setup cell to build the MobileNetV2-based model.
4. Train the frozen-base model, then run the fine-tuning cell.
5. Run the evaluation cell to generate metrics, confusion matrix, and classification report.
6. (Optional) Run the hyperparameter sweep cell to reproduce the OFAT study.

## Key Learnings

- Transfer learning enables competitive results on small datasets with far less training time than training from scratch.
- Fine-tuning is essential: frozen ImageNet features alone are generic and underfit CIFAR-10's low-resolution, domain-specific images.
- Unfreezing convolutional layers causes a temporary accuracy/loss spike (batch-norm statistics get disturbed) before the model re-stabilizes and improves further.
- Class confusions concentrate among visually/semantically similar categories (Cat/Dog, Bird/Deer/Frog, Ship/Airplane/Truck).

## References

1. LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, IEEE, 1998.
2. Krizhevsky et al., *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS, 2012.
3. Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, ICLR, 2015.
4. Szegedy et al., *Going Deeper with Convolutions*, CVPR, 2015.
5. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
6. Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press, 2016.
7. [TensorFlow Documentation](https://www.tensorflow.org)
8. [Keras Documentation](https://keras.io)
