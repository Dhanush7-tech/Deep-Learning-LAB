# Image Classification using Convolutional Neural Networks (CNNs)

## 📌 Experiment Overview
This experiment provides hands-on experience in designing, implementing, and evaluating Convolutional Neural Networks (CNNs) for multi-class image classification. Built using the **TensorFlow/Keras** deep learning framework, the project explores the foundational building blocks of CNNs and how they progressively learn hierarchical image representations. 

Through this lab, we investigate how convolution kernels extract meaningful visual features (like edges and textures) and how pooling operations optimize computational efficiency by reducing spatial dimensions.

## 📊 Dataset Description: CIFAR-10
This experiment utilizes the widely recognized **CIFAR-10 (Canadian Institute For Advanced Research)** dataset. It is a standard benchmark for computer vision and machine learning tasks.

*   **Total Images:** 60,000 color images
*   **Image Dimensions:** 32 x 32 pixels (3 channels: RGB)
*   **Number of Classes:** 10 mutually exclusive object categories
*   **Data Split:** 
    *   **Training Set:** 50,000 images (5,000 images per class)
    *   **Testing Set:** 10,000 images (1,000 images per class)
*   **Classes:** 
    1. Airplane
    2. Automobile
    3. Bird
    4. Cat
    5. Deer
    6. Dog
    7. Frog
    8. Horse
    9. Ship
    10. Truck

## 🧠 Key Concepts & Architecture
The implemented CNN architecture incorporates several core layers and operations:
*   **Convolutional Layers:** To extract spatial features using convolution kernels.
*   **Activation Functions:** Introducing non-linearity (e.g., ReLU) to the network.
*   **Pooling Layers:** Reducing spatial dimensions of feature maps (e.g., Max Pooling, Average Pooling).
*   **Flattening:** Converting 2D spatial features into a 1D vector.
*   **Fully Connected (Dense) Layers:** Mapping the extracted features to final class probabilities.

## ⚙️ Hyperparameters Investigated
The experiment involves tuning and analyzing the influence of the following hyperparameters on model performance:
*   **Kernel Size** (e.g., 3x3, 5x5)
*   **Stride & Padding** (Valid vs. Same)
*   **Number of Filters** (Depth of the feature maps)
*   **Optimizer Selection** (e.g., Adam, SGD)
*   **Batch Size & Training Epochs**

## 📈 Evaluation & Visualizations
To thoroughly interpret the learning behavior and performance of the network, the following metrics and visualizations are generated:

**Performance Metrics:**
*   Accuracy
*   Precision, Recall, and F1-Score
*   Confusion Matrix
*   Classification Report

**Visual Analysis:**
*   **Sample Images & Class Distribution:** Visualizing the raw CIFAR-10 data.
*   **Training/Validation Curves:** Plotting loss and accuracy over epochs to monitor learning and detect overfitting.
*   **Intermediate Feature Maps:** Visualizing the outputs of hidden convolutional layers to understand what the network "sees" at different depths.
*   **Trainable Parameters Analysis:** Comparing the parameter counts across different layer configurations and pooling strategies.

## 🛠️ Technologies Used
*   **Python**
*   **TensorFlow / Keras** 
*   **NumPy** (Numerical operations)
*   **Matplotlib / Seaborn** (Data visualization)
