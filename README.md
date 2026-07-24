Here's a short, simple README description you can use:

---

# Fashion-MNIST Classification with MLP

This project builds a simple neural network (MLP) to recognize clothing items from the Fashion-MNIST dataset — things like shirts, shoes, bags, and dresses from 28×28 grayscale images.

**What this project does:**
1. Loads and explores the Fashion-MNIST dataset (60,000 training images, 10,000 test images across 10 clothing categories)
2. Prepares the data by flattening images and scaling pixel values between 0 and 1
3. Builds a baseline neural network with two hidden layers
4. Trains the model and evaluates it using accuracy, precision, recall, F1-score, and a confusion matrix
5. Tries to improve the model automatically using Randomized Search to test different combinations of hyperparameters (like number of layers, neurons, learning rate, etc.)
6. Compares the automatically-optimized model against the original baseline model

**What we found:**
The simple, manually-built baseline model actually performed better (about 88% accuracy) than the automatically optimized one (about 86.5% accuracy). This shows that a well-chosen basic setup can sometimes beat a fancier automated search, especially when the search only has time to try a small number of combinations.

**Tech used:** Python, TensorFlow/Keras, Scikit-learn, SciKeras, Matplotlib/Seaborn for visualizations.
