# Introduction-of-Deep-Learning
A comprehensive repository covering Deep Learning core concepts, Neural Network architectures (ANN, CNN, RNN), and practical end-to-end projects using TensorFlow, Keras, and PyTorch.

# Deep Learning Masterclass 🧠⚡

Welcome to the **Deep Learning** repository! This project serves as a comprehensive guide to understanding, building, and training Deep Neural Networks from basic perceptrons to advanced architectures like CNNs, RNNs, and Transformers.

It includes mathematical intuitions, hands-on implementations, and end-to-end practical projects using modern Deep Learning frameworks like **TensorFlow / Keras** and **PyTorch**.

---

## 🚀 Key Modules & Architectures Covered

### 1. Neural Network Foundations (ANN) 🧬
* **Perceptrons & Multi-Layer Perceptrons (MLP):** Forward propagation, loss functions, and backpropagation mathematics.
* **Activation Functions:** ReLU, Sigmoid, Tanh, Leaky ReLU, and Softmax.
* **Optimization Algorithms:** SGD, Momentum, RMSprop, and Adam Optimizer.
* **Overfitting Control:** Regularization ($L_1$/$L_2$), Dropout layers, Batch Normalization, and Early Stopping.

### 2. Computer Vision (CNN) 👁️
* **Convolutional Neural Networks (CNNs):** Convolution layers, Pooling (Max/Average), Padding, and Stride.
* **Popular Architectures:** LeNet, AlexNet, VGG, ResNet, and MobileNet.
* **Transfer Learning:** Fine-tuning pre-trained models on custom datasets.

### 3. Sequence Models & NLP (RNN / LSTM) 🔄
* **Recurrent Neural Networks (RNNs):** Handling sequential data, Vanishing/Exploding Gradient problems.
* **LSTM & GRU:** Gated architectures for long-term memory dependencies.
* **Attention Mechanisms & Transformers:** Introduction to modern NLP models (BERT, GPT basics).

### 4. Model Deployment & Optimization 🛠️
* Model Saving & Loading (`.h5`, SavedModel format, PyTorch `.pth`).
* Model Quantization and conversion for edge devices (TensorFlow Lite / ONNX).

---

## 🛠️ Tech Stack & Frameworks

* **Languages:** Python 3
* **Deep Learning Frameworks:** `TensorFlow`, `Keras`, `PyTorch`
* **Computer Vision:** `OpenCV`, `PIL`
* **Data Processing & Plotting:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
* **Hardware Acceleration:** CUDA / GPU computing support

---

## 💻 Sample Code Snippet (CNN Image Classifier in TensorFlow/Keras)

Here is a quick look at building a Convolutional Neural Network (CNN) for image classification:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Building a simple CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax') # 10 classes output
])

# 2. Compiling the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Printing Model Summary
model.summary()
