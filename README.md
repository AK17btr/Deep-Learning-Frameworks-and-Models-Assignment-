# Deep Learning on CIFAR-10 with PyTorch

This project focuses on training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch. It includes:
- Model architecture development and training.
- Hyperparameter tuning and data augmentation.
- Exporting the trained model to ONNX format for further optimization.
- Measuring inference latency on both CPU and GPU.

---

## Project Overview

### **Model Architecture**
The CNN model contains:
- **3 Convolutional Layers**: Extract spatial features from the CIFAR-10 images.
- **2 Dense (Fully Connected) Layers**: Map features to the 10 CIFAR-10 classes.

### **Training Parameters**
- **Learning Rate**: Tuned during training.
- **Batch Size**: 64.
- **Epochs**: 10 (training stopped after learning stabilized).
- **Optimizer**: Adam.
- **Loss Function**: Cross-Entropy Loss.

### **Performance Metrics**
The training process evaluated the model using:
- Accuracy
- Precision
- Recall
- F1-Score

### **Data Augmentation**
Data augmentation techniques applied:
- Random Horizontal Flip
- Random Crop
- Normalization

---

## Latency Measurement

Latency was measured for the best-trained model:
- **ONNX Export**: The PyTorch model was exported to ONNX format.
- **Inference Latency**:
  - **CPU**: 56.53 ms
  - **GPU**: 141.10 ms

---

## Repository Contents

- `models/`: Contains the PyTorch model weights (`.pth`) and the ONNX model (`.onnx`).
- `data/`: Layer-wise inputs, outputs, and weights saved as `.pt` files.
- `configs/`: JSON config file describing the layers in the model.
- `notebooks/`: Jupyter Notebook for model training and latency measurement.
- `report/`: Detailed report of the training process, hyperparameter tuning, and latency measurements.
- `requirements.txt`: List of dependencies for running the project.

---

## Getting Started

### **Dependencies**
To reproduce the results, install the required Python libraries:
```bash
pip install -r requirements.txt
