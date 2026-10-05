# Motion Sensor Classification using ANN Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains a deep learning classification project that trains an Artificial Neural Network (ANN) using TensorFlow and Keras to classify driving and motion behaviors (such as NORMAL vs AGGRESSIVE) based on accelerometer and gyroscope sensor readings[cite: 18].

---

## Dataset Notice
*Note: The dataset (`train_motion_data.csv` and `test_motion_data.csv`) used in this project[cite: 18] contains motion sensor time-series measurements.*

---

## Dataset Features & Preprocessing
* **Timestamp Removal**: Dropped unnecessary `Timestamp` columns from training and testing sets[cite: 18].
* **Label Encoding**: Transformed categorical class labels into numeric values using `LabelEncoder`[cite: 18].
* **Feature Scaling**: Normalized features using `StandardScaler`[cite: 18].

---

## Project Workflow
1. **Data Loading & Cleaning**: Reading CSV datasets and removing timestamp variables[cite: 18].
2. **Preprocessing**: Label encoding and standard scaling[cite: 18].
3. **Model Architecture**: Building a Sequential neural network featuring Dense layers with ReLU activations, Dropout regularization layers (0.3), and a Softmax output layer[cite: 18].
4. **Model Compilation**: Compiling with the Adam optimizer and sparse categorical crossentropy loss[cite: 18].
5. **Training with Callbacks**: Implementing `EarlyStopping` with patience to monitor validation loss and restore best weights[cite: 18].
6. **Model Persistence**: Saving the optimized neural network model into `motion_ann_model_optimized.h5`[cite: 18].
7. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/motion-sensor-classification.git](https://github.com/YOUR_USERNAME/motion-sensor-classification.git)
   cd motion-sensor-classification
