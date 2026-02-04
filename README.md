# 🚗 Driver Drowsiness Detection System

[![Python](https://img.shields.io)](https://www.python.org/)
[![TensorFlow](https://img.shields.io)](https://www.tensorflow.org)
[![OpenCV](https://img.shields.io)](https://opencv.org)
[![License: MIT](https://img.shields.io)](https://opensource.org)

A real-time deep learning system designed to detect driver drowsiness and prevent road accidents by monitoring eye closure and triggering an alarm.

---

## 📌 Overview
Drowsy driving is a leading cause of road fatalities. This project implements a non-intrusive monitoring system that:
1. Captures live video feed using **OpenCV**.
2. Isolates the eye region of interest (ROI).
3. Classifies eye state (Open/Closed) using a fine-tuned **MobileNet CNN**.
4. Triggers an **audible alarm** if eyes remain closed for a specific threshold of frames.

---

## 🏗️ Technical Architecture

### 1. Data Processing
* **Frame Capture:** Live feed acquisition from webcam via OpenCV.
* **ROI Extraction:** Haar Cascades or facial landmark detection to isolate eyes.
* **Augmentation:** To improve generalization, the training pipeline includes:
    * Random rotations and zooms.
    * Horizontal flips and brightness adjustments.

### 2. Model Development
* **Base Model:** [MobileNet](https://keras.io) (pre-trained on ImageNet).
* **Fine-Tuning:** The base layers were frozen to retain feature extraction, while the **last few layers were unfreezed** to learn specific eye-state nuances from the custom dataset.
* **Regularization:** 
    * Implemented **L2 Regularization** in dense layers to prevent weight explosion.
    * Added **Dropout** layers to mitigate overfitting.

### 3. Real-Time Logic
* **State Persistence:** A frame counter tracks consecutive "Closed" eye detections.
* **Alert System:** If `counter > THRESHOLD`, a sound alert is triggered using `winsound` or `pygame`.

---
