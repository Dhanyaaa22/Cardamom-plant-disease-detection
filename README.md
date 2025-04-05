# Cardamom Plant Disease Detection 

This project uses **IoT** and **Machine Learning** to detect diseases in cardamom plants at an early stage, helping farmers take timely action and improve crop yield.

## Overview

The system combines **image processing** techniques and a **deep learning model** (CNN based on VGG19) to classify diseases in cardamom leaves. It also integrates with IoT hardware to enable real-time monitoring and automated alerts.

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- VGG19 CNN architecture
- Raspberry Pi / Arduino (for IoT integration)
- Camera module for image capture

## How It Works

1. **Image Collection**: Leaf images are captured using a camera connected to an IoT device.
2. **Preprocessing**: Images are cleaned and enhanced using OpenCV (noise removal, resizing, contrast adjustment).
3. **Model Training**: A CNN model based on VGG19 is trained to classify leaf images into healthy or diseased categories.
4. **Real-Time Detection**: The trained model runs on the IoT device to detect diseases in real time.
5. **Alerts**: If a disease is detected, the system can send alerts or log the result for further action.

## Dataset

A custom dataset of cardamom leaf images was created, categorized into:
- Healthy leaves
- Fungal infections
- Bacterial infections
- Nutrient deficiencies

*(You can mention the number of images and data sources if available.)*

## Features

- Automated disease classification with high accuracy
- Real-time detection with IoT integration
- Portable and scalable for small to medium plantations
- Supports early intervention to reduce crop loss

## 📁 Project Structure

```bash
cardamom-disease-detection/
├── dataset/
├── model/
│   └── vgg19_model.h5
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── detect_disease.py
├── iot/
│   └── iot_integration.py
├── README.md
