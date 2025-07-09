# 🌦️ Weather Image Classification using CNN

This project focuses on classifying weather conditions from images using Convolutional Neural Networks (CNNs). While traditional weather classification relies on sensor data or manual observation, this project demonstrates that visual cues from weather imagery can be used effectively to train a deep learning model that predicts weather types such as cloudy, foggy, rainy, snowy, and sunny.

---

## Project Overview

The project focuses on building a CNN trained on labeled weather images to perform multi-class classification. The model learns visual patterns in the images such as cloud shapes, light conditions, and textures that distinguish one weather class from another. To improve performance and reduce overfitting, data augmentation, and dropout regularization are used.

Key steps include:
- **Image preprocessing**: Resize, normalize, and augment input images
- **Model design**: CNN with two convolutional layers followed by fully connected layers
- **Training**: Optimized using the Adam optimizer with cross-entropy loss
- **Hyperparameter tuning**: Various combinations of dropout, learning rate, and filter sizes tested
- **Evaluation**: Accuracy and confusion matrix on test set

---

## Data Collection

The dataset consists of labeled folders of weather images, with classes such as `cloudy`, `foggy`, `rain`, `shine`, and `snow`. The images were downloaded from publicly available repositories and vary in size and quality. To ensure consistency during training, all images are resized to 128×128 resolution and normalized using ImageNet mean and standard deviation values.

---

## Methodology

1. **Preprocessing**:
   - Resize all images to a uniform shape
   - Apply random horizontal flips and rotations for augmentation
   - Normalize using standard mean and std values

2. **Model Architecture**:
   - Two convolutional layers and max pooling
   - Dropout applied after convolutional and dense layers
   - Fully connected layers for classification

3. **Training Strategy**:
   - Loss: CrossEntropyLoss
   - Optimizer: Adam
   - Metric: Accuracy on training, validation, and test sets
   - Hyperparameter tuning via grid search

---

## Project Structure

- `weather_prediction.ipynb`: The main notebook containing the entire workflow from preprocessing to evaluation
- `dataset/`: Contains subfolders of weather images organized by class (e.g., `cloudy/`, `foggy/`, etc.)

---

## Future Work

- Incorporate pre-trained models like ResNet or EfficientNet for better performance
- Address class imbalance using weighted loss functions or oversampling
- Deploy the trained model as a web app or mobile interface for real-time weather recognition

---
