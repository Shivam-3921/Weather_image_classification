# Weather Image Classification using CNN

This project focuses on classifying weather conditions from images using Convolutional Neural Networks (CNNs). While traditional weather classification relies on sensor data or manual observation, this project demonstrates that visual cues from weather imagery can be used effectively to train a deep learning model that predicts weather types such as cloudy, foggy, rainy, snowy, and sunny.

---

## Project Overview

The project focuses on building a CNN trained on labeled weather images to perform multi-class classification. The model learns visual patterns in the images, such as cloud shapes, light conditions, and textures that distinguish one weather class from another. To improve performance and reduce overfitting, data augmentation and dropout regularization are used. Additionally, a ResNet18 model was implemented using transfer learning, which achieved significantly higher accuracy compared to the custom CNN while maintaining a similar training time.

Key steps include:
- **Image preprocessing**: Resize, normalize, and augment input images
- **Model design**: CNN with two convolutional layers followed by fully connected layers
- **Training**: Optimized using the Adam optimizer with cross-entropy loss
- **Hyperparameter tuning**: Various combinations of dropout and learning rate tested
- **Evaluation**: Accuracy and confusion matrix on test set

---

## Data Collection

The dataset consists of labeled folders of weather images, with classes such as `cloudy`, `foggy`, `rain`, `shine`, and `snow`. The images were downloaded from publicly available repositories and vary in size and quality. To ensure consistency during training, all images are resized to 160x160 resolution and normalized using ImageNet mean and standard deviation values.

---

## Methodology

1. **Preprocessing**:
   - Resize all images to a uniform shape
   - Apply random horizontal flips and rotations for augmentation
   - Normalize using standard mean and std values

2. **Model Architecture**:
   - Two convolutional layers and max pooling
   - Batch normalization for regularization
   - Fully connected layers for classification
   - Dropout applied after dense layers
   

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

## Conclusion

Transfer learning is a powerful technique in image classification, where we freeze the layers of a pretrained model (here ResNet-18) and train only the final layer. This approach achieves significantly better results than a custom CNN, as demonstrated in the Jupyter notebook.