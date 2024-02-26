# Simple CNN for MNIST Classification

This repository contains a PyTorch implementation of a simple Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The code includes model training on the MNIST dataset and provides an example of using the trained model for predictions on new images.

## Model Architecture
The CNN architecture consists of three convolutional layers followed by max-pooling and ReLU activation functions. The fully connected layers at the end perform the final classification.
The model is trained on the MNIST dataset using the Adam optimizer and cross-entropy loss.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)

## Sample Output
Upon running the provided scripts, you should see an output similar to the following:

Test accuracy: 0.9784

The model predicts that the image belongs to class 8.

## Note
- This code assumes that you have the MNIST dataset available and properly set up.
- The training script (train_model.py) trains the model for a single epoch for simplicity, and you may adjust the num_epochs variable for further training.
- Ensure that you replace 'path/to/your/image.png' with the actual path to your image.

  Happy Coding!
