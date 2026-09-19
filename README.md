# CIFAR-10 Image Classification with CNN

A Convolutional Neural Network implemented in PyTorch for
classifying images from the CIFAR-10 dataset.

## Architecture

Input: 32 × 32 RGB image

Conv2D (3 → 12)
ReLU
Max Pooling

Conv2D (12 → 24)
ReLU
Max Pooling

Fully Connected (600 → 120)
ReLU
Fully Connected (120 → 84)
ReLU
Fully Connected (84 → 10)

## Training

Dataset: CIFAR-10
Batch size: 32
Optimizer: SGD
Learning rate: 0.001
Momentum: 0.9
Loss: Cross Entropy
Epochs: 20

## Results

Test accuracy: **[66%]**

## Inference

The `predict.py` script allows the trained model to classify
external RGB images.
