# Convolutional Neural Network using Keras

This project implements a basic Convolutional Neural Network (CNN) model using the Keras library for binary image classification tasks. The model is built using `Sequential` API, with convolutional, pooling, and fully connected layers.

## Overview

This model processes 100x100 RGB images and classifies them into two categories. The architecture is designed to extract features using convolutional layers, followed by pooling and dense layers for classification.

### Model Architecture

1. **Convolutional Layer**:
   - Input shape: `(100, 100, 3)` (100x100 pixel image with 3 channels - RGB)
   - Number of filters: `32`
   - Kernel size: `7x7`
   - Activation: `ReLU`

2. **Max Pooling Layer**:
   - Reduces spatial dimensions to reduce computational complexity.

3. **Flatten Layer**:
   - Flattens the feature maps into a 1D array for fully connected layers.

4. **Fully Connected Layers (Dense Layers)**:
   - Two dense layers with `ReLU` activation for feature extraction.
   - Output layer with `sigmoid` activation for binary classification (output between 0 and 1).

### Code

```python
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential

# Initialize the model
model = Sequential()

# Add Convolutional layer
model.add(
    Convolution2D(
        kernel_size=(7, 7), 
        filters=32, 
        activation='relu',
        strides=(1, 1),
        input_shape=(100, 100, 3)
    )
)

# Add Max Pooling layer
model.add(MaxPool2D())

# Flatten the output
model.add(Flatten())

# Add Fully Connected layers
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Print model summary
model.summary()
