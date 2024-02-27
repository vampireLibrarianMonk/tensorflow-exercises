# Importing necessary libraries

# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and
# matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# Importing specific modules from keras, which is now part of TensorFlow
# Callbacks are utilities called at certain points during model training. EarlyStopping stops training when a monitored
# metric has stopped improving, and ModelCheckpoint saves the model after every epoch.
from keras.callbacks import EarlyStopping, ModelCheckpoint
# load_model is used to load a saved model. Sequential is a linear stack of layers.
from keras.models import Sequential
# Dense is a standard layer type that is used in many neural networks.
from keras.layers import Dense

# TensorFlow Lite provides tools and classes for converting TensorFlow models into a highly optimized format suitable
# for deployment on mobile devices, embedded systems, or other platforms with limited computational capacity. This
# module includes functionalities for model conversion, optimization, and inference. By importing `lite`, you gain
# access to the TFLiteConverter class for model conversion, optimization options like quantization, and utilities for
# running TFLite models on target devices.
from tensorflow import lite

# Define a simple sequential model with a single Dense (fully connected) layer.
# The model will have a single input feature and a linear activation function.
model = Sequential([
    Dense(units=1, input_shape=[1], activation='linear')
])

# Compile the model with the Stochastic Gradient Descent (SGD) optimizer.
# Use mean squared error as the loss function, suitable for regression problems.
model.compile(optimizer='sgd', loss='mean_squared_error')

# Create dummy data for training.
# `xs` represents the input features, and `ys` represents the target outputs.
# These arrays are used to train the model to learn the relationship y = 2x - 1.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model on the dummy data.
# We specify the number of iterations over the entire dataset (epochs) and suppress the training log (verbose=0).
model.fit(xs, ys, epochs=100, verbose=0)

# Output a message indicating the completion of training.
print("Model trained.")

# Save the trained model to a file in Hierarchical Data Format version 5 (HDF5).
# This allows the model to be loaded and used later without retraining.
model.save('../models/exercise_0.h5')

# Convert the trained model into the TensorFlow Lite format.
# This step prepares the model for deployment on mobile or embedded devices by reducing its size and potentially
# improving inference speed.
converter = lite.TFLiteConverter.from_keras_model(model)

# Apply default optimizations for the conversion process, including quantization.
# Quantization reduces the precision of the model's weights and activations, which can decrease size and increase
# inference speed with minimal impact on accuracy.
converter.optimizations = [lite.Optimize.DEFAULT]

# Convert the model to its TensorFlow Lite version with applied optimizations.
tflite_model = converter.convert()

# Save the converted (and possibly quantized) TensorFlow Lite model to a file.
# The model is now ready to be deployed on compatible devices.
with open('../models/exercise_0.tflite', 'wb') as f:
    f.write(tflite_model)

# Output a message indicating the TensorFlow Lite model has been saved successfully.
print("Quantized model saved.")
