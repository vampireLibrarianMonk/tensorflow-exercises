import numpy as np
import tensorflow as tf

# Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data by normalizing the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Use only 10 elements for demonstration
train_images, train_labels = train_images[:10], train_labels[:10]
test_images, test_labels = test_images[:10], test_labels[:10]

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Convert the trained model to TensorFlow Lite format with dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Save the quantized model to file
tflite_model_file = 'quantized_model.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_quantized_model)

print("Quantized model is saved as 'quantized_model.tflite'.")

