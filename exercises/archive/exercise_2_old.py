"""
Credit to Daniel Bourke for currently the quickest way to work with 10 classes of data.
This script should train a TensorFlow model on the fashion MNIST
dataset to ~90% test accuracy.

It'll save the model to the current directory using the ".h5" extension.

You can use it to test if your local machine is fast enough to complete the
TensorFlow Developer Certification.

If this script runs in under 5-10 minutes through PyCharm, you're good to go.

The models/datasets in the exam are similar to the ones used in this script.
"""
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers
from tensorflow.python.platform import build_info as tf_build_info

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check CUDA version
build_info = tf_build_info.build_info
cuda_version = build_info.get("cuda_version")
print("CUDA version:", cuda_version)

# Check cuDNN version (not directly available, need to check manually or from TensorFlow's documentation)
cudnn_version = build_info.get("cudnn_version")
print("cuDNN version:", cudnn_version)

# Print GPU information using TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Print details of the available GPUs
        for gpu in gpus:
            print("\nGPU Details:", tf.config.experimental.get_device_details(gpu))
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found.")

# Get data
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Convert the labels to one-hot encoding
num_classes = 10  # Number of classes in Fashion MNIST
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

# Example: Printing one-hot encoded labels for the first training sample
print("Original Label:", train_labels[0])  # Original label
print("One-Hot Encoded Label:", train_labels_one_hot[0])  # One-hot encoded label

# Define class names for Fashion MNIST
class_names = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Get a random index
random_index = random.randint(0, len(train_images) - 1)

# Get the random image and its label
random_image = train_images[random_index]
random_label = train_labels[random_index]

# Print the label and its corresponding class name
print("Label:", random_label)
print("Class Name:", class_names[random_label])

# Print the shape of the image
print("Image Shape:", random_image.shape)

# Normalize images (get values between 0 & 1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Check shape of input data
print(f"Print the train image shape {train_images.shape}.")
print(f"Print the train labels shape {train_labels.shape}.")

# Build model
model = tf.keras.Sequential([
    # Reshape inputs to be compatible with Conv2D layer
    layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.Flatten(), # flatten outputs of final Conv layer to be suited for final Dense layer
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(loss="categorical_crossentropy", # if labels aren't one-hot use sparse (if labels are one-hot, drop sparse)
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit model
print("Training model...")
model.fit(x=train_images,
          y=train_labels_one_hot,  # Use one-hot encoded labels
          epochs=10,
          validation_data=(test_images, test_labels_one_hot))  # Use one-hot encoded labels for validation data too

# Evaluate model
print("Evaluating model...")
model.evaluate(test_images, test_labels_one_hot)  # Use one-hot encoded labels for evaluation

# Save model to current working directory
model.save("test_image_model.h5")