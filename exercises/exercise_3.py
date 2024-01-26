# German Traffic Sign Recognition Benchmark GTSRB
# https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

# TensorFlow is an open-source machine learning library developed by Google. It's used for both research and production
# at Google.
# * cast: This function is used to change the data type of tensor. For example, it can convert a tensor from one type
#   (like int32) to another (like float32). This is particularly useful in scenarios where you need to ensure
#   consistency in data types for computational operations in TensorFlow.
from tensorflow import cast
# * data: Is primarily used for data preprocessing and pipeline building. It offers tools for
#   reading and writing data in various formats, transforming it, and making it ready for machine learning models.
#   Efficient data handling is crucial in machine learning workflows, and TensorFlow's data module simplifies this
#   process significantly.
from tensorflow import data
# * float32: A data type in TensorFlow, representing a 32-bit floating-point number. It's widely used in machine
#   learning as it provides a good balance between precision and computational efficiency. float32 is often the default
#   data type for TensorFlow's neural network weights and other floating-point computations.
from tensorflow import float32
# * keras, originally an independent neural network library, now integrated into TensorFlow, simplifies the creation and
#   training of deep learning models. Keras is known for its user-friendliness and modular approach, allowing for easy
#   and fast prototyping. It provides high-level building blocks for developing deep learning models while still
#   enabling users to dive into lower-level operations if needed.
from tensorflow import keras
# * tensorflow.python.client: Provides functionalities to query the properties of the hardware devices TensorFlow can
#   access. Specifically, this module is often used to list and get detailed information about the system's available
#   CPUs, GPUs, and other hardware accelerators compatible with TensorFlow.
from tensorflow.python.client import device_lib
# * image: This module in TensorFlow contains various functions and utilities for image processing. It includes tools
#   for tasks like image resizing, color adjustment, and image augmentation - operations that are crucial in many
#   computer vision applications. The image module helps in preparing image data to be fed into machine learning models.
from tensorflow import image as tensor_image
# * keras.layers: This component of Keras provides a wide array of layers for building neural networks, including
#   convolutional layers, pooling layers, dense (fully connected) layers, and more. These layers are the building blocks
#   of neural networks and can be stacked to create complex architectures tailored to specific machine learning tasks.
from keras import layers
# * keras.models: This module in Keras is essential for creating neural network models. It includes classes like
#   Sequential and the Functional API for building models. The Sequential model is straightforward, allowing layers to
#   be added in sequence, suitable for simple architectures. The Functional API, on the other hand, provides greater
#   flexibility for creating complex models with advanced features like shared layers and multiple inputs/outputs.
#   Both types enable comprehensive model management, including training, evaluation, and saving/loading
#   functionalities, making them versatile for a wide range of deep learning applications.
from keras import models
# * keras.callbacks: The keras.callbacks module offers a set of tools that can be applied during the training process of
#   a model. These callbacks are used for various purposes like monitoring the model's performance in real-time, saving
#   the model at certain intervals, early stopping when the performance plateaus, adjusting learning rates, and more.
#   They are crucial for enhancing and controlling the training process, allowing for automated and optimized model
#   training. Callbacks like ModelCheckpoint, EarlyStopping, TensorBoard, and ReduceLROnPlateau are commonly used for
#   efficient model training and fine-tuning.
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Versioning sourcing
from tensorflow import __version__ as tf_version

# Is used for opening and reading URLs, primarily used for fetching data across the web. It allows you to send HTTP and
# other requests, access web pages, download data, and interact with APIs.
from urllib import request

# Provides tools for creating new ZIP archives, extracting files from existing archives, adding files to existing
# archives, and more.
from zipfile import ZipFile

# Regular Expressions
# 1. search: This function is used to perform a search for a pattern in a string and returns a match object if the
# pattern is found, otherwise None. It's particularly useful for string pattern matching and extracting specific
# segments from text.
from re import search

# Key aspects of 'check_output':
# 1. **Process Execution**: The 'check_output' function is used to run a command in the subprocess/external process and
#    capture its output. This is especially useful for running system commands and capturing their output directly
#    within a Python script.
# 2. **Return Output**: It returns the output of the command, making it available to the Python environment. If the
#    called command results in an error (non-zero exit status), it raises a CalledProcessError.
# 3. **Use Cases**: Common use cases include executing a shell command, reading the output of a command, automating
#    scripts that interact with the command line, and integrating external tools into a Python workflow.
# Example Usage:
# Suppose you want to capture the output of the 'ls' command in a Unix/Linux system. You can use 'check_output' like
# this:
# output = check_output(['ls', '-l'])
from subprocess import check_output
# Key aspects of 'CalledProcessError':
#  1. Error Handling: CalledProcessError is an exception raised by check_output when the command it tries to execute
#   returns a non-zero exit status, indicating failure. This exception is particularly useful for error handling in
#   scripts where the success of an external command is crucial.
#  2. Exception Details: The exception object contains information about the error, including the return code, command
#  executed, and output (if any). This aids in debugging by providing clear insights into why the external command
#  failed.
#  3. Handling the Exception: In practical use, it is often caught in a try-except block, allowing the script to respond
#  appropriately to the failure of the external command, like logging the error or trying a fallback operation.
from subprocess import CalledProcessError


def download_and_extract_data():
    # Define the URL of the zip file containing German traffic signs data
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip'

    # Download the zip file from the URL and save it locally as 'germantrafficsigns.zip'
    request.urlretrieve(url, 'germantrafficsigns.zip')

    # Open the downloaded zip file in read mode
    with ZipFile('germantrafficsigns.zip', 'r') as zip_ref:
        # Extract all contents of the zip file to the current working directory
        zip_ref.extractall()


def preprocess(image, label, img_shape=30):
    """
    Preprocess the given image and label for machine learning model input.

    This function performs standard image preprocessing steps: resizing the image
    to a square of a specified size (default is 30x30 pixels) and normalizing its
    pixel values. The normalization involves converting pixel values to a float32
    data type and scaling them to a range of 0 to 1. The function returns the
    processed image and its associated label.

    Parameters:
    image (tensor): The input image to be processed.
    label : The label associated with the image.
    img_shape (int, optional): The size to which the image will be resized,
                               given as the length of one side of the square
                               image. Default is 30 pixels.

    Returns:
    tuple: A tuple containing the processed image and its label.
    """
    # Resize the input image to a specified size (default 30x30 pixels)
    image = tensor_image.resize(image, [img_shape, img_shape])

    # Normalize the image by casting its pixel values to float32 and scaling between 0 and 1
    image = cast(image, float32) / 255.0

    # Return the processed image and the label
    return image, label


def create_model():
    """
    Create and train a convolutional neural network model on traffic sign images.

    This function first downloads and extracts a dataset of traffic sign images.
    It then defines and compiles a convolutional neural network (CNN) model,
    and trains this model on the downloaded dataset. The dataset is divided
    into training and validation sets, with each image preprocessed using the
    'preprocess' function. The CNN model consists of two convolutional blocks
    followed by dense layers, and it is compiled with categorical crossentropy
    as the loss function and Adam optimizer.

    Returns:
    model: The trained TensorFlow Keras model.
    """

    # Download and extract data
    download_and_extract_data()

    # Batch size for training and validation
    batch_size = 32

    # Create training dataset
    train_data = keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        label_mode='categorical',
        image_size=(30, 30),
        batch_size=batch_size)

    # Create validation dataset
    test_data = keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        label_mode='categorical',
        image_size=(30, 30),
        batch_size=batch_size)

    # Preprocess and optimize datasets
    train_data = train_data.map(
        preprocess, num_parallel_calls=data.experimental.AUTOTUNE).prefetch(
        data.experimental.AUTOTUNE)
    test_data = test_data.map(
        preprocess, num_parallel_calls=data.experimental.AUTOTUNE)

    # Define the CNN model architecture
    inline_model = models.Sequential([
        # Input layer - specifies the shape of the input data (30x30 pixels with 3 color channels)
        layers.InputLayer(input_shape=(30, 30, 3)),

        # First Convolutional block
        # Conv2D layer with 32 filters, each of size 3x3, using ReLU activation function
        layers.Conv2D(32, 3, activation='relu', padding='same'),

        # MaxPooling layer with a 2x2 pool size to reduce spatial dimensions (height and width)
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional block
        # Conv2D layer with 64 filters, each of size 3x3, using ReLU activation function
        layers.Conv2D(64, 3, activation='relu', padding='same'),

        # MaxPooling layer with a 2x2 pool size for further spatial dimension reduction
        layers.MaxPooling2D((2, 2)),

        # Flatten the output from the convolutional layers to create a single long feature vector
        layers.Flatten(),

        # Dense layers
        # First dense layer with 128 nodes and ReLU activation function
        layers.Dense(128, activation='relu'),

        # Output dense layer with 43 nodes (assumed number of classes) and softmax activation function
        # Softmax is used for multi-class classification
        layers.Dense(43, activation='softmax')
    ])

    # Compile the model
    inline_model.compile(
        # Set the loss function to 'categorical_crossentropy' for multi-class classification
        loss="categorical_crossentropy",

        # Use the Adam optimizer, a popular choice that combines benefits of other approaches
        # like AdaGrad and RMSProp, suitable for a wide range of tasks and generally good
        # at handling sparse gradients on noisy problems
        optimizer=keras.optimizers.Adam(),

        # Track 'accuracy' as the metric for evaluating the model's performance
        # This metric will be used to monitor the training and testing steps
        metrics=["accuracy"]
    )

    # EarlyStopping callback with your desired parameters. For instance, you might want to stop training if the
    # validation loss doesn't improve for 5 consecutive epochs. You can also set a restore_best_weights parameter
    # to true, which will keep the model weights from the epoch with the best value of the monitored quantity:
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
    )

    # Create an EarlyStopping callback
    early_stopping = EarlyStopping(
        # This parameter specifies the metric to be monitored. In this case, it's accuracy, which is the accuracy on
        # the validation dataset. The callback will monitor the accuracy at the end of each epoch.
        monitor='accuracy',
        #  Patience is the number of epochs with no improvement after which training will be stopped. Setting patience=5
        #  means the training process will continue for 5 more epochs even after detecting a stop in improvement. This
        #  allowance is beneficial to rule out the possibilities of random fluctuations in training metrics.
        patience=5,
        # When set to True, the model weights will be restored to the weights of the epoch with the best value of the
        # monitored metric. This ensures that even if the model's performance degrades for a few epochs before training
        # is stopped, you will retain the best-performing version of the model.
        restore_best_weights=True
    )

    # Define the model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        # This parameter specifies the path where the model or weights should be saved. The file is saved in the HDF5
        # format, which is a popular format for storing large numerical objects like weights of a neural network.
        filepath='../models/exercise_2.h5',
        # When set to True, the callback will only save the model when the monitored metric has improved, meaning the
        # current model is better than any previous model.
        save_best_only=True,  # Save only the best model
        # This parameter specifies which metric to monitor. Here, it's set to monitor the mean absolute error (mae).
        # The callback will track the changes in this metric after each epoch.
        monitor='mae',  # Monitor the mean absolute error
        # It is particularly useful in regression problems or situations where you want to account for the average
        # magnitude of errors in predictions, without considering their direction.
        #
        # Other commonly monitored metrics include:
        # * "val_loss": Monitors the loss on the validation dataset. It's useful when your primary objective is to
        # minimize loss.
        # * "val_accuracy": Tracks accuracy on the validation set. Ideal for classification problems where accuracy is
        # the main concern.
        # * "loss": Observes the model's total loss during training. This is more focused on the model's performance on
        # the training dataset.
        # * "accuracy": Similar to val_accuracy, but for the training set.
        #
        # Metrics for Multi-output Models: If your model has multiple outputs, Keras will add prefixes to the metric
        # names based on the output names.
        mode='min'  # The lower the MAE, the better
        # Other types of mode
        # * "auto": Automatically infers from the name of the monitored quantity. For instance, it sets to "max" for
        # metrics like "accuracy", and to "min" for metrics like "loss".
        # * "max": The model is saved when the monitored quantity increases. This mode is suitable for metrics where a
        # higher value indicates better performance, such as "accuracy".
    )

    # Train the model
    inline_model.fit(
        # Use 'train_data' as the input data for training the model
        train_data,

        # Set the number of iterations over the entire dataset to 20
        # 'epochs' defines how many times the learning algorithm will work through the entire training dataset
        epochs=20,

        # Define the number of steps to yield from the generator before declaring one epoch finished
        # and starting the next epoch. Here it is set to the length of 'train_data'
        # This ensures that the model sees all training data in each epoch
        steps_per_epoch=len(train_data),

        # Use 'test_data' for validation during the training process
        # Validation data is used to evaluate the loss and any model metrics at the end of each epoch
        validation_data=test_data,

        # The number of steps to yield from the validation generator at the end of every epoch
        # Here, it's incorrectly set to the length of 'train_data', which might be a mistake
        # Typically, this should be set to the length of 'test_data' or 'validation_data'
        validation_steps=len(train_data),

        # Reference the early stopping function
        callbacks=[early_stopping]
    )

    # Evaluate the model on validation data
    inline_model.evaluate(test_data)

    # Return the trained model
    return inline_model


def print_gpu_info():
    # Undocumented Method
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    # Get the list of all devices
    devices = device_lib.list_local_devices()

    for device in devices:
        if device.device_type == 'GPU':
            # Extract the physical device description
            desc = device.physical_device_desc

            # Use regular expressions to extract the required information
            gpu_id_match = search(r'device: (\d+)', desc)
            name_match = search(r'name: (.*?),', desc)
            compute_capability_match = search(r'compute capability: (\d+\.\d+)', desc)

            if gpu_id_match and name_match and compute_capability_match:
                gpu_id = gpu_id_match.group(1)
                gpu_name = name_match.group(1)
                compute_capability = compute_capability_match.group(1)

                # Convert memory limit from bytes to gigabytes and round it
                memory_limit_gb = round(device.memory_limit / (1024 ** 2))

                print(
                    f"\tGPU ID {gpu_id} --> {gpu_name} --> "
                    f"Memory Limit {memory_limit_gb} MB --> "
                    f"Compute Capability {compute_capability}")


# if-statement will execute only if the script is the main program being run.
# This is a common practice in Python to structure scripts for both stand-alone use and importable functionality.
if __name__ == '__main__':
    print("Hardware Found:")
    # GPU
    print_gpu_info()

    print("Software Versions:")

    # CUDA
    try:
        # Execute the 'nvcc --version' command and decode the output
        nvcc_output = check_output("nvcc --version", shell=True).decode()

        # Use regular expression to find the version number
        match = search(r"V(\d+\.\d+\.\d+)", nvcc_output)
        if match:
            cuda_version = match.group(1)
            print("\tCUDA Version", cuda_version)
        else:
            print("\tCUDA Version not found")

    except CalledProcessError as e:
        print("Error executing nvcc --version:", e)

    # NVIDIA Driver
    try:
        # Execute the nvidia-smi command and decode the output
        nvidia_smi_output = check_output("nvidia-smi", shell=True).decode()

        # Split the output into lines
        lines = nvidia_smi_output.split('\n')

        # Find the line containing the driver version
        driver_line = next((line for line in lines if "Driver Version" in line), None)

        # Extract the driver version number
        if driver_line:
            driver_version = driver_line.split('Driver Version: ')[1].split()[0]
            print("\tNVIDIA Driver:", driver_version)

            # Extract the maximum supported CUDA version
            cuda_version = driver_line.split('CUDA Version: ')[1].strip().replace("|", "")
            print("\tMaximum Supported CUDA Version:", cuda_version)
        else:
            print("\tNVIDIA Driver Version or CUDA Version not found.")

    except Exception as e:
        print("Error fetching NVIDIA Driver Version or CUDA Version:", e)

    # TensorFlow
    print("\tTensorFlow:", tf_version)

    print("\n")

    # Create the model
    model = create_model()

    # Save the model
    model.save("../models/exercise_2.h5")
