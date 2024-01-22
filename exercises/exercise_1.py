# Importing necessary libraries

# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and
# matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# TensorFlow is an open-source machine learning library developed by Google. It's used for both research and production
# at Google.
# * keras: Originally an independent neural network library, now integrated into TensorFlow, simplifies the creation and
#   training of deep learning models. Keras is known for its user-friendliness and modular approach, allowing for easy
#   and fast prototyping. It provides high-level building blocks for developing deep learning models while still
#   enabling users to dive into lower-level operations if needed.
from tensorflow import keras
# * tensorflow.python.client: Provides functionalities to query the properties of the hardware devices TensorFlow can
#   access. Specifically, this module is often used to list and get detailed information about the system's available
#   CPUs, GPUs, and other hardware accelerators compatible with TensorFlow.
from tensorflow.python.client import device_lib

# Versioning sourcing
from tensorflow import __version__ as tf_version

# Importing specific modules from keras, which is now part of TensorFlow
# Callbacks are utilities called at certain points during model training. EarlyStopping stops training when a monitored
# metric has stopped improving, and ModelCheckpoint saves the model after every epoch.
from keras.callbacks import EarlyStopping, ModelCheckpoint
# load_model is used to load a saved model. Sequential is a linear stack of layers.
from keras.models import load_model, Sequential
# Dense is a standard layer type that is used in many neural networks.
from keras.layers import Dense

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


def create_model():
    # Creating a simple dataset
    depth = 10000
    training_range = np.arange(-depth, depth)  # np.arange creates evenly spaced values within a given interval.

    # This case is the simplest one I could find adding the offset to the training range to create a test range for
    # later prediction.
    offset = 7
    test_range = training_range + offset  # Simple linear relationship for the target variable

    # In the context of neural networks, data types are crucial for managing memory and computational efficiency.
    # float32 is a common data type representing a 32-bit floating-point number.
    # It's widely used in neural network computations for a balance between precision and memory usage.
    digit = 'float32'

    # Reshaping and converting data type for TensorFlow compatibility
    # The -1 tells NumPy to calculate the size of this dimension automatically based on the length of the array and the
    # other given dimension, which is 1. This effectively transforms the array into a two-dimensional array with one
    # column and as many rows as necessary to accommodate all elements.
    x_train = training_range.reshape(-1, 1).astype(digit)
    y_train = test_range.reshape(-1, 1).astype(digit)

    # Building the neural network model
    # Dense layer with a single neuron. Input shape is 1 since our input has only one feature.
    model_1 = Sequential([
        Dense(1, input_shape=(1,))
    ])

    # Setting up the early stopping callback
    # Mean Absolute Error (MAE) is the average of the absolute differences between the predicted values and the actual
    # values. It measures how close the predictions of a model are to the actual outcomes.
    monitor_metric = 'mae'
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,  # Monitor the mean absolute error
        patience=5  # Number of epochs with no improvement after which training will be stopped.
    )

    # Setting up the model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath='../model/exercise_1.h5',  # File path to save the model
        save_best_only=True,  # Save only the model that has the best performance on the monitored metric
        monitor=monitor_metric,  # Metric to monitor
        mode='min'  # The training will aim to minimize the monitored metric
    )

    # Compile the model
    model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                    loss=keras.losses.mean_squared_error,
                    metrics=[monitor_metric])

    # Training the model
    model_1.fit(
        x_train,
        y_train,
        epochs=35,  # The number of times to iterate over the training data arrays
        batch_size=32,  # Number of samples per gradient update
        callbacks=[
            early_stopping_callback,  # Implementing early stopping
            model_checkpoint_callback  # Implementing model checkpoint saving
        ]
    )

    # Loading the best saved model
    saved_model = load_model('../models/exercise_1.h5')  # Load the model saved by ModelCheckpoint

    # Using the model for prediction
    predicted_depth = 100
    base_x = np.arange(-predicted_depth, predicted_depth + 1, 10)  # New data for prediction
    new_x_values = base_x.reshape(-1, 1)  # Reshaping data for prediction
    predicted_y = saved_model.predict(new_x_values)  # Making predictions

    # Show the new dataset and the associated predictions
    print("Predicted y for x =", new_x_values.flatten(), ":", predicted_y.flatten())

    return model_1  # Returning the trained model


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
    model.save("../models/exercise_1.h5")
