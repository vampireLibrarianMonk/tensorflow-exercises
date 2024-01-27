# Daily Minimum Temperatures Dataset
# Contains daily minimum temperatures from January 1981 to December 1990.

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

# Import the Matplotlib library's pyplot module as 'plt' to enable data visualization through various types of plots
# and charts, providing a MATLAB-like plotting framework for creating static, interactive, and animated visualizations
# in Python.
import matplotlib.pyplot as plt

# * numpy.array is used to import the array function from the NumPy library, but it renames it as np_array for use
# within the code. This function is crucial in NumPy for creating array objects, which are central to the library's
# operations. These arrays are multi-dimensional, efficient, and provide the foundation for a wide range of scientific
# computing and data manipulation tasks in Python. Renaming it to np_array can help avoid naming conflicts or simply
# provide a shorthand that's more convenient for the coder's preferences.
from numpy import array as np_array

# Import `min` and `max` functions from numpy as `np_min` and `np_max` respectively, for performing minimum and
# maximum operations on arrays efficiently.
from numpy import min as np_min
from numpy import max as np_max

# Import the `read_csv` function from pandas for loading and parsing CSV files into DataFrame objects,
# facilitating data manipulation and analysis.
from pandas import read_csv

# Regular Expressions
# 1. search: This function is used to perform a search for a pattern in a string and returns a match object if the
# pattern is found, otherwise None. It's particularly useful for string pattern matching and extracting specific
# segments from text.
from re import search

# Import the `data` module from TensorFlow to utilize efficient data loading, preprocessing,
# and iteration tools for handling datasets, enabling optimized data pipelines for model training and evaluation.
from tensorflow import data

# Import the `load_model` function from Keras to enable loading a pre-trained model from a file,
# allowing for model evaluation, further training, or inference without needing to redefine the model architecture.
from keras.models import load_model

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

# * tensorflow.python.client: Provides functionalities to query the properties of the hardware devices TensorFlow can
#   access. Specifically, this module is often used to list and get detailed information about the system's available
#   CPUs, GPUs, and other hardware accelerators compatible with TensorFlow.
from tensorflow.python.client import device_lib

# Versioning sourcing
from tensorflow import __version__ as tf_version


# This function normalizes a numerical data series by applying min-max scaling. It subtracts the minimum value of the
# series from each element, and then divides by the range of the series (max - min), resulting in a scaled series where
# all values are within the range [0, 1]. The function returns the normalized series along with the original minimum and
# maximum values, which can be useful for reversing the normalization or for normalizing other related data in the same
# way.

def normalize_series(series):
    min_val = np_min(series)  # Find the minimum value in the series
    max_val = np_max(series)  # Find the maximum value in the series
    # Apply min-max scaling to normalize the series to the range [0, 1]
    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series, min_val, max_val


# This function transforms time series data into a format suitable for supervised learning by creating sequences
# of past observations as input features (X) and sequences of future observations as output labels (y).
# `input_data` is the original time series data, `past` is the number of past observations to use for predicting
# the future, and `future` is the number of future observations to predict. The function iterates over the input
# data, creating overlapping windows of `past` observations as features and the subsequent `future` observations
# as labels for each window. It returns two numpy arrays: X containing the input features and y containing the
# corresponding labels, making it easier to train machine learning models for forecasting tasks.

def create_dataset(input_data, past, future):
    X, y = [], []
    # Loop to generate input-output pairs where each input sequence of length `past`
    # is mapped to an output sequence of length `future`.
    for i in range(len(input_data) - past - future + 1):
        X.append(input_data[i:(i + past)])  # Extract past observations as input
        y.append(input_data[(i + past):(i + past + future)])  # Extract future observations as output
    # Convert lists to numpy arrays for efficient numerical computations
    return np_array(X), np_array(y)


# This function defines, compiles, and trains a neural network model for time series forecasting.
# The model uses LSTM layers suited for sequential data like time series. The architecture starts with an LSTM layer
# to process the input sequences, followed by a RepeatVector layer to adjust the network's output dimension to the
# desired number of future time steps. Another LSTM layer captures temporal dependencies in the output sequence, and
# a TimeDistributed Dense layer generates the predicted values for each future time step.
# The model is compiled with the Adam optimizer and mean squared error loss, suitable for regression tasks.
# Training is performed on the provided training data with a specified batch size, and validation is conducted on a
# separate validation dataset. The function returns the trained model, ready for making predictions on new data.

def solution_model(input_train_data, input_valid_data, past, future, input_batch_size):
    # Define the model architecture with LSTM for sequence processing and prediction
    model = models.Sequential([
        layers.LSTM(100, activation='relu', input_shape=(past, 1)),  # Input LSTM layer
        layers.RepeatVector(future),  # Prepare the network to output `future` time steps
        layers.LSTM(100, activation='relu', return_sequences=True),  # Output LSTM layer
        layers.TimeDistributed(layers.Dense(1))  # Dense layer to output predictions for each time step
    ])

    # Compile the model with Adam optimizer and MSE loss function
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on the training dataset with validation on the validation dataset
    model.fit(input_train_data, epochs=20, batch_size=input_batch_size, validation_data=input_valid_data)

    # Return the trained model
    return model


# Function to make predictions using the trained model
def make_predictions(input_model, input_data):
    predictions = input_model.predict(input_data)
    return predictions


# Since the data was normalized, we need to rescale the predictions back to the original temperature scale
def rescale(predictions, min_val, max_val):
    return predictions * (max_val - min_val) + min_val


# This function `print_gpu_info` is designed to display detailed information about the available GPUs on the system.
# It utilizes TensorFlow's `device_lib.list_local_devices()` method to enumerate all computing devices recognized by
# TensorFlow. For each device identified as a GPU, the function extracts and prints relevant details including the GPU's
# ID, name, memory limit (converted to megabytes), and compute capability. The extraction of GPU information involves
# parsing the device's description string using regular expressions to find specific pieces of information. This
# function can be particularly useful for debugging or for setting up configurations in environments with multiple GPUs,
# ensuring that TensorFlow is utilizing the GPUs as expected.
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


# This script is the main entry point for training a time series forecasting model on daily minimum temperatures. The
# dataset is loaded from a public URL, containing daily temperature observations in Melbourne, Australia. The data
# is first normalized to scale the temperature values between 0 and 1, aiding in model training. The normalized data is
# then transformed into a supervised learning problem where the model learns to predict future temperature values based
# on past observations. The dataset is split into training and validation sets, with the training set used to fit the
# model and the validation set to evaluate its performance. The model is defined, compiled, and trained using the LSTM
# architecture suitable for sequential data like time series. After training, the model is saved to a file for future
# use or deployment. The process demonstrates key steps in preparing time series data, building a neural network model
# for forecasting, and training and saving the model.

if __name__ == "__main__":
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

    # Load and prepare the dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    df = read_csv(url, parse_dates=['Date'], index_col='Date')

    # Data summary: The dataset contains date-indexed daily minimum temperatures recorded in Melbourne.

    # Normalize the temperature data
    temperatures = df['Temp'].values
    normalized_temps, min_temp, max_temp = normalize_series(temperatures)

    # Create datasets for training and validation
    n_past = 10
    n_future = 10
    batch_size = 32

    # Normalize data
    X, y = create_dataset(normalized_temps, n_past, n_future)

    # Create split percentage
    split_time = int(len(X) * 0.8)

    # Create train and validation data
    X_train, y_train = X[:split_time], y[:split_time]
    X_valid, y_valid = X[split_time:], y[split_time:]

    # Convert to TensorFlow datasets for efficient loading and batching
    train_data = data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(1)
    valid_data = data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size).prefetch(1)

    # Train the model with the specified architecture and hyperparameters
    model = solution_model(train_data, valid_data, n_past, n_future, batch_size)

    # Save the trained model for later use or deployment
    model.save("../models/exercise_5.h5")

    # Load the trained model
    model = load_model("../models/exercise_5.h5")

    # Use the model to make predictions on the validation dataset
    predicted_temperatures = make_predictions(model, valid_data)

    # Rescale the predicted and actual temperatures
    predicted_temperatures_rescaled = rescale(predicted_temperatures, min_temp, max_temp)
    y_valid_rescaled = rescale(y_valid, min_temp, max_temp)

    # Plot the first N predictions against the actual values for visual comparison
    N = 100  # Number of points to plot
    plt.figure(figsize=(15, 6))
    plt.plot(predicted_temperatures_rescaled[:N, :, 0].flatten(), label="Predicted Temperatures", color='red',
             linestyle='--')
    plt.plot(y_valid_rescaled[:N, :].flatten(), label="Actual Temperatures", color='blue')
    plt.title("Comparison of Predicted and Actual Temperatures")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()
