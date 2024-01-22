# * json: Is essential for reading and decoding JSON data from a file-like object, converting it into Python data
# structures like dictionaries or lists. Renaming it to json_load can help in avoiding naming conflicts or provide a
# more descriptive function name in the context of the code.
from json import load as json_load
# * keras.layers: This component of Keras provides a wide array of layers for building neural networks, including
#   convolutional layers, pooling layers, dense (fully connected) layers, and more. These layers are the building blocks
#   of neural networks and can be stacked to create complex architectures tailored to specific machine learning tasks.

# Tensorflow Keras
from keras import layers
# * keras.models: This module in Keras is essential for creating neural network models. It includes classes like
#   Sequential and the Functional API for building models. The Sequential model is straightforward, allowing layers to
#   be added in sequence, suitable for simple architectures. The Functional API, on the other hand, provides greater
#   flexibility for creating complex models with advanced features like shared layers and multiple inputs/outputs.
#   Both types enable comprehensive model management, including training, evaluation, and saving/loading
#   functionalities, making them versatile for a wide range of deep learning applications.
from keras import models
# * keras.preprocessing.text: Is instrumental in text preprocessing for deep learning models, as it allows for the
# conversion of text data into a more manageable form, typically sequences of integers, where each integer maps to a
# specific word in the text. This conversion is essential for preparing textual data for training neural network models,
# particularly in natural language processing tasks.
from keras.preprocessing.text import Tokenizer
# * keras.preprocessing.sequence: Is widely used in preparing sequential data, particularly for neural network models in
# natural language processing. It standardizes the lengths of sequences by padding them with zeros or truncating them to
# a specified length, ensuring that all sequences in a dataset have the same length for batch processing.
from keras.preprocessing.sequence import pad_sequences
# * tensorflow.python.client: Provides functionalities to query the properties of the hardware devices TensorFlow can
#   access. Specifically, this module is often used to list and get detailed information about the system's available
#   CPUs, GPUs, and other hardware accelerators compatible with TensorFlow.
from tensorflow.python.client import device_lib

# Versioning sourcing
from tensorflow import __version__ as tf_version
# * is used to import the array function from the NumPy library, but it renames it as np_array for use within the code.
# This function is crucial in NumPy for creating array objects, which are central to the library's operations. These
# arrays are multi-dimensional, efficient, and provide the foundation for a wide range of scientific computing and data
# manipulation tasks in Python. Renaming it to np_array can help avoid naming conflicts or simply provide a shorthand
# that's more convenient for the coder's preferences.
from numpy import array as np_array

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

from urllib import request as url_request


def create_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    url_request.urlretrieve(url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 10000
    output_dim = 32
    max_length = 120

    sentences = []
    labels = []
    with open('sarcasm.json', 'r') as file:
        data = json_load(file)

        for item in data:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

    # Tokenization with specified configurations
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    # Convert texts to sequences of integers
    sequences = tokenizer.texts_to_sequences(sentences)

    # Padding
    padding_type = 'post'
    X = pad_sequences(sequences, maxlen=max_length, padding=padding_type)

    # Convert labels to a numpy array
    labels = np_array(labels)

    # Split the data with a training size of 20000
    training_size = 20000
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = labels[:training_size], labels[training_size:]

    # Model definition
    model = models.Sequential([
        layers.Embedding(input_dim=embedding_dim, output_dim=output_dim, input_length=max_length),
        layers.Dropout(0.2),
        layers.Conv1D(32, 5, activation='relu'),  # New Conv1D layer
        layers.MaxPooling1D(pool_size=4),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),  # New Bi-directional LSTM layer
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),  # Increased number of neurons
        layers.Dropout(0.5),  # Increased dropout rate
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    epochs = 10
    batch_size = 32
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return model


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
    model.save("../models/exercise_2.h5")


