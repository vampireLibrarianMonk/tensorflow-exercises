# * json: Is essential for reading and decoding JSON data from a file-like object, converting it into Python data
# structures like dictionaries or lists. Renaming it to json_load can help in avoiding naming conflicts or provide a
# more descriptive function name in the context of the code.
from json import load as json_load
# * keras.layers: This component of Keras provides a wide array of layers for building neural networks, including
#   convolutional layers, pooling layers, dense (fully connected) layers, and more. These layers are the building blocks
#   of neural networks and can be stacked to create complex architectures tailored to specific machine learning tasks.

# Tensorflow Keras
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
    # Dataset reference
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    url_request.urlretrieve(url, 'sarcasm.json')

    # Load the json file and map each headline to whether or not it is sarcastic or not
    sentences = []
    labels = []
    with open('sarcasm.json', 'r') as file:
        data = json_load(file)

        for item in data:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

    # Set the maximum number of words to keep, based on word frequency. Only the most common `vocab_size` words will be
    # kept.
    vocab_size = 1000

    # Initialize a Tokenizer object with a specified vocabulary size.
    # `num_words=vocab_size` tells the tokenizer to only use the `vocab_size` most common words.
    # `oov_token="<OOV>"` sets the token to be used for out-of-vocabulary words. Words not seen in the training data
    # will be replaced with "<OOV>".
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

    # Updates the internal vocabulary of the tokenizer based on the list of `sentences`.
    # This method processes each sentence, tokenizes it, and updates the word index based on the words found in the
    # sentence.
    tokenizer.fit_on_texts(sentences)

    # Convert the list of sentences into sequences of integers. Each integer corresponds to a word's index in the
    # tokenizer's word index.
    # This transformation is based on the vocabulary established by the tokenizer during its fit on the training texts.
    sequences = tokenizer.texts_to_sequences(sentences)

    # Set the dimensionality of the embedding layer. This number represents the size of the vector space in which words
    # will be embedded.
    # It defines how many factors or dimensions will be used to represent each word in this space.
    embedding_dim = 10000

    # Set the maximum length of sequences. If a sequence is shorter than this length, it will be padded with zeros.
    # If it's longer, it will be truncated to this length. This ensures that all sequences have the same length.
    max_length = 120

    # Define the type of padding to use when sequences are shorter than `max_length`.
    # 'post' means that if a sequence needs padding, zeros will be added to the end of the sequence.
    padding_type = 'post'

    # Pad the sequences to ensure they all have the same length. This is necessary because most deep learning models
    # require inputs to be of the same size. The `pad_sequences` function achieves this by padding shorter sequences
    # (with zeros, by default) or truncating longer ones to the `max_length` specified.
    X = pad_sequences(sequences, maxlen=max_length, padding=padding_type)

    # Convert the list of labels into a NumPy array for efficient computation and compatibility with various machine
    # learning libraries.
    # This is particularly useful because NumPy arrays offer a wide range of mathematical operations and are optimized
    # for performance.
    labels = np_array(labels)

    # Define the size of the training dataset. In this case, the first 20,000 examples will be used for training.
    training_size = 20000

    # Split the feature data into training and testing sets.
    # `X_train` will contain the first `training_size` examples, and `X_test` will contain the remaining examples.
    # This is done by slicing the array `X` using the `training_size` variable.
    train_data, test_data = X[:training_size], X[training_size:]

    # Similarly, split the label data (`labels`) into training and testing sets.
    # train_labels will contain the labels corresponding to the training data, and test_labels will contain the labels
    # for the testing data.
    # The labels are split in the same manner as the feature data, ensuring corresponding features and labels are
    # matched in both training and testing sets.
    train_labels, test_labels = labels[:training_size], labels[training_size:]

    # Set the dimensionality of the output space for the Embedding layer. This value determines the size of the
    # embedding vectors.
    output_dim = 32

    # Define the neural network model as a sequential model, which means that each layer has exactly one input tensor
    # and one output tensor.
    model = models.Sequential([
        # The Embedding layer transforms integer-encoded vocabulary into dense vector embeddings.
        # * `input_dim` is the size of the vocabulary, which should be `vocab_size + 1` because index 0 is reserved for
        # padding.
        # * `output_dim` is the dimension of the dense embedding.
        # * `input_length` is the length of input sequences.
        layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_length),

        # Dropout layer randomly sets input units to 0 with a frequency of 0.2 at each step during training, which helps
        # prevent overfitting.
        layers.Dropout(0.2),

        # Conv1D layer for convolution over the 1D sequence. It uses 32 filters and a kernel size of 5.
        # The activation function 'relu' introduces non-linearity to the learning process, allowing the model to learn
        # more complex patterns.
        layers.Conv1D(32, 5, activation='relu'),

        # MaxPooling1D layer reduces the dimensionality of the input by taking the maximum value over a window (of size
        # 4 here) for each dimension along the features' axis.
        layers.MaxPooling1D(pool_size=4),

        # Bidirectional LSTM layer processes the sequence both forwards and backwards (bidirectionally) with 64 units.
        # `return_sequences=True` makes the layer return the full sequence of outputs for each input, necessary for
        # stacking with other sequence-processing layers.
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),

        # GlobalAveragePooling1D layer computes the average of the input's dimensions, reducing its dimensionality and
        # preparing it for the dense layer.
        layers.GlobalAveragePooling1D(),

        # Dense layer with 128 neurons and 'relu' activation function, introducing another level of non-linearity and
        # allowing the network to learn more complex representations.
        layers.Dense(128, activation='relu'),

        # Another Dropout layer, this time with a higher dropout rate of 0.5, to further mitigate the risk of
        # overfitting, especially given the larger number of neurons in the preceding Dense layer.
        layers.Dropout(0.5),

        # Final Dense output layer with a single neuron, using the 'sigmoid' activation function to output a value
        # between 0 and 1,
        # which can be interpreted as the probability of the input being in a particular class (e.g., sarcastic or not
        # sarcastic in this context).
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model by specifying the optimizer, loss function, and metrics to monitor.
    # * 'adam' optimizer is a popular choice for many types of neural networks due to its adaptive learning rate
    # capabilities.
    # * 'binary_crossentropy' is suitable for binary classification problems. It measures the performance of a
    # classification model whose output is a probability value between 0 and 1.
    # * 'metrics=['accuracy']': explained inline with other possibilities
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            # 'accuracy': Computes the accuracy rate across all predictions for classification problems.
            # It's the number of correct predictions divided by the total number of predictions.
            'accuracy',

            # 'precision': Measures the proportion of true positive predictions in the positive predictions made by the
            # model.
            # It is particularly useful when the cost of a false positive is high.
            # 'precision' = True Positives / (True Positives + False Positives)

            # 'recall': Measures the proportion of true positive predictions in the actual positive labels.
            # It is important when the cost of a false negative is high.
            # 'recall' = True Positives / (True Positives + False Negatives)

            # 'f1-score': Harmonic mean of precision and recall. It's a better measure than accuracy for imbalanced
            # classes.
            # 'f1-score' = 2 * (precision * recall) / (precision + recall)

            # 'auc': Area Under the ROC Curve. AUC provides an aggregate measure of performance across all
            # classification thresholds.
            # One way of interpreting AUC is as the probability that the model ranks a random positive example more
            # highly than a random negative example.

            # 'mean_squared_error' or 'mse': Measures the average of the squares of the errors between actual and
            # predicted values.
            # It is used for regression problems.

            # 'mean_absolute_error' or 'mae': Measures the average of the absolute differences between actual and
            # predicted values.
            # It provides a linear score without direction, meaning the lower the better, regardless of under or
            # over forecasting.

            # 'mean_absolute_percentage_error' or 'mape': Measures the average of the absolute percentage errors by
            # comparing the prediction with the actual value.
            # This can be more interpretable in terms of percentage, but can be skewed by small denominators.
        ]
    )

    # Set the number of epochs, which is the number of complete passes through the training dataset.
    epochs = 10
    # Set the batch size, which is the number of training examples utilized in one iteration.
    batch_size = 32

    # Fit the model to the training data.
    # * 'train_data' and 'train_labels' are the features and labels for the training dataset, respectively.
    # * 'epochs=epochs' specifies how many times the learning algorithm will work through the entire training dataset.
    # * 'batch_size=batch_size' determines the number of samples per gradient update for training.
    # * 'validation_data=(test_data, test_labels)' provides the validation dataset that the model will evaluate its
    # performance on at the end of each epoch.
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))

    # Evaluate the model's performance on the test dataset, which it has not seen during training.
    # This function returns the loss value and metrics (accuracy in this case) for the model in test mode.
    loss, accuracy = model.evaluate(test_data, test_labels)

    # Print the test accuracy, multiplying by 100 to convert from a proportion to a percentage.
    # The formatted string uses {:.2f} to round the accuracy to two decimal places.
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
    created_model = create_model()

    # Save the model
    created_model.save("../models/exercise_2.h5")
