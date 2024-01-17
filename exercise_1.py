# Importing necessary libraries

# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and
# matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# TensorFlow is an open-source machine learning library developed by Google. It's used for both research and production
# at Google.
import tensorflow as tf

# Importing specific modules from keras, which is now part of TensorFlow
# Callbacks are utilities called at certain points during model training. EarlyStopping stops training when a monitored
# metric has stopped improving, and ModelCheckpoint saves the model after every epoch.
from keras.callbacks import EarlyStopping, ModelCheckpoint
# load_model is used to load a saved model. Sequential is a linear stack of layers.
from keras.models import load_model, Sequential
# Dense is a standard layer type that is used in many neural networks.
from keras.layers import Dense


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
        filepath='best_model.h5',  # File path to save the model
        save_best_only=True,  # Save only the model that has the best performance on the monitored metric
        monitor=monitor_metric,  # Metric to monitor
        mode='min'  # The training will aim to minimize the monitored metric
    )

    # Compile the model
    model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=tf.keras.losses.mean_squared_error,
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
    saved_model = load_model('best_model.h5')  # Load the model saved by ModelCheckpoint

    # Using the model for prediction
    predicted_depth = 100
    base_x = np.arange(-predicted_depth, predicted_depth + 1, 10)  # New data for prediction
    new_x_values = base_x.reshape(-1, 1)  # Reshaping data for prediction
    predicted_y = saved_model.predict(new_x_values)  # Making predictions

    # Show the new dataset and the associated predictions
    print("Predicted y for x =", new_x_values.flatten(), ":", predicted_y.flatten())

    return model_1  # Returning the trained model


if __name__ == '__main__':
    model = create_model()
    model.save("exercise_1_model.h5")
