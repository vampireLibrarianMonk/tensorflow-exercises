import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Loop through the directory (useful if reading multiple files, e.g., in a Kaggle environment)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the dataset
path = '../data/PET_PRI_GND_DCUS_NUS_W.csv'
df = pd.read_csv(path, delimiter=',')

# Convert 'Date' to datetime and extract day, month, and year as separate columns
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.drop('Date', axis=1, inplace=True)

# Map month numbers to month names
month_mapping = {
    1: 'JAN', 2: 'FEB', 3: 'MARCH', 4: 'APRIL', 5: 'MAY', 6: 'JUNE',
    7: 'JULY', 8: 'AUG', 9: 'SEPT', 10: 'OCT', 11: 'NOV', 12: 'DEC'
}
df['Month'] = df['Month'].map(month_mapping)

# Handle NaN values for numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Replace infinite values with the maximum non-infinite value of the column
for column in numeric_cols:
    if np.isinf(df[column]).any():
        max_val = df[~np.isinf(df[column])][column].max()
        df[column].replace(np.inf, max_val, inplace=True)
        df[column].replace(-np.inf, -max_val, inplace=True)

# Creating dummy variables for categorical data
df = pd.get_dummies(df, columns=['Month', 'Day', 'Year'], drop_first=True)

# Split data into features and target
target = df['D1']
features = df.drop('D1', axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)

# Further split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# Ensure all data is numeric and finite
X_train = X_train.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], np.nan).dropna(axis=1)
X_val = X_val[X_train.columns]  # Ensure X_val has the same columns as X_train

# Convert to float32 for TensorFlow compatibility
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# Define the model
model = Sequential([
    Dense(76, activation='relu', kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
          bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)),
    Dense(200, activation='relu', kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
          bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)),
    Dropout(0.5),
    Dense(1)
])

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)

# Fit the model
model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), batch_size=100, epochs=150, callbacks=[early_stop])

# Plot training and validation loss
losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()

# Ensure X_test contains only numeric columns and handle NaN/infinite values
X_test = X_test.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], np.nan).dropna(axis=1)

# Ensure X_test has the same columns in the same order as X_train
X_test = X_test[X_train.columns]

# Convert to a consistent data type, typically float32
X_test = X_test.astype(np.float32)

# Making predictions
Y_pred = model.predict(X_test)

# Evaluation
results = pd.DataFrame(columns=['MAE', 'MSE', 'R2-score'])
results.loc['Deep Neural Network'] = [
    mean_absolute_error(y_test, Y_pred).round(3),
    mean_squared_error(y_test, Y_pred).round(3),
    r2_score(y_test, Y_pred).round(3)
]

# Display results
print(results)

# Additional metrics
print(f"10% of the mean of the target variable is {np.round(0.1 * target.mean(), 3)}")

# Visualize the performance
results.sort_values('R2-score', ascending=False).style.background_gradient(cmap='Greens', subset=['R2-score'])
