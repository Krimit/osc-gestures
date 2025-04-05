#from google.colab import files
import os

import pandas as pd
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data
# Load data from CSV
csv_file_path = 'hand_gesture_data.csv'
data = pd.read_csv(csv_file_path)
print("loaded data")

# Separate features and labels
X = data.drop('label', axis=1).values
y = data['label'].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("split to training and testing sets")

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))


# Step 3: Save the Model in HDF5 Format
hdf5_model_path = 'hand_gesture_model.h5'
model.save(hdf5_model_path)
print(f"Model saved to {hdf5_model_path}")

# Step 4: Convert and Save the Model in TFLite Format
# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'hand_gesture_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to {tflite_model_path}")

# Optional: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)