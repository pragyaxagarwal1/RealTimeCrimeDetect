# Import required libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob

# Define action labels
actions = ["kick", "punch", "idle"]

# Initialize empty lists for data and labels
data = []
labels = []

# Check and load multiple CSV files for each action
for action in actions:
    # Get all CSV files for each action (e.g., kick_1.csv, punch_1.csv, etc.)
    file_pattern = f"action_data/{action}_*.csv"
    csv_files = glob.glob(file_pattern)

    # Check if any files are found
    if len(csv_files) == 0:
        print(f"❌ No files found for action: {action}")
        continue

    # Process each CSV file for the current action
    for file_path in csv_files:
        # Read CSV and process data
        df = pd.read_csv(file_path, header=None)  # No header in the new format

        # Debug: Print file shape
        print(f"✅ Loaded: {file_path} | Shape: {df.shape}")

        # Process each row with error handling for invalid data
        for i in range(len(df)):
            row = df.iloc[i].values

            try:
                # Convert all values to float
                landmark_data = np.array(row, dtype=np.float32)

                # Dynamically reshape the landmark data based on its length
                num_landmarks = len(landmark_data) // 3
                if len(landmark_data) % 3 == 0 and num_landmarks == 33:  # Expected 33 landmarks (x, y, z)
                    landmark_data = landmark_data.reshape(33, 3)
                    data.append(landmark_data)
                    labels.append(action)
                else:
                    print(f"⚠️ Skipping invalid row in {file_path}: Incorrect landmark data")

            except ValueError:
                print(f"❌ Skipping row {i + 1} in {file_path}: Invalid data")

# Check if data is loaded properly
if len(data) == 0 or len(labels) == 0:
    print("❌ No valid data found. Check your CSV files.")
    exit()

# Convert to NumPy arrays
X = np.array(data, dtype=np.float32)
y = np.array(labels)

# Debug: Check landmark shape before reshaping
print(f"✅ Initial X shape: {X.shape}")

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Debug: Print final dataset shapes
print(f"✅ X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"✅ X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"✅ Unique labels: {set(labels)}")

# Define LSTM-based model
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(33, 3)))
model.add(Dense(32, activation="relu"))
model.add(Dense(len(actions), activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model with reduced epochs and smaller batch size for stability
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=16)

# Save trained model
model.save("action_model.h5")
print("✅ Model trained and saved as action_model.h5!")
