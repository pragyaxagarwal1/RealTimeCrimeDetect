# Import required libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Define paths and actions
DATA_FOLDER = "action_data"  # Folder where CSVs are stored
ACTIONS = ["kick", "punch", "idle"]  # List of actions

# Define variables
X, y = [], []

# Load all CSV files for each action and merge data
for action in ACTIONS:
    action_path = DATA_FOLDER
    file_count = 0

    # Loop through all files in the directory
    for filename in os.listdir(action_path):
        if filename.startswith(action) and filename.endswith(".csv"):
            file_path = os.path.join(action_path, filename)
            df = pd.read_csv(file_path)

            # Extract landmark data and ignore header row
            for _, row in df.iterrows():
                landmarks = row[1:].values.astype(float)  # Skip 'action' column
                X.append(landmarks)
                y.append(ACTIONS.index(action))
            
            file_count += 1
    
    print(f"✅ Loaded {file_count} files for action '{action}'.")

# Convert data to NumPy arrays
X = np.array(X)
y = np.array(y)

# Reshape X to fit CNN input shape: (samples, 33 landmarks, 3 coordinates)
X = X.reshape(-1, 33, 3)

# Normalize landmark data (optional but improves performance)
X = X / np.max(X)

# One-hot encode target labels
y = to_categorical(y, num_classes=len(ACTIONS))

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data shape: X_train: {X_train.shape}, X_test: {X_test.shape}")

# Define CNN model
model = Sequential()
model.add(Conv1D(64, 3, activation="relu", input_shape=(33, 3)))
model.add(Conv1D(128, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(ACTIONS), activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
print("🚀 Training model...")
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# Save the trained model
MODEL_PATH = "action_model.h5"
model.save(MODEL_PATH)

print(f"✅ Model training complete and saved as '{MODEL_PATH}'.")
