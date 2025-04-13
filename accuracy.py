import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to the dataset (same location as during training)
data_dir = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\train"
categories = os.listdir(data_dir)
img_size = 48  # Target image size

X = []
y = []

# Load and preprocess images (same as during training)
for idx, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

X = np.array(X)
y = np.array(y)

# Normalize pixel values to range [0, 1]
X = X.astype('float32') / 255.0

# Add an extra dimension for channels (grayscale images have 1 channel)
X = np.expand_dims(X, axis=-1)

# Convert labels to categorical
y = to_categorical(y, num_classes=len(categories))

# Train-test split (same as during training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = load_model('emotion_recognition_model_cnn.keras')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print the test accuracy
print(f"Test Accuracy: {test_acc * 100:.2f}%")
