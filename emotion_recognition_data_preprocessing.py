import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path to the extracted folder
data_dir = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\train"
categories = os.listdir(data_dir)
img_size = 48  # Size of images

X = []
y = []

# Function to extract Local Binary Pattern (LBP) features from images
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')  # Uniform LBP
    # Compute the histogram of LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    # Normalize the histogram
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

# Load and preprocess images
for idx, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            img = cv2.resize(img, (img_size, img_size))  # Resize image
            lbp_hist = extract_lbp_features(img)  # Extract LBP features
            X.append(lbp_hist)
            y.append(idx)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(categories))  # One-hot encode labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model in .h5 format
model.save("emotion_recognition_model.h5")

print("Model training complete! Model saved as 'emotion_recognition_model.h5'.")
