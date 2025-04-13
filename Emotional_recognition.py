import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Path to the extracted dataset
data_dir = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\train"
categories = os.listdir(data_dir)
img_size = 48  # Target image size

X = []
y = []

# Function to extract Local Binary Pattern (LBP) features
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

# Load and preprocess images
for idx, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            lbp_hist = extract_lbp_features(img)
            X.append(lbp_hist)
            y.append(idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(categories))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(categories), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_emotion_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history = model.fit(
    X_train, y_train,
    epochs=10,  # ✅ Reduced to 10 epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save("emotion_recognition_model.keras")
print("Model training complete! Model saved as 'emotion_recognition_model.keras'.")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
