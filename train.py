import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Path to the extracted dataset
data_dir = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\train"
categories = os.listdir(data_dir)
img_size = 48  # Target image size

X = []
y = []

# Load and preprocess images
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

y = to_categorical(y, num_classes=len(categories))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture with CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_emotion_model_cnn.h5", monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Training
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,  # Set to 10 epochs
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save("emotion_recognition_model_cnn.keras")
print("Model training complete! Model saved as 'emotion_recognition_model_cnn.keras'.")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
