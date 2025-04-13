import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# Define the emotion categories (make sure these match the number of output classes of the model)
categories = ['angry', 'happy', 'sad', 'surprised', 'neutral']  # Update this if necessary

# Load the trained model
model = load_model("emotion_recognition_model.keras")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBP feature extractor
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = grayscale[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        features = extract_lbp_features(face)
        features = features.reshape(1, -1)  # Reshape for model input

        # Predict the emotion
        prediction = model.predict(features)

        # Debug: Print the prediction output
        print(f"Prediction Output: {prediction}")

        # Check the shape of the prediction and use np.argmax to find the correct emotion
        if prediction.shape[1] == len(categories):
            emotion = categories[np.argmax(prediction)]
            print(f"Predicted Emotion: {emotion}")
        else:
            print("Prediction shape does not match the number of categories!")
            continue

        # Display result on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
