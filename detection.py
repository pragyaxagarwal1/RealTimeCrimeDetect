import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_recognition_model.keras")

# Emotion categories (should be same order as your training folder names)
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Change if needed

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
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face = grayscale[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # Normalize the image (scaling pixel values between 0 and 1)
        face = face.astype('float32') / 255.0
        features = extract_lbp_features(face)
        features = features.reshape(1, -1)  # Reshape for model

        prediction = model.predict(features)
        emotion = categories[np.argmax(prediction)]

        # Display result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
