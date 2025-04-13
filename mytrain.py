import cv2
import os

# Define paths for each emotion category
emotions = ['angry', 'happy', 'sad', 'surprised', 'neutral']
data_dir = "emotion_data"  # Path where images will be saved

# Create directories for each emotion if they don't exist
for emotion in emotions:
    os.makedirs(os.path.join(data_dir, emotion), exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Counter for number of images
img_count = 0
current_emotion = 'neutral'  # Default emotion (start with neutral)

print("Press 'a' for angry, 'h' for happy, 's' for sad, 'p' for surprised, 'n' for neutral.")
print("Press 'q' to quit data collection.")
print("Press 'c' to capture an image.")

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()

    # Check if frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Capture', frame)

    # Capture key press to change emotion label
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):  # Angry
        current_emotion = 'angry'
        print("Emotion: Angry")
    elif key == ord('h'):  # Happy
        current_emotion = 'happy'
        print("Emotion: Happy")
    elif key == ord('s'):  # Sad
        current_emotion = 'sad'
        print("Emotion: Sad")
    elif key == ord('p'):  # Surprised
        current_emotion = 'surprised'
        print("Emotion: Surprised")
    elif key == ord('n'):  # Neutral
        current_emotion = 'neutral'
        print("Emotion: Neutral")
    elif key == ord('c'):  # Capture Image
        if len(faces) > 0:  # Only capture if a face is detected
            img_name = os.path.join(data_dir, current_emotion, f"image_{img_count}.jpg")
            cv2.imwrite(img_name, frame)
            img_count += 1
            print(f"Image saved as {img_name}")
        else:
            print("No face detected! Please ensure your face is visible.")
    elif key == ord('q'):  # Quit
        print("Exiting data collection...")
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
