# Import required libraries
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "action_model.h5"
model = load_model(MODEL_PATH)

# Define actions and corresponding labels
ACTIONS = ["kick", "punch", "idle"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define function to extract landmarks from the detected pose
def extract_landmarks(landmarks):
    data = []
    for lm in landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])  # Add x, y, z coordinates
    return np.array(data).reshape(1, 33, 3)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Main loop for real-time action detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Unable to access webcam. Exiting...")
        break

    # Flip frame horizontally and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose landmarks
    result = pose.process(rgb_frame)

    # Check if landmarks are detected
    if result.pose_landmarks:
        # Draw landmarks and connections on the frame
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks and normalize
        landmarks = extract_landmarks(result.pose_landmarks)
        landmarks = landmarks / np.max(landmarks)  # Normalize before prediction

        # Make prediction using the trained model
        prediction = model.predict(landmarks)

        # Get the action with highest confidence
        action_index = np.argmax(prediction)
        action_name = ACTIONS[action_index]
        confidence = prediction[0][action_index]

        # Display the action and confidence on the frame
        cv2.putText(frame, f"Action: {action_name} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Show message if no pose is detected
        cv2.putText(frame, "No action detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Action Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
