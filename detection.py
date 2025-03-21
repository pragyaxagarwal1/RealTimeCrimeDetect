import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = tf.keras.models.load_model("action_model.h5")

# Define actions
actions = ["kick", "punch", "idle"]
label_encoder = LabelEncoder()
label_encoder.fit(actions)

# Initialize Mediapipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define drawing specifications for landmarks and connections
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)  # Stick figure style
connection_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# 💡 Set higher resolution (e.g., 1280x720 or 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural view
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Check if any landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Flatten and normalize landmark data
        landmark_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

        # Ensure correct shape before prediction
        if landmark_data.shape[0] == 99:  # 33 landmarks * 3 (x, y, z)
            landmark_data = landmark_data.reshape(33, 3)

            # Add batch dimension and predict
            prediction = model.predict(np.expand_dims(landmark_data, axis=0))
            action_index = np.argmax(prediction)
            predicted_action = label_encoder.inverse_transform([action_index])[0]

            # Display predicted action on the frame
            cv2.putText(frame, f"Action: {predicted_action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks and connections with bigger sizes
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_style, connection_style)

    # Display frame with action prediction and larger resolution
    cv2.imshow("Action Detection (High Resolution)", frame)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
