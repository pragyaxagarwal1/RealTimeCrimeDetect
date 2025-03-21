import cv2
import os
import numpy as np
import mediapipe as mp
import csv
import time

# Define actions and save directory
actions = {"k": "kick", "p": "punch", "i": "idle"}
save_dir = "action_data"

# Create directory if it doesn't exist    os.makedirs(save_dir)

# Ask user to choose an action
print("Press 'k' for Kick, 'p' for Punch, 'i' for Idle")
key = input("Select an action (k/p/i): ").strip().lower()

if key not in actions:
    print("❌ Invalid selection. Press 'k', 'p', or 'i'.")
    exit()

action = actions[key]
print(f"✅ Action selected: {action}")

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

# Get the sample number
sample_count = len([f for f in os.listdir(save_dir) if f.startswith(action)]) + 1
file_name = f"{action}_{sample_count}.csv"
file_path = os.path.join(save_dir, file_name)

# Add 2-second delay before recording
print("⏳ Get ready... Starting in 2 seconds!")
time.sleep(2)
print("🎥 Recording started...")

# Open CSV file for writing
with open(file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for natural view
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Check if landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Flatten landmark data (33 landmarks * 3 values)
            landmark_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

            # Save landmarks if valid
            if landmark_data.shape[0] == 99:
                writer.writerow(landmark_data)

        # Show live feed with status and landmarks
        cv2.putText(frame, f"Recording {action}_{sample_count}. Press 'q' to stop.", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Recording {action}", frame)

        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"✅ Recording saved as {file_name}")
            break

# Release webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()

print(f"🎉 Successfully recorded one sample for action '{action}' and saved as CSV in '{save_dir}'")
