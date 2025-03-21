# Import required libraries
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define actions and output folder
ACTIONS = ["kick", "punch", "idle"]
OUTPUT_FOLDER = "action_data"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize counters for each action
action_counters = {action: 1 for action in ACTIONS}
recording_duration = 5  # Time in seconds for each recording
collecting_data = False
start_time = None
current_action = "idle"

# Define function to save landmarks
def save_landmarks(landmarks, action, file_number):
    file_path = os.path.join(OUTPUT_FOLDER, f"{action}_{file_number}.csv")
    with open(file_path, "w") as f:
        # Write header with landmark coordinates
        f.write("action," + ",".join([f"x{i},y{i},z{i}" for i in range(33)]) + "\n")
        
        # Write landmark data
        if landmarks:
            row = [action]
            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            f.write(",".join(map(str, row)) + "\n")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Main loop to capture video and record actions
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to get pose landmarks
    result = pose.process(rgb_frame)

    # Draw landmarks if detected
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display current action and recording status
    cv2.putText(frame, f"Action: {current_action}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if collecting_data:
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording... {int(recording_duration - elapsed_time)}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Stop recording after set duration
        if elapsed_time >= recording_duration:
            save_landmarks(result.pose_landmarks, current_action, action_counters[current_action])
            print(f"✅ {current_action}_{action_counters[current_action]}.csv saved!")
            action_counters[current_action] += 1  # Increment file counter
            collecting_data = False
            current_action = "idle"

    # Check key press to start recording for an action
    key = cv2.waitKey(1) & 0xFF
    if key == ord("k"):  # Press 'k' to record "kick"
        current_action = "kick"
        collecting_data = True
        start_time = time.time()
    elif key == ord("p"):  # Press 'p' to record "punch"
        current_action = "punch"
        collecting_data = True
        start_time = time.time()
    elif key == ord("i"):  # Press 'i' for "idle" (no recording)
        current_action = "idle"
        collecting_data = False
    elif key == ord("q"):  # Press 'q' to quit
        break

    # Show the video feed
    cv2.imshow("Action Recorder", frame)

# Release and close
cap.release()
cv2.destroyAllWindows()
