# Import required libraries
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8_model_path.pt')  # Update with the correct model path

# Define a dictionary to store flagged people and their timestamps
flagged_people = {}

# Define constants
VICTIM_COLOR = (0, 255, 0)  # Green
ASSAULTER_COLOR = (0, 0, 255)  # Red
NORMAL_COLOR = (255, 255, 255)  # White
HOLD_TIME = 10  # Hold time for 10 seconds

# Function to draw boxes with color persistence
def draw_box_with_persistence(frame, bbox, person_id, label, color):
    if person_id in flagged_people:
        elapsed_time = time.time() - flagged_people[person_id]['timestamp']
        # Check if 10 seconds have passed
        if elapsed_time <= HOLD_TIME:
            color = flagged_people[person_id]['color']
        else:
            del flagged_people[person_id]  # Remove the person after 10 seconds

    # Draw the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to update flagged person
def flag_person(person_id, label):
    if label == "victim":
        color = VICTIM_COLOR
    elif label == "assaulter":
        color = ASSAULTER_COLOR
    else:
        color = NORMAL_COLOR
    
    # Update the flag status
    flagged_people[person_id] = {
        'timestamp': time.time(),
        'color': color
    }

# Open the webcam feed
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if necessary

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    for result in results.pred[0]:
        x1, y1, x2, y2, conf, class_id = result
        label = model.names[int(class_id)]
        person_id = int(result[-1])  # Assuming person_id is available

        # Flag as victim or assaulter based on conditions (replace with your logic)
        if conf > 0.8 and label == 'person':
            if some_condition_for_victim:  # Define condition for victim
                flag_person(person_id, 'victim')
            elif some_condition_for_assaulter:  # Define condition for assaulter
                flag_person(person_id, 'assaulter')
            else:
                flag_person(person_id, 'normal')
        
        # Draw boxes
        color = NORMAL_COLOR
        if person_id in flagged_people:
            color = flagged_people[person_id]['color']
        
        draw_box_with_persistence(frame, (x1, y1, x2, y2), person_id, label, color)

    # Display results
    cv2.imshow('YOLOv8 Live Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
