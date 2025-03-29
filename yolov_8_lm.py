import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass
class SimplifiedHandLandmarks:
    """Simplified hand landmarks with 7 key points"""
    wrist: Tuple[float, float]  # Wrist position
    thumb_tip: Tuple[float, float]  # Thumb tip
    index_base: Tuple[float, float]  # Base of index finger
    index_tip: Tuple[float, float]  # Index fingertip
    middle_tip: Tuple[float, float]  # Middle fingertip
    ring_tip: Tuple[float, float]  # Ring fingertip
    pinky_tip: Tuple[float, float]  # Pinky fingertip

@dataclass
class PersonData:
    """Data structure to store detection and landmark information for a person"""
    box: List[int]  # [x1, y1, x2, y2]
    pose_landmarks: Optional[List[Tuple[float, float]]] = None
    left_hand: Optional[SimplifiedHandLandmarks] = None
    right_hand: Optional[SimplifiedHandLandmarks] = None
    left_gesture: str = "None"
    right_gesture: str = "None"

class PersonDetectorWithSimplifiedLandmarks:
    def __init__(self, yolo_model_size="n", min_detection_confidence=0.5):
        """
        Initialize detector for person detection with simplified landmarks
        
        Args:
            yolo_model_size (str): Size of YOLOv8 model (n, s, m, l, x)
            min_detection_confidence (float): Confidence threshold for detection
        """
        self.conf_threshold = min_detection_confidence
        
        # Initialize YOLOv8 for person detection
        self.yolo_model = YOLO(f"yolov8{yolo_model_size}.pt")
        
        # Initialize MediaPipe Holistic for body and hand landmarks
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        
        # Important landmark indices
        self.hand_keypoints = {
            "wrist": 0,
            "thumb_tip": 4,
            "index_base": 5,
            "index_tip": 8,
            "middle_tip": 12,
            "ring_tip": 16,
            "pinky_tip": 20
        }
        
        # Selected body keypoints (subset of 33)
        self.selected_pose_keypoints = [
            0,   # nose
            11,  # left shoulder
            12,  # right shoulder
            13,  # left elbow
            14,  # right elbow
            15,  # left wrist
            16,  # right wrist
            23,  # left hip
            24,  # right hip
            25,  # left knee
            26,  # right knee
            27,  # left ankle
            28   # right ankle
        ]
    
    def _extract_simplified_hand_landmarks(self, hand_landmarks):
        """Extract only the specified key points from hand landmarks"""
        if not hand_landmarks:
            return None
            
        points = {}
        for name, idx in self.hand_keypoints.items():
            landmark = hand_landmarks.landmark[idx]
            points[name] = (landmark.x, landmark.y)
            
        return SimplifiedHandLandmarks(
            wrist=points["wrist"],
            thumb_tip=points["thumb_tip"],
            index_base=points["index_base"],
            index_tip=points["index_tip"],
            middle_tip=points["middle_tip"],
            ring_tip=points["ring_tip"],
            pinky_tip=points["pinky_tip"]
        )
    
    def _detect_open_palm(self, hand: SimplifiedHandLandmarks) -> bool:
        """Check if hand gesture is an open palm using simplified landmarks"""
        if not hand:
            return False
        
        # Check if fingertips are higher than base (for a vertical hand)
        fingers_extended = (
            hand.index_tip[1] < hand.index_base[1] and
            hand.middle_tip[1] < hand.index_base[1] and
            hand.ring_tip[1] < hand.index_base[1] and
            hand.pinky_tip[1] < hand.index_base[1]
        )
        
        # Check thumb position (simplified)
        thumb_extended = (
            abs(hand.thumb_tip[0] - hand.wrist[0]) > 
            abs(hand.index_base[0] - hand.wrist[0])
        )
        
        return fingers_extended and thumb_extended
    
    def _detect_closed_fist(self, hand: SimplifiedHandLandmarks) -> bool:
        """Check if hand gesture is a closed fist using simplified landmarks"""
        if not hand:
            return False
        
        # Check if fingertips are lower than base (fingers curled)
        fingers_bent = (
            hand.index_tip[1] > hand.index_base[1] and
            hand.middle_tip[1] > hand.index_base[1] and
            hand.ring_tip[1] > hand.index_base[1] and
            hand.pinky_tip[1] > hand.index_base[1]
        )
        
        # Check thumb position (simplified)
        thumb_bent = (
            abs(hand.thumb_tip[0] - hand.wrist[0]) < 
            abs(hand.index_base[0] - hand.wrist[0])
        )
        
        return fingers_bent and thumb_bent
    
    def _detect_pointing(self, hand: SimplifiedHandLandmarks) -> bool:
        """Check if hand is pointing (index extended, others closed)"""
        if not hand:
            return False
        
        # Index finger extended
        index_extended = hand.index_tip[1] < hand.index_base[1]
        
        # Other fingers bent
        other_fingers_bent = (
            hand.middle_tip[1] > hand.index_base[1] and
            hand.ring_tip[1] > hand.index_base[1] and
            hand.pinky_tip[1] > hand.index_base[1]
        )
        
        return index_extended and other_fingers_bent
    
    def detect_gesture(self, hand: SimplifiedHandLandmarks) -> str:
        """Identify hand gesture from simplified landmarks"""
        if not hand:
            return "None"
        
        if self._detect_open_palm(hand):
            return "Open Palm"
        elif self._detect_closed_fist(hand):
            return "Closed Fist"
        elif self._detect_pointing(hand):
            return "Pointing"
        
        return "Unknown"
    
    def process_frame(self, frame):
        """Process frame to detect persons with landmarks for each person"""
        # Detect persons using YOLOv8
        yolo_results = self.yolo_model(frame, conf=self.conf_threshold, classes=[0])  # 0 is person class
        
        # Get person bounding boxes
        person_boxes = []
        for result in yolo_results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                person_boxes.append([x1, y1, x2, y2])
        
        # If no persons detected, return original frame
        if not person_boxes:
            return frame, []
        
        # Make a copy for drawing
        annotated_frame = frame.copy()
        
        # List to store data for each detected person
        detected_persons = []
        
        # Process each detected person
        for box in person_boxes:
            x1, y1, x2, y2 = box
            
            # Ensure box is within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Skip if box is too small
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
                
            # Create person data object
            person = PersonData(box=box)
            
            # Extract person ROI
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue
                
            # Convert ROI to RGB for MediaPipe
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_roi)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Scale factors for mapping back to original frame
            scale_x = (x2 - x1) / person_roi.shape[1]
            scale_y = (y2 - y1) / person_roi.shape[0]
            
            # Process pose landmarks
            if results.pose_landmarks:
                pose_points = []
                
                # Draw selected pose landmarks
                for idx in self.selected_pose_keypoints:
                    landmark = results.pose_landmarks.landmark[idx]
                    # Map coordinates back to original frame
                    px = int(landmark.x * person_roi.shape[1] * scale_x) + x1
                    py = int(landmark.y * person_roi.shape[0] * scale_y) + y1
                    pose_points.append((px, py))
                    cv2.circle(annotated_frame, (px, py), 5, (0, 255, 255), -1)
                
                # Connect landmarks with lines
                connections = [
                    (0, 1), (0, 2),  # Nose to shoulders
                    (1, 3), (3, 5),  # Left arm
                    (2, 4), (4, 6),  # Right arm
                    (1, 7), (2, 8),  # Shoulders to hips
                    (7, 9), (9, 11),  # Left leg
                    (8, 10), (10, 12)  # Right leg
                ]
                
                for connection in connections:
                    if connection[0] < len(pose_points) and connection[1] < len(pose_points):
                        cv2.line(annotated_frame, 
                                pose_points[connection[0]], 
                                pose_points[connection[1]], 
                                (0, 255, 0), 2)
                
                person.pose_landmarks = pose_points
            
            # Process left hand landmarks
            if results.left_hand_landmarks:
                left_hand = self._extract_simplified_hand_landmarks(results.left_hand_landmarks)
                
                # Map simplified landmarks to original frame
                if left_hand:
                    points = []
                    for point_name in ["wrist", "thumb_tip", "index_base", "index_tip", 
                                     "middle_tip", "ring_tip", "pinky_tip"]:
                        point = getattr(left_hand, point_name)
                        px = int(point[0] * person_roi.shape[1] * scale_x) + x1
                        py = int(point[1] * person_roi.shape[0] * scale_y) + y1
                        points.append((px, py))
                    
                    # Draw simplified hand landmarks
                    for point in points:
                        cv2.circle(annotated_frame, point, 5, (255, 0, 0), -1)
                    
                    # Connect points with lines for better visualization
                    connections = [(0, 1), (0, 2), (2, 3), (0, 4), (0, 5), (0, 6)]
                    for connection in connections:
                        cv2.line(annotated_frame, points[connection[0]], 
                               points[connection[1]], (255, 0, 0), 2)
                    
                    # Detect gesture and display
                    person.left_gesture = self.detect_gesture(left_hand)
                    if person.left_gesture != "None" and person.left_gesture != "Unknown":
                        cv2.putText(annotated_frame, f"L: {person.left_gesture}", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 0, 0), 2)
                    
                    # Store mapped hand landmarks
                    remapped_hand = SimplifiedHandLandmarks(
                        wrist=points[0],
                        thumb_tip=points[1],
                        index_base=points[2],
                        index_tip=points[3],
                        middle_tip=points[4],
                        ring_tip=points[5],
                        pinky_tip=points[6]
                    )
                    person.left_hand = remapped_hand
            
            # Process right hand landmarks
            if results.right_hand_landmarks:
                right_hand = self._extract_simplified_hand_landmarks(results.right_hand_landmarks)
                
                # Map simplified landmarks to original frame
                if right_hand:
                    points = []
                    for point_name in ["wrist", "thumb_tip", "index_base", "index_tip", 
                                     "middle_tip", "ring_tip", "pinky_tip"]:
                        point = getattr(right_hand, point_name)
                        px = int(point[0] * person_roi.shape[1] * scale_x) + x1
                        py = int(point[1] * person_roi.shape[0] * scale_y) + y1
                        points.append((px, py))
                    
                    # Draw simplified hand landmarks
                    for point in points:
                        cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)
                    
                    # Connect points with lines for better visualization
                    connections = [(0, 1), (0, 2), (2, 3), (0, 4), (0, 5), (0, 6)]
                    for connection in connections:
                        cv2.line(annotated_frame, points[connection[0]], 
                               points[connection[1]], (0, 0, 255), 2)
                    
                    # Detect gesture and display
                    person.right_gesture = self.detect_gesture(right_hand)
                    if person.right_gesture != "None" and person.right_gesture != "Unknown":
                        text_x = min(x1 + 100, frame.shape[1] - 100)
                        cv2.putText(annotated_frame, f"R: {person.right_gesture}", 
                                  (text_x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 0, 255), 2)
                    
                    # Store mapped hand landmarks
                    remapped_hand = SimplifiedHandLandmarks(
                        wrist=points[0],
                        thumb_tip=points[1],
                        index_base=points[2],
                        index_tip=points[3],
                        middle_tip=points[4],
                        ring_tip=points[5],
                        pinky_tip=points[6]
                    )
                    person.right_hand = remapped_hand
            
            # Add person data to list
            detected_persons.append(person)
        
        return annotated_frame, detected_persons
    
    def process_image(self, image_path):
        """Process a single image file"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        annotated_frame, detected_persons = self.process_frame(frame)
        
        cv2.imshow("Person Detection with Simplified Landmarks", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detected_persons
    
    def process_video(self, video_path=None, camera_index=0):
        """Process video file or webcam feed"""
        cap = cv2.VideoCapture(video_path if video_path else camera_index)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            annotated_frame, detected_persons = self.process_frame(frame)
            
            # Display person count
            cv2.putText(
                annotated_frame,
                f"Persons: {len(detected_persons)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Person Detection with Simplified Landmarks", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Create detector
    detector = PersonDetectorWithSimplifiedLandmarks(yolo_model_size="n")
    
    # Process webcam feed
    # detector.process_image("path/to/your/image.jpg")
    detector.process_video()  # Uses webcam by default
    # detector.process_video("path/to/your/video.mp4")  # Or process a video file