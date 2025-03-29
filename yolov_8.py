import cv2
import torch
from ultralytics import YOLO

def detect_persons(image_path=None, video_path=None, camera_index=0, conf_threshold=0.5):
    """
    Detects persons in images, videos, or camera feed using YOLOv8
    
    Args:
        image_path (str, optional): Path to image file
        video_path (str, optional): Path to video file
        camera_index (int, optional): Camera index for webcam detection
        conf_threshold (float, optional): Confidence threshold for detections
        
    Returns:
        None: Displays output with detections
    """
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # you can change to other sizes (s, m, l, x) for better performance
    
    # Set class to filter (0 is person in COCO dataset)
    class_ids = [0]  # 0 represents 'person' class
    
    if image_path:
        # Process image
        img = cv2.imread(image_path)
        results = model(img, conf=conf_threshold, classes=class_ids)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Display output
        cv2.imshow("YOLOv8 Person Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif video_path:
        # Process video
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = model(frame, conf=conf_threshold, classes=class_ids)
            annotated_frame = results[0].plot()
            
            cv2.imshow("YOLOv8 Person Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # Use webcam
        cap = cv2.VideoCapture(camera_index)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = model(frame, conf=conf_threshold, classes=class_ids)
            annotated_frame = results[0].plot()
            
            cv2.imshow("YOLOv8 Person Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    # Detect persons in an image
    # detect_persons(image_path="path/to/your/image.jpg")
    
    # Detect persons in a video
    # detect_persons(video_path="path/to/your/video.mp4")
    
    # Detect persons using webcam
    detect_persons()