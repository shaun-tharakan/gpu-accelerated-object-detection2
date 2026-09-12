import cv2
import time
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLO model. 
        Loads the model only once to avoid overhead during frame processing.
        """
        self.model = YOLO(model_name)

    def detect_frame(self, frame, device="cpu"):
        """
        Runs object detection on a single frame.
        
        Args:
            frame: OpenCV/numpy image frame in BGR format
            device: "cpu" or "cuda"
            
        Returns:
            A list of dictionaries containing structured detection results.
        """
        # Run the model on the frame using the specified device
        results = self.model(frame, device=device, verbose=False)
        
        structured_results = []
        
        # Parse the Ultralytics results into the agreed-upon format
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates, confidence, and class info
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Append to our structured output format
                structured_results.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })
                
        return structured_results

    def annotate_frame(self, frame, detections):
        """
        Draws bounding boxes and labels on the frame based on detections.
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            
            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw a background rectangle for the text
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Put the class name and confidence text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return annotated_frame

    def process_video(self, video_path, device="cpu"):
        """
        Video pipeline: reads a video, detects objects, annotates, and yields metrics.
        Exposes FPS/latency information without UI dependencies.
        
        Yields:
            annotated_frame, latency_ms, current_fps
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Measure latency for a single frame
            start_time = time.time()
            detections = self.detect_frame(frame, device=device)
            latency_ms = (time.time() - start_time) * 1000
            
            # Annotate the frame
            annotated_frame = self.annotate_frame(frame, detections)
            
            # Calculate overall FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            yield annotated_frame, latency_ms, fps
            
        cap.release()