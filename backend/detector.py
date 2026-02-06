from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, img, classes=[0], conf=None):
        """
        Detects objects in the image.
        Returns a list of tuples: ([x1, y1, x2, y2], class_id)
        """
        if img is None:
            return []
            
        confidence = conf if conf is not None else self.conf
        results = self.model.predict(img, conf=confidence, classes=classes, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                detections.append(((x1, y1, x2, y2), cls))
        return detections
