from ultralytics import YOLO
import numpy as np

class PresenceService:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        # Class IDs
        self.CLASS_PERSON = 0
        self.CLASS_PHONE = 67

    def detect_presence(self, img, conf_person=0.6, conf_phone=0.4):
        """
        Detects person and mobile phone.
        Returns status: 'ok', 'no_user', 'mobile_detected'
        """
        if img is None:
            return "no_user"

        # Run inference
        # We use a lower base confidence to catch objects, then filter.
        results = self.model.predict(img, conf=0.3, classes=[self.CLASS_PERSON, self.CLASS_PHONE], verbose=False)
        
        persons = []
        phones = []

        for result in results:
            for box in result.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == self.CLASS_PERSON and conf >= conf_person:
                    persons.append(coords)
                elif cls == self.CLASS_PHONE and conf >= conf_phone:
                    phones.append(coords)

        # 1. No Person Detected
        if len(persons) == 0:
            return "no_user", [], []

        # 2. Mobile Phone Usage Detected
        if len(phones) > 0:
            # Check interaction
            # If any phone intersects with any person, trigger.
            # Loose intersection: strict intersection might miss holding phone near chest.
            # We'll use a lenient overlap check.
            for p_box in persons:
                for ph_box in phones:
                    if self._check_intersection(p_box, ph_box):
                        return "mobile_detected", persons, phones
        
        return "ok", persons, phones

    def _check_intersection(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight
        
        if interArea > 0:
            return True
            
        # Optional: Check proximity if no direct intersection?
        # For now, strict intersection of bounding boxes is usually enough if phone is "in front" of person.
        return False
