import cv2
import sys
import numpy as np
from detector import PersonDetector
from emotion_detector import EmotionDetector
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Driver code to test Emotion Detection logic locally without the web backend.
    """
    print("--------------------------------------------------")
    print("Emotion Detection Driver Code")
    print("Press 'q' to quit.")
    print("--------------------------------------------------")

    # 1. Initialize Detectors
    try:
        # We don't strictly one for emotion, but good for fallback body detection
        detector = PersonDetector("yolov8n.pt", conf=0.5)
        emotion_model = EmotionDetector() 
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return

    # 2. Open Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    # Haar for face
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # A. Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_crop = None
        face_coords = None

        if len(faces) > 0:
            # Largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Margin
            mx, my = int(w*0.2), int(h*0.2)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(frame.shape[1], x + w + mx)
            y2 = min(frame.shape[0], y + h + my)
            
            face_crop = frame[y1:y2, x1:x2]
            face_coords = (x, y, w, h)
        
        # B. Fallback to Body if no face
        else:
            detections = detector.detect(frame, classes=[0])
            # filter just in case
            persons = [d[0] for d in detections if d[1] == 0]
            
            if persons:
                 # Largest person
                largest_person = max(persons, key=lambda p: (p[2]-p[0]) * (p[3]-p[1]))
                px1, py1, px2, py2 = largest_person
                phead_h = int((py2 - py1) * 0.25) # Guess head is top 25%
                face_crop = frame[py1:py1+phead_h, px1:px2]
                face_coords = (px1, py1, px2-px1, phead_h)

        # C. Predict Emotion
        if face_crop is not None and face_crop.size > 0:
            try:
                results = emotion_model.detect_emotion(face_crop)
                
                # D. Viz
                fx, fy, fw, fh = face_coords
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                
                # Valence Label
                cv2.putText(frame, f"{results['valence']}", (fx, max(fy-10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Top Emotions
                y_off = fy + fh + 20
                for en, es in sorted(results['emotions'].items(), key=lambda x:x[1], reverse=True)[:2]:
                     cv2.putText(frame, f"{en}: {es:.2f}", (fx, y_off), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                     y_off += 25
                
                # Engagement
                cv2.putText(frame, f"Engagement: {results['engagement_score']}%", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            except Exception as e:
                # logging.error(f"Prediction Error: {e}") 
                pass

        cv2.imshow("Emotion Driver Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
