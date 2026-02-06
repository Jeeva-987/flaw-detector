from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import os

from services.presence import PresenceService
from services.emotion import EmotionService

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Services...")
presence_service = PresenceService()
emotion_service = EmotionService()

# Haar Cascade for face detection (Fast fallback)
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
except:
    face_cascade = None

print("Services Ready.")

@app.post("/predict")
async def analyze_emotion(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # 1. Check Presence
    status, persons, phones = presence_service.detect_presence(img)

    if status != "ok":
        # Create debug image even if failed status
        debug_vis = img.copy()
        # Draw persons
        for p in persons:
            cv2.rectangle(debug_vis, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
        # Draw phones
        for ph in phones:
            cv2.rectangle(debug_vis, (ph[0], ph[1]), (ph[2], ph[3]), (0, 0, 255), 2)
            cv2.putText(debug_vis, "PHONE", (ph[0], ph[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return _response(status, None, debug_vis)

    # 2. Extract Face (Haar or Person Crop)
    face_crop, face_coords = _extract_face(img, persons, face_cascade)
    
    if face_crop is None or face_crop.size == 0:
        return _response("ok", None, img)

    # 3. Predict Emotion
    scores = emotion_service.predict(face_crop)
    if not scores:
         # Failed to predict
         return _response("ok", None, img)
    
    analysis = emotion_service.analyze(scores)

    # 4. Debug Viz
    debug_vis = img.copy()
    if face_coords:
        fx, fy, fw, fh = face_coords
        cv2.rectangle(debug_vis, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
        if analysis:
            cv2.putText(debug_vis, analysis['valence'], (fx, max(fy-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return _response("ok", analysis, debug_vis)

def _extract_face(img, persons, cascade):
    # Try Haar first
    if cascade:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest
            # Margin
            mx, my = int(w*0.2), int(h*0.2)
            x1, y1 = max(0, x-mx), max(0, y-my)
            x2, y2 = min(img.shape[1], x+w+mx), min(img.shape[0], y+h+my)
            return img[y1:y2, x1:x2], (x, y, w, h)
            
    # Fallback to Person Crop Head Estimate
    if persons:
        # Largest person
        p = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        px1, py1, px2, py2 = p
        pw, ph = px2-px1, py2-py1
        # Head estimate top 25%
        head_h = int(ph * 0.25)
        return img[py1:py1+head_h, px1:px2], (px1, py1, pw, head_h)

    return None, None

def _response(status, analysis, debug_img):
    jpg_as_text = ""
    if debug_img is not None:
        _, buffer = cv2.imencode('.jpg', debug_img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": status,
        "analysis": analysis,
        "debug_image": f"data:image/jpeg;base64,{jpg_as_text}" if jpg_as_text else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

