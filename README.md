# 🎓 AI-Powered Coding Platform

An intelligent web application that monitors user presence, detects mobile phone usage, and analyzes emotions in real-time using advanced AI models. Perfect for proctoring, focus monitoring, and engagement tracking during coding sessions or online learning.

## ✨ Features

- **🔍 Presence Detection**: Real-time person detection using YOLOv8
- **📱 Phone Usage Detection**: Identifies when users are distracted by mobile phones
- **😊 Emotion Analysis**: Advanced emotion recognition using HSEmotion model
- **📊 Engagement Metrics**: Calculates valence and engagement scores
- **🎯 Multi-Person Support**: Handles multiple people in the frame
- **⚡ Real-Time Processing**: Fast inference with optimized model pipelines

## 🏗️ Project Structure

```
.
├── backend/                    # FastAPI Backend Server
│   ├── api.py                 # Main API endpoints
│   ├── main.py                # Standalone testing driver
│   ├── detector.py            # Person detection module
│   ├── emotion_detector.py    # Emotion detection module
│   ├── color_utils.py         # Color analysis utilities
│   ├── requirements.txt       # Python dependencies
│   ├── yolov8n.pt            # YOLOv8 model weights
│   └── services/              # Service layer
│       ├── emotion.py         # Emotion recognition service
│       └── presence.py        # Presence detection service
└── frontend/                   # React Frontend Application
```

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for Python
- **YOLOv8**: State-of-the-art object detection (Ultralytics)
- **HSEmotion**: High-performance emotion recognition
- **OpenCV**: Computer vision processing
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations

### Frontend
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Webcam (for real-time detection)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

5. API will be available at `http://localhost:8000`
   - Interactive API docs: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser at `http://localhost:5173`

## 📡 API Endpoints

### POST `/predict`
Analyzes an uploaded image for presence, phone usage, and emotions.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "status": "ok",
  "person_count": 1,
  "phone_detected": false,
  "emotions": {
    "anger": 0.01,
    "contempt": 0.02,
    "disgust": 0.01,
    "fear": 0.03,
    "happiness": 0.85,
    "neutral": 0.05,
    "sadness": 0.02,
    "surprise": 0.01
  },
  "engagement": 0.75,
  "valence": 0.82,
  "debug_image": "base64_encoded_image"
}
```

## 💡 Usage

### Web Application
1. Open the frontend application in your browser
2. Upload an image or use your webcam
3. The system will automatically:
   - Detect if a person is present
   - Check for mobile phone usage
   - Analyze facial emotions
   - Provide engagement and valence scores

### Standalone Testing
Run the backend driver code for local testing:
```bash
cd backend
python main.py
```
Press 'q' to quit the camera feed.

## 🎯 Use Cases

- **Online Proctoring**: Monitor student attention during exams
- **Focus Tracking**: Measure concentration during work sessions
- **Engagement Analytics**: Analyze user engagement in online learning
- **Distraction Detection**: Identify mobile phone usage patterns
- **Emotion Research**: Study emotional responses during tasks

## ⚙️ Configuration

### Presence Detection
- Person confidence threshold: `0.6` (adjustable in `presence.py`)
- Phone confidence threshold: `0.4` (adjustable in `presence.py`)

### Emotion Detection
- Model: `enet_b0_8_best_vgaf` (HSEmotion)
- Device: CPU (can be changed to GPU)

## 🐛 Troubleshooting

### Model Loading Issues
If you encounter issues loading the HSEmotion model, clear the torch cache:
```bash
rm -rf ~/.cache/torch/hub/checkpoints/
```

### Camera Access
Ensure your browser has camera permissions enabled for localhost.

### CORS Issues
If you experience CORS errors, verify that the backend CORS middleware is properly configured in `api.py`.

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

Built with ❤️ using AI and Computer Vision
