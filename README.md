# Skin Tone Analyzer

A premium web application to predict skin tone (Black/White) and suggest skincare remedies using AI.

## Project Structure
- `backend/`: FastAPI server with YOLOv8 and Color Analysis.
- `frontend/`: React + TypeScript + Tailwind CSS application.

## Setup Instructions

### 1. Backend Setup
1. Open a terminal.
2. Navigate to the backend directory:
   ```bash
   cd backend
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

### 2. Frontend Setup
1. Open a new terminal.
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```

## Usage
- Open the provided Local URL (usually `http://localhost:5173`).
- Upload a photo of a person.
- The AI will detect the person, analyze the skin tone, and provide tailored remedies.
