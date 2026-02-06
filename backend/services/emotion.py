import numpy as np
from hsemotion.facial_emotions import HSEmotionRecognizer
import os
import torch

class EmotionService:
    def __init__(self, model_name='enet_b0_8_best_vgaf'):
        self.model = None
        self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Loading HSEmotion model: {model_name}...")
        try:
            # Check if we can force download or handle cache issues?
            # HSEmotion uses torch.hub.load logic internally or similar.
            self.model = HSEmotionRecognizer(model_name=model_name, device='cpu')
            print("HSEmotion model loaded successfully.")
        except Exception as e:
            print(f"Error loading HSEmotion: {e}")
            # Fallback?
            # If "WeightsUnpickler error", maybe delete cache?
            # We can't easily fix it here without file access.
            self.model = None

    def predict(self, face_img):
        if self.model is None:
            return None

        try:
            if isinstance(face_img, np.ndarray):
                # Ensure RGB
                if face_img.shape[-1] == 3:
                     # OpenCV is BGR, Model likely wants RGB
                     face_img = face_img[..., ::-1]
            
            emotion, scores = self.model.predict_emotions(face_img, logits=False)
            
            # Normalize scores
            scores_norm = {k.lower(): v for k, v in scores.items()}
            
            return scores_norm
        except Exception as e:
            print(f"Emotion Prediction Error: {e}")
            return None

    def analyze(self, scores):
        """
        Derive valence and engagement from raw scores.
        """
        if not scores:
            return None

        happiness = scores.get('happiness', 0.0)
        neutral = scores.get('neutral', 0.0)
        surprise = scores.get('surprise', 0.0)
        anger = scores.get('anger', 0.0)
        sadness = scores.get('sadness', 0.0)
        disgust = scores.get('disgust', 0.0)
        fear = scores.get('fear', 0.0)
        
        # Valence
        val_pos = happiness
        val_neg = anger + sadness + disgust + fear
        val_neu = neutral + surprise
        
        if val_pos > val_neg and val_pos > val_neu:
            valence = "Positive"
        elif val_neg > val_pos:
            valence = "Negative"
        else:
            # If neutral dominant but secondary emotion is significant (>0.2)
            # detect it as "Neutral+"
            max_non_neutral = max(happiness, anger, sadness, disgust, fear, surprise)
            if max_non_neutral > 0.2:
                 # Check which one
                 if happiness == max_non_neutral: valence = "Neutral-Happy"
                 elif anger == max_non_neutral: valence = "Neutral-Angry"
                 elif sadness == max_non_neutral: valence = "Neutral-Sad"
                 else: valence = "Neutral"
            else:
                 valence = "Neutral"

        # Engagement
        engagement_raw = (happiness * 1.0) + (surprise * 0.9) + (anger * 0.8) + (fear * 0.7) + (neutral * 0.1) + (sadness * 0.2)
        engagement_score = min(max(int(engagement_raw * 100), 0), 100)

        return {
            "emotions": {
                "happiness": round(happiness, 4),
                "neutral": round(neutral, 4),
                "surprise": round(surprise, 4),
                "anger": round(anger, 4),
                "sadness": round(sadness, 4),
                "fear": round(fear, 4),
                "disgust": round(disgust, 4)
            },
            "valence": valence,
            "engagement_score": engagement_score,
            "all_scores": scores
        }
