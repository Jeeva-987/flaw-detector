import numpy as np
from hsemotion.facial_emotions import HSEmotionRecognizer

class EmotionDetector:
    def __init__(self, model_name='enet_b0_8_best_vgaf'):
        """
        Initializes the EmotionDetector with HSEmotion (ENet-B0).
        """
        print(f"Loading HSEmotion model: {model_name}...")
        try:
            self.model = HSEmotionRecognizer(model_name=model_name, device='cpu')
            print("HSEmotion model loaded.")
        except Exception as e:
            print(f"Error loading HSEmotion: {e}")
            self.model = None

    def detect_emotion(self, image_array):
        """
        Detects emotion from a numpy image array (BGR or RGB).
        Returns a dictionary of metrics.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        # HSEmotion expects RGB usually, but let's check input.
        # Assuming image_array is BGR (standard OpenCV)
        # HSEmotion internal code often converts or expects RGB.
        # The library usually handles face crops directly.
        
        # Predict
        # predict_emotions returns (primary_emotion, scores_dict)
        try:
            # properly handle BGR/RGB if needed. 
            # safe bet: convert to RGB as most DL models expect it
            if isinstance(image_array, np.ndarray):
                img = image_array[..., ::-1] # BGR to RGB
            else:
                img = np.array(image_array)

            emotion, scores = self.model.predict_emotions(img, logits=False)
            
            # scores is a dict like {'Anger': 0.01, 'Contempt':...}
            # HSEmotion labels: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
            
            # Normalize to lower case for consistency
            scores = {k.lower(): v for k, v in scores.items()}
            
            happiness = scores.get('happiness', 0.0)
            neutral = scores.get('neutral', 0.0)
            surprise = scores.get('surprise', 0.0)
            anger = scores.get('anger', 0.0)
            sadness = scores.get('sadness', 0.0)
            disgust = scores.get('disgust', 0.0)
            fear = scores.get('fear', 0.0)
            
            # Valence and Engagement Logic
            val_pos = happiness
            val_neg = anger + sadness + disgust + fear
            val_neu = neutral + surprise
            
            if val_pos > val_neg and val_pos > val_neu:
                valence_label = "Positive"
            elif val_neg > val_pos:
                valence_label = "Negative"
            else:
                valence_label = "Neutral"

            # Engagement Score
            engagement_raw = (happiness * 1.0) + (surprise * 0.9) + (anger * 0.8) + (fear * 0.7) + (neutral * 0.1) + (sadness * 0.2)
            engagement_score = min(max(int(engagement_raw * 100), 0), 100)

            return {
                "emotions": {
                    "happiness": round(happiness, 4),
                    "neutral": round(neutral, 4),
                    "surprise": round(surprise, 4),
                    "anger": round(anger, 4)
                },
                "valence": valence_label,
                "engagement_score": engagement_score,
                "all_scores": scores
            }

        except Exception as e:
            print(f"HSEmotion Pred Error: {e}")
            return {"error": str(e)}
