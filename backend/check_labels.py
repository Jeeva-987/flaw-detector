from transformers import pipeline
import numpy as np
from PIL import Image

def check_labels():
    model_name = "dima806/facial_emotions_image_detection"
    print(f"Loading {model_name}...")
    classifier = pipeline("image-classification", model=model_name)
    
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color='red')
    
    results = classifier(img)
    print("Labels found:")
    for res in results:
        print(f"- {res['label']}")

if __name__ == "__main__":
    check_labels()
