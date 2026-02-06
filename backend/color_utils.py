import cv2
import numpy as np
from sklearn.cluster import KMeans

def dominant_color(img, k=3):
    # resize for speed
    if img.size == 0:
        return (0, 0, 0)
    img = cv2.resize(img, (64, 64))
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(img)
    counts = np.bincount(labels)

    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return tuple(map(int, dominant))

def get_skin_tone_prediction(img_bgr):
    """
    Predicts 'Black' or 'White' skin tone using skin isolation and brightness analysis.
    """
    if img_bgr is None or img_bgr.size == 0:
         return "Unknown", []

    # 1. Use HSV Masking to Isolate Skin
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    # These are broad ranges for human skin
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Check for reddish skin (hue wrapping 170-180)
    lower_skin2 = np.array([170, 20, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    # Combine masks
    skin_mask = cv2.add(mask1, mask2)
    
    # Extract skin pixels
    # We only care about the Value channel (Brightness)
    v_channel = hsv[:, :, 2]
    skin_brightness_values = v_channel[skin_mask > 0]
    
    # Debug/Fallback: If very few skin pixels found (<5%), assume the whole crop is skin (or bad lighting)
    pixel_count = img_bgr.shape[0] * img_bgr.shape[1]
    if len(skin_brightness_values) < (0.05 * pixel_count):
        avg_brightness = np.mean(v_channel) # fallback to whole image avg
        # Maybe it's just really dark skin or weird lighting?
    else:
        avg_brightness = np.mean(skin_brightness_values)
    
    # 2. Decision Threshold
    # Range is 0-255. 
    # Fair skin is usually > 140.
    # Darker skin is usually < 130.
    # Mid-range: 120-140 can be tricky. Let's set 135 as the split.
    threshold = 135
    
    if avg_brightness < threshold:
        prediction = "Black"
    else:
        prediction = "White"

    remedies = []
    if prediction == "Black":
        remedies = [
            "Use a broad-spectrum sunscreen (SPF 30+) daily to prevent hyperpigmentation.",
            "Incorporate Vitamin C serum to brighten skin tone.",
            "Exfoliate 1-2 times a week to remove dead skin cells.",
            "Keep skin hydrated with a gel-based moisturizer.",
            "Drink plenty of water to flush out toxins."
        ]
    else:
        # For White/Fair skin
        remedies = [
            "Apply sunscreen (SPF 50+) as fair skin burns easily.",
            "Use a gentle, hydrating cleanser.",
            "Moisturize daily to maintain skin barrier.",
            "Look for antioxidants like Niacinamide.",
            "Avoid excessive sun exposure during peak hours."
        ]

    return prediction, remedies

def color_name(bgr):
    # approximate color names (legacy function)
    b, g, r = bgr
    
    # Simple logic
    if r > 200 and g > 200 and b > 200: return "White"
    if r < 60 and g < 60 and b < 60: return "Black"
    
    return "Other"
