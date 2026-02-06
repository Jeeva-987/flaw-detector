import mediapipe as mp
print(dir(mp))
try:
    print(mp.solutions)
    print("mp.solutions exists")
except AttributeError as e:
    print(f"Error: {e}")
