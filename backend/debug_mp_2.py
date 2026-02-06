import mediapipe as mp
try:
    import mediapipe.python.solutions as solutions
    print("Direct import succeeded")
    print(dir(solutions))
except ImportError as e:
    print(f"Direct import failed: {e}")

try:
    print(mp.solutions)
except AttributeError:
    print("mp.solutions missing")
