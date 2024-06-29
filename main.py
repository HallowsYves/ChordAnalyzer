import mediapipe as mp
import cv2
import time

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2
import numpy as np

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

# Set up MediaPipe components
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)   

# Start the hand landmarker with the given options
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        # Capture the latest frame from the camera
        valid, frame = video_capture.read()
        
        if not valid:
            print("Failed to capture frame")
            break
        
        # Display the frame using OpenCV
        cv2.imshow('frame', frame)
        
        # Convert the OpenCV frame (NumPy array) to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Get the current timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Process the frame with the hand landmarker
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
