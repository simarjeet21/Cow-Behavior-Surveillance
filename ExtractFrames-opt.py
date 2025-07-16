import cv2
import os
import torch  # PyTorch for YOLOv5


def extract_frames(video_path, output_dir_with_cows, frame_rate=1, model=None, confidence_threshold=0.3):
    """
    Extract frames from a video file and save them as image files only if a cow is detected.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_with_cows, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration = total_frames / fps  # Duration in seconds
    print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f} seconds")

    # Calculate the frame interval for saving 1 frame every `frame_rate` seconds
    frame_interval = fps * frame_rate  # How many frames to skip
    frame_count = 0
    saved_count_with_cows = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames at the defined frame_rate
        if frame_count % frame_interval == 0:
            results = model(frame)  # Apply YOLOv5 model to the frame
            detections = results.xywh[0]  # Get the detected objects (x, y, w, h, confidence, class)

            # Check if any detection corresponds to a cow (class 19) with sufficient confidence
            cows_detected = any(detection[5] == 19 and detection[4] >= confidence_threshold for detection in detections)

            if cows_detected:
                # Save frames with cows
                frame_name = f"frame_{saved_count_with_cows:05d}.jpg"
                frame_path = os.path.join(output_dir_with_cows, frame_name)
                cv2.imwrite(frame_path, frame)
                print(f"Saved (with cow): {frame_path}")
                saved_count_with_cows += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count_with_cows} frames saved with cows.")

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the small, fast version

# Set paths for your MacBook
video_path = "./videoplayback.mp4.mp4"  # Replace 'yourusername' with your actual macOS username
output_dir_with_cows = "./FramesWithCow"  # Output directory
frame_rate = 5  # Save 1 frame every 5 seconds
confidence_threshold = 0.3  # Minimum confidence for cow detection

# Run the frame extraction function
extract_frames(video_path, output_dir_with_cows, frame_rate, model, confidence_threshold)