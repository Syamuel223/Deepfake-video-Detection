import cv2
import os

def extract_frames(video_path, output_folder, max_frames=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while success and count < max_frames:
        frame_filename = f"{video_name}_frame{count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        count += 1

    cap.release()

# Set input/output paths
real_videos = "videos/real"
fake_videos = "videos/fake"
output_real = "extracted_frames/real"
output_fake = "extracted_frames/fake"

# Extract frames from real videos
os.makedirs(output_real, exist_ok=True)
for filename in os.listdir(real_videos):
    if filename.endswith(".mp4"):
        extract_frames(os.path.join(real_videos, filename), output_real)

# Extract frames from fake videos
os.makedirs(output_fake, exist_ok=True)
for filename in os.listdir(fake_videos):
    if filename.endswith(".mp4"):
        extract_frames(os.path.join(fake_videos, filename), output_fake)
