# scripts/predicts.py
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from collections import Counter
from tensorflow.keras.models import load_model

def load_trained_model(model_path="models/deepfake_cnn_model.h5"):
    return load_model(model_path)

def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success, frame = cap.read()

    while success and count < max_frames:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = cap.read()
        count += 1

    cap.release()
    return frames

def detect_and_crop_face(frame, target_size=(96, 96)):  # 👈 Ensure it's 96x96
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if faces:
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face = frame[y:y + height, x:x + width]
        face_img = Image.fromarray(face).resize(target_size)
        return np.array(face_img)
    return None

def predict_video(video_path, model):
    frames = extract_frames(video_path)
    predictions = []

    for i, frame in enumerate(frames):
        face = detect_and_crop_face(frame)
        if face is not None:
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            pred = np.argmax(model.predict(face), axis=1)[0]
            predictions.append(pred)
        else:
            predictions.append(None)

    return predictions
