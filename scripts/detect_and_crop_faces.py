import os
import cv2
from mtcnn import MTCNN
from PIL import Image

def crop_faces_from_folder(input_folder, output_folder):
    detector = MTCNN()

    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img)

            for i, face in enumerate(faces):
                x, y, w, h = face['box']
                face_crop = img[y:y+h, x:x+w]
                try:
                    face_crop = Image.fromarray(face_crop).resize((96, 96))
                    save_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face{i}.jpg")
                    face_crop.save(save_path)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")

# 👇 Set paths
input_real = "extracted_frames/real"
input_fake = "extracted_frames/fake"
output_real = "cropped_faces/real"
output_fake = "cropped_faces/fake"

crop_faces_from_folder(input_real, output_real)
crop_faces_from_folder(input_fake, output_fake)
