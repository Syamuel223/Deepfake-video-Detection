from PIL import Image
import numpy as np
import os

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).resize((96, 96))  # ✅ Resize to 96x96
                img = np.array(img)
                if img.shape == (96, 96, 3):
                    images.append(img / 255.0)  # ✅ Normalize
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return images, labels
