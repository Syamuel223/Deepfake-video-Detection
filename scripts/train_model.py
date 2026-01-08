import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 👇 Custom callback for progress bar
class TQDMProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🟢 Epoch {epoch + 1}/{self.params['epochs']}")
        self.progress_bar = tqdm(total=self.params['steps'], desc="Training", unit="step")

    def on_train_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()

# 👇 Load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            img = Image.open(path).resize((96, 96))  # smaller = faster
            img = np.array(img)
            if img.shape == (96, 96, 3):  # Ignore grayscale or broken
                images.append(img)
                labels.append(label)
    return images, labels

def load_dataset():
    real_images, real_labels = load_images_from_folder('cropped_faces/real', 0)
    fake_images, fake_labels = load_images_from_folder('cropped_faces/fake', 1)

    X = np.array(real_images + fake_images)
    y = np.array(real_labels + fake_labels)

    X = X / 255.0
    y = to_categorical(y, num_classes=2)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# 👇 Define CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())  # After 3 MaxPooling2D layers, output shape is (96/8)^2 * 128 = 12*12*128 = 18432
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 👇 Main training function
def main():
    print("📦 Loading data...")
    X_train, X_val, y_train, y_val = load_dataset()
    print(f"✅ Loaded {len(X_train)} training and {len(X_val)} validation samples.")

    print("🛠️ Building model...")
    model = build_model()

    print("🚀 Training model...")
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,               # 👈 Reduced for faster testing
        batch_size=32,
        callbacks=[TQDMProgressBar()],
        verbose=0               # 👈 Turn off default logging
    )
    print(f"⏱️ Training completed in {(time.time() - start)/60:.2f} minutes.")

    os.makedirs("models", exist_ok=True)
    model.save("models/deepfake_cnn_model.h5")
    print("✅ Model saved as 'models/deepfake_cnn_model.h5'")

if __name__ == "__main__":
    main()
