from tensorflow.keras.models import load_model

model = load_model('models/deepfake_cnn_model.h5')
model.summary()
print("Expected input shape:", model.input_shape)
