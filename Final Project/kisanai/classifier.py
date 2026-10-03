"""
kisanai/classifier.py
ResNet50 classifier for 7 Solanaceae disease/healthy classes.
Includes full class distribution logging for runtime diagnosis.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Detect project root directory dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DISEASE_PATH = os.path.join(BASE_DIR, "model", "kisanai_resnet50_head.h5")

# Class order strictly aligned with ImageDataGenerator alphabetical index mapping
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


class DiseaseClassifier:
    def __init__(self, model_path=DEFAULT_DISEASE_PATH):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Disease model file not found at: {self.model_path}")
        
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = CLASS_NAMES

    def predict(self, pil_image: Image.Image):
        # 1. Resize and convert to RGB (matches keras_image.load_img)
        img = pil_image.convert("RGB").resize((224, 224))
        
        # 2. Convert to float32 batch array
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        # 3. Apply standard ResNet50 ImageNet normalization (zero-centered mean subtraction + RGB to BGR)
        x_processed = preprocess_input(x)

        # 4. Predict
        preds = self.model.predict(x_processed, verbose=0)[0]

        # 5. Diagnostic terminal output: inspect distribution across all classes
        print("\n" + "=" * 45)
        print("DISEASE CLASSIFIER PROBABILITY BREAKDOWN")
        print("=" * 45)
        for name, prob in zip(self.class_names, preds):
            print(f"  {name:<25}: {prob * 100:6.2f}%")
        print("=" * 45 + "\n")

        # 6. Extract top prediction
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        full_label = self.class_names[top_idx]

        # Split into crop and clean disease names
        parts = full_label.split("___")
        crop = parts[0]
        disease = parts[1].replace("_", " ")

        return crop, disease, confidence