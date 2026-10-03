"""
kisanai/validator.py
Module for validating if an uploaded image is a valid crop leaf.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf

class LeafValidator:
    def __init__(self, model_path="model/kisanai_leaf_validator.keras", threshold=0.80):
        """
        Initializes the LeafValidator.
        :param model_path: Path to the trained .keras validator model file.
        :param threshold: Confidence threshold required to accept an image as a leaf (default: 0.80).
        """
        self.threshold = threshold
        self.model_path = model_path
        self.img_size = (224, 224)

        if not os.path.exists(self.model_path):
            # Fallback path check if run from a different subfolder
            alt_path = os.path.join(os.path.dirname(__file__), "..", "model", "kisanai_leaf_validator.keras")
            if os.path.exists(alt_path):
                self.model_path = alt_path
            else:
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Load model
        self.model = tf.keras.models.load_model(self.model_path)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL image for MobileNetV2 input.
        """
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to expected dimensions
        image = image.resize(self.img_size, Image.Resampling.LANCZOS)

        # Convert to array and expand batch dimension
        img_array = tf.keras.utils.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def validate(self, image: Image.Image):
        """
        Validates whether the image contains a valid leaf.
        
        :param image: Input PIL Image.
        :return: (is_valid: bool, confidence: float)
        """
        processed_input = self.preprocess_image(image)
        raw_pred = self.model.predict(processed_input, verbose=0)

        # Handle binary classification output (Dense(1, activation='sigmoid'))
        # By default in image_dataset_from_directory:
        # Index 0 = 'leaf', Index 1 = 'non-leaf' (alphabetical order)
        # Therefore: raw_pred closer to 0 means Leaf, closer to 1 means Non-Leaf.
        # If your training mapped label 1 to leaf, adjust accordingly:
        if raw_pred.shape[-1] == 1:
            # Check probability for 'leaf' class (1 - raw_pred if alphabetical, or raw_pred)
            leaf_prob = float(1.0 - raw_pred[0][0])
        else:
            # Categorical output fallback (Dense(2, activation='softmax'))
            leaf_prob = float(raw_pred[0][0])

        # Binary check against the 0.80 threshold
        is_valid_leaf = leaf_prob >= self.threshold

        return is_valid_leaf, leaf_prob