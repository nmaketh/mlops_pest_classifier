# src/prediction.py

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import io
import json
import os

class PestClassifier:
    def __init__(self, model_path='models/pest_classifier1.h5', labels_path='models/class_labels.json'):
        """
        Initializes the PestClassifier.
        Paths are relative to the project root directory, from where `uvicorn src.app:app` should be run.
        """
        self.model = None
        self.class_names = None
        self.model_path = model_path
        self.labels_path = labels_path
        self.img_height = 150 # Must match training (defined in notebook's Cell 1)
        self.img_width = 150  # Must match training (defined in notebook's Cell 1)
        self._load_model_and_labels()

    def _load_model_and_labels(self):
        """
        Loads the Keras model and class names from files.
        This method resolves paths relative to the project root.
        """
        # Determine the absolute path to the project root.
        # This script (prediction.py) is in `src/`.
        # os.path.dirname(__file__) gives `src/`'s path.
        # os.path.abspath(os.path.join(..., os.pardir)) goes up one level to the project root.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

        # Construct absolute paths for model and labels
        abs_model_path = os.path.join(project_root, self.model_path)
        abs_labels_path = os.path.join(project_root, self.labels_path)

        print(f"Attempting to load model from: {abs_model_path}")
        try:
            self.model = keras.models.load_model(abs_model_path)
            print(f"Model loaded successfully from {abs_model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {abs_model_path}. Please ensure the model file exists and is not corrupted: {e}")

        print(f"Attempting to load class names from: {abs_labels_path}")
        try:
            with open(abs_labels_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Class names loaded successfully from {abs_labels_path}: {self.class_names}")
        except Exception as e:
            raise RuntimeError(f"Error loading class names from {abs_labels_path}. Please ensure the file exists and is valid JSON: {e}")

    def preprocess_image(self, image_bytes: bytes):
        """
        Preprocesses an image from bytes for model prediction.
        This must EXACTLY match the preprocessing (especially normalization) used during training.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure 3 channels
            image = image.resize((self.img_width, self.img_height))
            img_array = tf.keras.utils.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)   # Create a batch dimension (e.g., (1, 150, 150, 3))
            img_array = img_array / 255.0  # Normalize pixel values to [0,1] - CRITICAL: MUST MATCH TRAINING
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")

    def predict(self, image_bytes: bytes):
        """
        Performs prediction on the preprocessed image.
        Returns the predicted class name and confidence, and all class probabilities.
        """
        if self.model is None or self.class_names is None:
            raise RuntimeError("Model or class names not loaded. Check __init__ and _load_model_and_labels.")

        try:
            processed_image = self.preprocess_image(image_bytes)
            predictions = self.model.predict(processed_image)
            # predictions will be a numpy array of probabilities, e.g., [[0.05, 0.9, 0.03, ...]]

            predicted_class_idx = np.argmax(predictions[0]) # Get index of highest probability
            confidence = float(predictions[0][predicted_class_idx]) * 100 # Convert to percentage

            predicted_class_name = self.class_names[predicted_class_idx]

            # Prepare all_predictions dictionary with class names and their probabilities
            all_predictions = {
                self.class_names[i]: float(predictions[0][i]) * 100 # Convert to percentage
                for i in range(len(self.class_names))
            }

            return {
                "predicted_class": predicted_class_name,
                "confidence": confidence,
                "all_predictions": all_predictions
            }
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

# Initialize the classifier globally when prediction.py is imported.
# This loads the model and labels once when the FastAPI application starts,
# avoiding reloading for every request.
print("Initializing PestClassifier...")
classifier = PestClassifier(
    model_path='models/pest_classifier1.h5', # Path relative to project root
    labels_path='models/class_labels.json'    # Path relative to project root
)

if __name__ == '__main__':
    # Simple test if prediction.py is run directly
    print("\n--- Testing PestClassifier directly ---")
    try:
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (150, 150), color='red')
        img_byte_arr = io.BytesIO()
        dummy_image.save(img_byte_arr, format='JPEG')
        dummy_image_bytes = img_byte_arr.getvalue()

        result = classifier.predict(dummy_image_bytes)
        print("Test prediction result:", result)
    except Exception as e:
        print(f"Test failed: {e}")