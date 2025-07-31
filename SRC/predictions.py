# src/prediction.py

import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import io

class PestClassifier:
    def __init__(self, model_path, class_labels_path, img_height=150, img_width=150):
        self.model = self._load_model(model_path)
        self.class_names = self._load_class_labels(class_labels_path)
        self.img_height = img_height
        self.img_width = img_width

    def _load_model(self, model_path):
        """Loads the pre-trained Keras model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"Loading model from: {model_path}")
        return tf.keras.models.load_model(model_path)

    def _load_class_labels(self, class_labels_path):
        """Loads class labels from a JSON file."""
        if not os.path.exists(class_labels_path):
            raise FileNotFoundError(f"Class labels file not found at: {class_labels_path}")
        print(f"Loading class labels from: {class_labels_path}")
        with open(class_labels_path, 'r') as f:
            return json.load(f)

    def preprocess_image(self, image_bytes: bytes):
        """
        Preprocesses an image byte stream for model prediction.
        Resizes to target dimensions and normalizes pixel values.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure 3 channels
            image = image.resize((self.img_width, self.img_height))
            img_array = tf.keras.utils.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch
            img_array = img_array / 255.0  # Normalize to [0,1] as per model training
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")

    def predict(self, preprocessed_image_array):
        """
        Makes a prediction using the loaded model.
        Returns the predicted class name and confidence.
        """
        predictions = self.model.predict(preprocessed_image_array)
        score = tf.nn.softmax(predictions[0]) # Apply softmax to convert logits to probabilities
        predicted_class_id = np.argmax(score)
        predicted_class_name = self.class_names[predicted_class_id]
        confidence = float(np.max(score))

        return predicted_class_name, confidence

if __name__ == '__main__':
    # This block is for local testing of the prediction class
    # Make sure to run your notebook first to save the model and class labels
    # Adjust paths if necessary for local testing
    MODEL_PATH = '../src/models/pest_classifier.h5'
    CLASS_LABELS_PATH = '../src/models/class_labels.json'

    try:
        classifier = PestClassifier(MODEL_PATH, CLASS_LABELS_PATH)
        print("PestClassifier initialized successfully.")

        # Example: Predict using a dummy image (replace with a real image path if testing locally)
        # You'd load actual image bytes here if testing as if from an API call
        # For a quick test, let's create a dummy image array
        dummy_image_array = np.random.rand(1, classifier.img_height, classifier.img_width, 3).astype(np.float32)
        dummy_image_array = dummy_image_array / 255.0 # Normalize

        predicted_class, confidence = classifier.predict(dummy_image_array)
        print(f"Dummy prediction: Class = {predicted_class}, Confidence = {confidence:.2f}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you've run the Jupyter notebook and saved the model/labels.")
    except Exception as e:
        print(f"An unexpected error occurred during local testing: {e}")
