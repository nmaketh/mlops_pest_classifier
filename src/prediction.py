# mlops_pest_classifier/src/prediction.py

import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from io import BytesIO
import logging

# Import from our local modules
from preprocessing import preprocess_image
from model import load_trained_model, load_class_labels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for model and labels (loaded once)
model = None
class_labels = None

def _load_resources():
    """Loads the model and labels if they haven't been loaded yet."""
    global model, class_labels
    if model is None:
        try:
            model = load_trained_model()
            logging.info("Model loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Error loading model: {e}")
            raise RuntimeError("Model file not found. Please ensure it's in the 'models/' directory.")

    if class_labels is None:
        try:
            class_labels = load_class_labels()
            logging.info("Class labels loaded successfully.")
            # Verify class_labels is a list here if you want extra robustness
            if not isinstance(class_labels, list):
                logging.error("Loaded class_labels is not a list. Please ensure class_labels.json contains a JSON array.")
                raise TypeError("Loaded class_labels is not a list. Expected a JSON array of strings.")
        except FileNotFoundError as e:
            logging.error(f"Error loading class labels: {e}")
            raise RuntimeError("Class labels file not found. Please ensure it's in the 'models/' directory.")

def predict_image(image_bytes: bytes):
    """
    Makes a prediction on an image.
    Args:
        image_bytes (bytes): The raw bytes of the image file.
    Returns:
        dict: A dictionary containing the prediction results.
              Example: {"prediction": "Aphids", "confidence": 0.98, "all_probabilities": {...}}
    """
    _load_resources() # Ensure model and labels are loaded

    try:
        # Preprocess the image
        processed_image = preprocess_image(image_bytes=image_bytes)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Access class_labels using integer index (corrected line)
        predicted_label = class_labels[predicted_class_idx]

        # Generate all probabilities dictionary (corrected line)
        # Iterate through class_labels (which is now a list)
        all_probabilities = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
        
        logging.info(f"Prediction made: {predicted_label} with confidence {confidence:.2f}")

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise ValueError(f"Failed to process image or make prediction: {e}")

# Example of how to call it if running this file directly for testing
if __name__ == "__main__":
    dummy_image_path = "dummy_image.jpg"
    try:
        Image.new('RGB', (224, 224), color = 'red').save(dummy_image_path)
        with open(dummy_image_path, "rb") as f:
            dummy_image_bytes = f.read()
        
        print("Testing prediction with a dummy image:")
        result = predict_image(dummy_image_bytes)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)