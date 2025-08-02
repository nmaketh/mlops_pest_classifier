# mlops_pest_classifier/src/preprocessing.py

import tensorflow as tf
import numpy as np
from PIL import Image
import io

# CRITICAL CHANGE: Set IMAGE_SIZE to match your trained model's input
IMAGE_SIZE = (150, 150)

def preprocess_image(image_path: str = None, image_bytes: bytes = None):
    """
    Loads and preprocesses an image for model prediction.
    Accepts either a file path or image bytes.
    """
    if image_path:
        img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    elif image_bytes:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS) # Use LANCZOS for quality
    else:
        raise ValueError("Either image_path or image_bytes must be provided.")

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch dimension
    # IMPORTANT: Use the correct preprocessing function for your model's base
    # If your model was trained using tf.keras.applications.mobilenet_v2.preprocess_input
    # keep it. If it was simpler (e.g., just scaling to 0-1), adjust accordingly.
    # Assuming MobileNetV2's preprocessing as per earlier discussions:
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array) # MobileNetV2 specific preprocessing (scaling to -1 to 1)
    return img_array

def load_and_preprocess_training_images(image_paths):
    """
    Loads and preprocesses multiple images for retraining.
    Returns a numpy array of preprocessed images.
    """
    images = []
    for img_path in image_paths:
        img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        images.append(img_array)

    images = np.array(images)
    # IMPORTANT: Apply the same preprocessing as for prediction
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images

# You can add more preprocessing utilities here if needed, e.g., for data augmentation