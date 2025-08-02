# mlops_pest_classifier/src/model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import json
import os

# Define constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pest_classifier1.h5')
LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_labels.json')

# CRITICAL CHANGE: Set IMAGE_SIZE to match your trained model's input
DEFAULT_IMAGE_SIZE = (150, 150) # Use this for building new models

def load_trained_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def load_class_labels():
    """Loads the class labels from a JSON file."""
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Class labels file not found at: {LABELS_PATH}")
    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)
    return class_labels

def build_new_model(num_classes, image_size=DEFAULT_IMAGE_SIZE, fine_tune_layers=None):
    """
    Builds a new MobileNetV2-based model for retraining.
    Args:
        num_classes (int): Number of output classes.
        image_size (tuple): Input image dimensions (height, width). Defaults to DEFAULT_IMAGE_SIZE.
        fine_tune_layers (int, optional): Number of top layers to fine-tune.
                                          If None, only the new head is trained.
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=image_size + (3,), # Use the specified or default image_size
        include_top=False,
        weights='imagenet'
    )

    if fine_tune_layers is None:
        # Freeze the base model
        base_model.trainable = False
    else:
        # Fine-tune a specific number of layers from the top of the base model
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Example usage (for testing or if you want to define training in this file)
if __name__ == "__main__":
    try:
        model = load_trained_model()
        labels = load_class_labels()
        print("Model and labels loaded successfully.")
        print(f"Number of classes: {len(labels)}")
    except FileNotFoundError as e:
        print(e)
        print("Run the Jupyter Notebook to train and save the model first.")

    # Example of building a new model (for retraining purposes)
    # new_model = build_new_model(num_classes=5) # This will now default to 150x150
    # new_model.summary()