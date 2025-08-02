# test_model_load.py
import tensorflow as tf
import os

# Define the absolute path to your model file
# Adjust this path if you're running this script from a different directory
model_path = os.path.abspath('../models/pest_classifier1.h5')

print(f"Attempting to load model from: {model_path}")
print(f"TensorFlow version being used: {tf.__version__}")

try:
    # Attempt to load the Keras model
    model = tf.keras.models.load_model(model_path)
    print("\nSUCCESS: Model loaded successfully!")
    model.summary() # Print a summary to verify it's a Keras model

except Exception as e:
    print(f"\nERROR: Failed to load model. Reason: {e}")
    print("This indicates a problem with the .h5 file itself.")
    print("Possible causes:")
    print("  1. The .h5 file is corrupted or incomplete.")
    print("  2. It's not a valid Keras model (e.g., a different file type or format).")
    print("  3. The TensorFlow version used to save the model is significantly different.")