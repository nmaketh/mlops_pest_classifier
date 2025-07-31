# src/app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from SRC.predictions import PestClassifier # Import our prediction logic

# Define model and class label paths
# Ensure these paths are correct relative to where app.py will be run
# If running from project root, they should be:
MODEL_PATH = './src/models/pest_classifier.h5'
CLASS_LABELS_PATH = './src/models/class_labels.json'

# Initialize the FastAPI app
app = FastAPI(
    title="Pest Classification API",
    description="API for classifying agricultural pests from images.",
    version="0.1.0",
)

# Load the model globally when the app starts
try:
    classifier = PestClassifier(MODEL_PATH, CLASS_LABELS_PATH)
    print("FastAPI app: PestClassifier loaded successfully.")
except FileNotFoundError as e:
    print(f"FastAPI app startup error: {e}. Ensure model and labels exist.")
    classifier = None # Set to None to handle errors gracefully
except Exception as e:
    print(f"FastAPI app startup error: An unexpected error occurred: {e}")
    classifier = None


# Basic health check endpoint
@app.get("/", summary="Health Check", tags=["Health"])
async def root():
    """
    Returns a simple message to indicate the API is running.
    """
    return {"message": "Pest Classification API is running!"}

# Prediction endpoint
@app.post("/predict", summary="Predict Pest Class", tags=["Prediction"])
async def predict_pest(file: UploadFile = File(...)):
    """
    Predicts the pest class from an uploaded image.

    - **file**: An image file (e.g., JPG, PNG).
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. API is not ready.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Preprocess the image
        preprocessed_image = classifier.preprocess_image(image_bytes)

        # Make prediction
        predicted_class, confidence = classifier.predict(preprocessed_image)

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# This block allows you to run the FastAPI app directly using 'python src/app.py'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)