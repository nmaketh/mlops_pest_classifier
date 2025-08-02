# src/app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from prediction import classifier # Import the globally initialized classifier
import uvicorn
import os

app = FastAPI(
    title="Pest Classification API",
    description="API for classifying agricultural pests from images.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Called when the FastAPI application starts up.
    This ensures that the model (initialized globally in prediction.py)
    is loaded and ready to serve requests.
    """
    try:
        # The classifier is initialized globally in prediction.py, so it loads on import.
        # This startup event can be used for additional checks or logging.
        if classifier.model is None or classifier.class_names is None:
            # This case should ideally not happen if global initialization succeeded
            # but is a safeguard.
            await classifier._load_model_and_labels()
        print("FastAPI application started. Model and labels are loaded.")
    except Exception as e:
        print(f"FastAPI startup failed: {e}")
        raise RuntimeError("Failed to load ML model on startup.")

@app.get("/", response_class=HTMLResponse, summary="Root endpoint")
async def read_root():
    """
    Returns a simple HTML page with API information.
    """
    return """
    <html>
        <head>
            <title>Pest Classification API</title>
        </head>
        <body>
            <h1>Pest Classification API</h1>
            <p>Welcome to the Pest Classification API. Use the <code>/predict</code> endpoint to classify images.</p>
            <p>Go to <a href="/docs">/docs</a> for interactive API documentation (Swagger UI).</p>
            <p>Go to <a href="/redoc">/redoc</a> for alternative API documentation.</p>
        </body>
    </html>
    """

@app.post("/predict", summary="Predict pest class from image")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, and returns the predicted pest class and confidence.

    - **file**: The image file to classify (e.g., JPEG, PNG).
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        prediction_result = classifier.predict(image_bytes)
        return prediction_result
    except ValueError as ve:
        # For issues during image preprocessing (e.g., invalid image format)
        raise HTTPException(status_code=400, detail=f"Image processing error: {ve}")
    except RuntimeError as re:
        # For issues with model loading or prediction logic
        raise HTTPException(status_code=500, detail=f"Prediction service error: {re}. Please check server logs.")
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {e}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # To run this using `uvicorn`, navigate to your project root directory
    # (e.g., /workspaces/mlops_pest_classifier/) and run:
    # uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
    # DO NOT run `python app.py` directly from inside the `src/` directory.

    print("--- Running Uvicorn directly for local development (less common) ---")
    print("For proper module resolution, it's recommended to use the `uvicorn` command line from project root.")
    uvicorn.run(app, host="0.0.0.0", port=8000)