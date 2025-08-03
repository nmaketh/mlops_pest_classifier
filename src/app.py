from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # <--- ADD THIS IMPORT
from pydantic import BaseModel
import uvicorn
import os
import json
import logging
import shutil
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf # <--- ADD THIS IMPORT (It's used in _perform_retraining but not imported)

# Import from our local modules
from prediction import predict_image # This also lazy-loads the model and labels
from model import load_trained_model, build_new_model, load_class_labels # For retraining logic
from preprocessing import load_and_preprocess_training_images # For retraining preprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Agricultural Pest Classifier API",
    description="API for classifying agricultural pests from images and managing model retraining.",
    version="1.0.0"
)

# --- ADD CORS MIDDLEWARE HERE ---
origins = [
    "http://localhost",
    "http://localhost:8501",
    "https://mlopspestclassifier-7ksxnvfahqf9zpmbwpdwyj.streamlit.app/", # REPLACE with your actual Streamlit Cloud app URL
    # If you have other domains where your Streamlit app might run, add them here.
    # For initial testing, you could use a wildcard, but it's less secure for production:
    # "https://*.streamlit.app", # Allows all Streamlit Cloud apps (less secure)
    # "*" # Allows all origins (NOT recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- END CORS MIDDLEWARE ADDITION ---


# --- Global Paths and Directories ---
NEW_TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'new_training_data')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pest_classifier1.h5')
LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_labels.json')

# Create the directory for new training data if it doesn't exist
os.makedirs(NEW_TRAINING_DATA_DIR, exist_ok=True)

# --- Retraining Executor (for background tasks) ---
# Limit to 1 worker for sequential retraining to prevent resource contention
executor = ThreadPoolExecutor(max_workers=1)

# --- API Models ---
class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: dict

class RetrainStatus(BaseModel):
    status: str
    message: str
    new_model_accuracy: Optional[float] = None
    errors: Optional[str] = None

# --- API Routes ---

@app.get("/", summary="Health Check")
async def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"message": "Agricultural Pest Classifier API is running!"}

@app.post("/predict", response_model=PredictionResult, summary="Predict Pest from Image")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file and returns the predicted agricultural pest,
    its confidence, and probabilities for all classes.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        image_bytes = await file.read()
        prediction_result = predict_image(image_bytes)
        return prediction_result
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/upload_retraining_data", summary="Upload Data for Retraining")
async def upload_retraining_data(
    file: UploadFile = File(...),
    pest_label: str = Form(..., description="The known label for the pest in this image (e.g., 'Aphids')")
):
    """
    Uploads a single image with its known pest label to be used for future model retraining.
    The image is saved to a structured directory within 'data/new_training_data'.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    save_dir = os.path.join(NEW_TRAINING_DATA_DIR, pest_label.strip().replace(" ", "_")) # Sanitize label for directory name
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Uploaded {file.filename} for label '{pest_label}' to {file_path}")
        return {"message": f"File '{file.filename}' uploaded successfully for retraining under label '{pest_label}'."}
    except Exception as e:
        logging.error(f"Failed to upload file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")

@app.post("/trigger_retraining", response_model=RetrainStatus, summary="Trigger Model Retraining")
async def trigger_retraining():
    """
    Initiates the model retraining process using the newly uploaded data.
    This is a long-running operation and runs in a background thread to avoid blocking the API.
    """
    # Check if a retraining is already in progress by checking the executor's queue size
    if executor._work_queue.qsize() > 0: # Checks if there are tasks waiting or running
        logging.info("Retraining already in progress. Request skipped.")
        return JSONResponse(status_code=429, content={"status": "busy", "message": "Retraining already in progress. Please wait."})

    try:
        # Submit the retraining task to the thread pool executor
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(executor, _perform_retraining)
        
        # Optionally, you can add a callback to the future to log results
        future.add_done_callback(_retraining_completion_callback)
        
        logging.info("Model retraining process initiated in background.")
        return RetrainStatus(status="success", message="Model retraining initiated. Check server logs for progress.")
    except Exception as e:
        logging.error(f"Error initiating retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {e}")

def _retraining_completion_callback(future):
    """Callback function executed when the retraining future completes."""
    try:
        result = future.result() # Get the result (RetrainStatus object) or raise exception if failed
        if result.status == "success":
            logging.info(f"Retraining completed successfully. New accuracy: {result.new_model_accuracy:.4f}")
        elif result.status == "skipped":
            logging.warning(f"Retraining skipped: {result.message}")
        else: # status == "failed"
            logging.error(f"Retraining failed: {result.message}. Errors: {result.errors}")
    except Exception as e:
        logging.error(f"Retraining background task failed unexpectedly: {e}", exc_info=True)


def _perform_retraining():
    """
    Internal function to handle the actual retraining process.
    This function is run in in a separate thread by the executor.
    """
    logging.info("Starting model retraining process in background...")
    try:
        # Load existing labels (this will now be a list of strings from class_labels.json)
        class_labels_list = load_class_labels()
        if not isinstance(class_labels_list, list):
            raise TypeError("Loaded class_labels is not a list. Expected a JSON array of strings.")

        # Create a mapping from label name to integer index for new data
        label_to_index = {label: idx for idx, label in enumerate(class_labels_list)}

        # Gather all new training data
        new_data_images = []
        new_data_labels_indices = [] # Store integer indices here

        for label_name_dir in os.listdir(NEW_TRAINING_DATA_DIR):
            label_dir_path = os.path.join(NEW_TRAINING_DATA_DIR, label_name_dir)
            if os.path.isdir(label_dir_path):
                # Use the directory name as the label
                label_name = label_name_dir.replace("_", " ") # Convert sanitized dir name back to readable label
                
                # If new class detected, add it to our labels list and mapping
                if label_name not in label_to_index:
                    new_idx = len(class_labels_list)
                    class_labels_list.append(label_name) # Add new label to the list
                    label_to_index[label_name] = new_idx
                    logging.warning(f"New class '{label_name}' detected. Assigning index {new_idx}.")
                    # Save updated labels list immediately so subsequent uploads or retraining use it
                    with open(LABELS_PATH, 'w') as f:
                        json.dump(class_labels_list, f, indent=4) # Save as list

                current_label_idx = label_to_index[label_name]
                
                for image_filename in os.listdir(label_dir_path):
                    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        image_path = os.path.join(label_dir_path, image_filename)
                        new_data_images.append(image_path)
                        new_data_labels_indices.append(current_label_idx) # Append integer index

        if not new_data_images:
            logging.warning("No new training images found in '%s'. Retraining skipped.", NEW_TRAINING_DATA_DIR)
            return RetrainStatus(status="skipped", message="No new training data found.")

        logging.info(f"Found {len(new_data_images)} new images for retraining across {len(set(new_data_labels_indices))} classes.")

        # Preprocess new images
        processed_new_images = load_and_preprocess_training_images(new_data_images)
        
        # Convert labels to one-hot encoding
        num_classes = len(class_labels_list) # Use the updated list length for one-hot encoding
        new_data_labels_one_hot = tf.keras.utils.to_categorical(new_data_labels_indices, num_classes=num_classes)

        # Load the existing model to continue training or build a new one if necessary
        try:
            model = load_trained_model() # This reloads the model into the current thread's memory
            logging.info("Loaded existing model for retraining.")
            
            # Important: If new classes were added, the output layer might need to be rebuilt.
            # This is a simplified approach; in real-world, you might save and reload
            # the entire model or have more sophisticated transfer learning logic.
            if model.layers[-1].units != num_classes:
                logging.info(f"Adjusting model output layer from {model.layers[-1].units} to {num_classes} classes.")
                # Assuming the base model is the first layer of the existing model
                base_model_input = model.input
                # Rebuild the top layers from the base model's output
                # This needs to be carefully adjusted based on your 'model.py' -> 'build_new_model' structure.
                # A safer approach is to directly call build_new_model with the correct number of classes.
                # For example, if your build_new_model creates a functional model from MobileNetV2 base:
                # model_base = model.layers[0] # Assuming MobileNetV2 is the first layer
                # x = model_base.output
                # x = tf.keras.layers.GlobalAveragePooling2D()(x)
                # x = tf.keras.layers.Dense(128, activation='relu')(x)
                # x = tf.keras.layers.Dropout(0.3)(x)
                # predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                # model = tf.keras.Model(inputs=model_base.input, outputs=predictions)

                # A more robust way to ensure the model structure is correct for new classes:
                # You might need to adjust 'fine_tune_layers' if you are doing full retraining.
                logging.info("Rebuilding model with new number of classes via build_new_model.")
                model = build_new_model(num_classes=num_classes) # Assuming this function handles the freezing internally
                
                # Recompile with a potentially lower learning rate for fine-tuning the head
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Lower learning rate
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                logging.info("Model recompiled with adjusted output layer.")
            
        except Exception as e:
            logging.warning(f"Could not load existing model or adapt it ({e}). Building a new one from scratch for retraining (less ideal).", exc_info=True)
            # If the model couldn't be loaded or adapted, build a new one from scratch
            model = build_new_model(num_classes=num_classes) # Build with fresh weights
            logging.info("Built a new model from scratch for retraining.")

        # Train the model with new data
        # In a real scenario, you might combine old and new data,
        # or implement more sophisticated continuous learning techniques.
        logging.info(f"Starting model fit for {len(new_data_images)} images over 10 epochs...")
        history = model.fit(
            processed_new_images,
            new_data_labels_one_hot,
            epochs=10, # Number of epochs for retraining
            batch_size=32,
            verbose=1 # Display progress in logs
        )
        
        new_accuracy = history.history['accuracy'][-1]
        logging.info(f"Model retraining completed. New accuracy: {new_accuracy:.4f}")

        # Save the retrained model
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Retrained model saved to {MODEL_SAVE_PATH}")

        # After successful retraining, it's good practice to clear the new training data.
        # This prevents accidental re-training on the same "new" data and manages storage.
        shutil.rmtree(NEW_TRAINING_DATA_DIR)
        os.makedirs(NEW_TRAINING_DATA_DIR, exist_ok=True)
        logging.info("Cleared new training data directory.")

        # Re-initialize the global model in prediction.py module so it loads the new one
        # This is a bit of a hack for simplicity in this single-file setup.
        # In a larger system, you'd use a messaging queue or a reload mechanism.
        import src.prediction
        src.prediction.model = None # Force reload
        src.prediction.class_labels = None # Force reload
        logging.info("Forced reload of model and labels in prediction module.")

        return RetrainStatus(status="success", message="Model retrained successfully.", new_model_accuracy=new_accuracy)

    except Exception as e:
        logging.error(f"Error during background retraining process: {e}", exc_info=True)
        return RetrainStatus(status="failed", message="Model retraining failed.", errors=str(e))

if __name__ == "__main__":
    # Attempt to load resources on app startup. If they fail,
    # the prediction endpoint will throw an error, prompting the user
    # to ensure the model exists.
    try:
        load_trained_model()
        load_class_labels()
        logging.info("Initial model and labels loaded successfully on app startup.")
    except Exception as e:
        logging.warning(f"Initial model/label load failed: {e}. Prediction endpoint might fail if not addressed.", exc_info=True)
        # Continue starting app, but prediction won't work until model exists.

    uvicorn.run(app, host="0.0.0.0", port=8000)
