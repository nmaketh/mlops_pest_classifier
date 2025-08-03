from locust import HttpUser, task, between
import base64
import os

# Assuming you have a sample image to send for prediction
# You might want to place a small sample image in a 'test_data' folder
# For example, 'test_data/sample_pest.jpg'
SAMPLE_IMAGE_PATH = "data/new_training_data"

class MLUser(HttpUser):
    # Time between consecutive requests from a user (in seconds)
    wait_time = between(1, 2)

    # Replace with the actual URL of your API
    # If running locally: http://127.0.0.1:5000 (for Flask) or http://127.0.0.1:8000 (for FastAPI)
    # If deployed: Your cloud API endpoint URL
    host = "http://127.0.0.1:8000" # <--- IMPORTANT: Update this to your API's host!

    def on_start(self):
        """ on_start is called when a Locust user starts running """
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            print(f"WARNING: Sample image not found at {SAMPLE_IMAGE_PATH}. Prediction task might fail.")
            # You might want to create a dummy image or exit if this is critical
            # Example: Create a tiny dummy image for testing if not found
            try:
                from PIL import Image
                img = Image.new('RGB', (1, 1), color = 'red')
                os.makedirs(os.path.dirname(SAMPLE_IMAGE_PATH), exist_ok=True)
                img.save(SAMPLE_IMAGE_PATH)
                print(f"Created dummy image at {SAMPLE_IMAGE_PATH}")
            except ImportError:
                print("PIL not installed, cannot create dummy image.")

        # Encode the sample image to base64 once for all tasks
        with open(SAMPLE_IMAGE_PATH, "rb") as image_file:
            self.encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    @task
    def predict_image(self):
        """
        Sends a prediction request to your ML API.
        Adjust the endpoint and payload based on your API design.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "image": self.encoded_image # Assuming your API expects base64 encoded image
        }

        # Replace '/predict' with your actual prediction endpoint
        with self.client.post("/predict", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Request failed with status {response.status_code}: {response.text}")

# If your API accepts direct file upload, you might use this instead:
# @task
# def predict_image_file_upload(self):
#     with open(SAMPLE_IMAGE_PATH, "rb") as image_file:
#         files = {'file': image_file}
#         with self.client.post("/upload_and_predict", files=files, catch_response=True) as response:
#             if response.status_code == 200:
#                 response.success()
#             else:
#                 response.failure(f"Request failed with status {response.status_code}: {response.text}")