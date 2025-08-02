# streamlit_app.py
import streamlit as st # <<<< FIX IS HERE! Change 'streamlit_app' to 'streamlit'
import requests
from PIL import Image
import io

# --- Configuration ---
# Make sure this URL matches where your FastAPI app is running
FASTAPI_URL = "http://localhost:8000/predict" # Use localhost for local testing

st.set_page_config(
    page_title="Pest Classification App",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🌿 Agricultural Pest Classifier")
st.write("Upload an image of a plant/crop to classify the pest (if any).")

st.markdown("""
    ---
    **How to use:**
    1. Make sure your FastAPI backend is running on `http://localhost:8000`.
    2. Upload an image file below.
    3. The app will send the image to the FastAPI model for prediction.
    4. View the predicted pest class and confidence score.
    ---
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image:")
    image = Image.open(uploaded_file)
    st.image(image, caption='Image for prediction', use_column_width=True)

    # Convert image to bytes for sending to FastAPI
    img_bytes = io.BytesIO(uploaded_file.getvalue())
    files = {'file': (uploaded_file.name, img_bytes, uploaded_file.type)}

    st.subheader("Prediction Results:")
    with st.spinner("Classifying pest..."):
        try:
            # Send the image to the FastAPI endpoint
            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                prediction = response.json()
                st.success("Prediction Successful!")
                st.write(f"**Predicted Class:** {prediction['predicted_class']}")
                st.write(f"**Confidence:** {prediction['confidence']:.2f}%") # Format to 2 decimal places

            elif response.status_code == 400:
                st.error(f"Prediction Error (400 Bad Request): {response.json().get('detail', 'Invalid input')}")
            elif response.status_code == 503:
                st.error(f"Prediction Error (503 Service Unavailable): {response.json().get('detail', 'Model not ready')}")
                st.info("Please ensure your FastAPI backend is running and the model loaded successfully.")
            else:
                st.error(f"An unexpected error occurred: Status {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend.")
            st.info("Please ensure your FastAPI application is running at "
                    f"`{FASTAPI_URL.replace('/predict', '')}` and accessible.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Please upload an image to get a pest classification.")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a FastAPI backend for pest classification powered by a pre-trained "
    "machine learning model. The front-end is built with Streamlit."
)