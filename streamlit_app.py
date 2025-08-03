# mlops_pest_classifier/streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import logging
import os
import plotly.express as px # <--- CRITICAL CHANGE: MOVED THIS IMPORT TO THE TOP

# Configure logging for Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Assuming FastAPI is running locally on port 8000
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/predict"
UPLOAD_RETRAIN_ENDPOINT = f"{FASTAPI_BASE_URL}/upload_retraining_data"
TRIGGER_RETRAIN_ENDPOINT = f"{FASTAPI_BASE_URL}/trigger_retraining"

# --- Page Setup ---
st.set_page_config(
    page_title="Agricultural Pest Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌿 Agricultural Pest Classifier")
st.subheader("Identify common agricultural pests from images and manage model retraining.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Pest Prediction", "Model Retraining & Status", "Data Insights"])

# --- Main Content Area ---

if page == "Pest Prediction":
    st.header("Predict Pest Type")
    st.write("Upload an image of a crop pest to get an instant classification.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Deprecation Warning Fix: Use use_container_width
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        try:
            # Send request to FastAPI for prediction
            response = requests.post(PREDICT_ENDPOINT, files=files)

            if response.status_code == 200:
                prediction_result = response.json()
                st.success("Prediction Successful!")
                
                # Ensure correct keys are accessed
                st.write(f"**Predicted Pest:** :green[{prediction_result.get('prediction', 'N/A')}]")
                st.write(f"**Confidence:** :green[{prediction_result.get('confidence', 0):.2%}]")

                # Display all probabilities
                st.subheader("All Probabilities")
                all_probs = prediction_result.get('all_probabilities', {})
                if all_probs:
                    probabilities_df = pd.DataFrame(
                        {'Pest Type': list(all_probs.keys()),
                         'Probability': list(all_probs.values())}
                    ).sort_values(by='Probability', ascending=False)
                    
                    st.dataframe(probabilities_df, use_container_width=True)
                else:
                    st.info("No detailed probabilities available.")

            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Error from API: {response.status_code} - {error_detail}")
                logging.error(f"Prediction API error: {response.status_code} - {error_detail}")

        # Catch specific request exceptions (network issues, etc.)
        except requests.exceptions.RequestException as e:
            st.error(f"A network or connection error occurred during prediction: {e}")
            logging.error(f"Request failed: {e}", exc_info=True)
        # Catch any other unexpected errors during response processing
        except Exception as e:
            st.error(f"An unexpected error occurred while processing the prediction result: {e}")
            logging.error(f"Error processing prediction result in Streamlit: {e}", exc_info=True)

elif page == "Model Retraining & Status":
    st.header("Model Retraining & Status")
    st.write("Upload new labeled images to contribute to model retraining, or trigger a retraining process.")

    st.subheader("1. Upload New Training Data")
    uploaded_retrain_file = st.file_uploader("Choose an image for retraining...", type=["jpg", "jpeg", "png", "webp"], key="retrain_upload")
    pest_label_input = st.text_input("Enter the correct pest label for this image (e.g., 'Aphids')", key="retrain_label")

    if uploaded_retrain_file and pest_label_input:
        if st.button("Upload for Retraining", key="upload_btn"):
            with st.spinner("Uploading image..."):
                files = {'file': (uploaded_retrain_file.name, uploaded_retrain_file.getvalue(), uploaded_retrain_file.type)}
                data = {'pest_label': pest_label_input.strip()}
                
                try:
                    response = requests.post(UPLOAD_RETRAIN_ENDPOINT, files=files, data=data)
                    if response.status_code == 200:
                        st.success(f"Successfully uploaded: {response.json().get('message')}")
                        logging.info(f"Retrain data upload successful: {response.json().get('message')}")
                    else:
                        error_detail = response.json().get("detail", "Unknown error during upload")
                        st.error(f"Error uploading data: {response.status_code} - {error_detail}")
                        logging.error(f"Retrain data upload failed: {response.status_code} - {error_detail}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error during data upload: {e}")
                    logging.error(f"Network error during upload: {e}", exc_info=True)
                except Exception as e:
                    st.error(f"An unexpected error occurred during data upload: {e}")
                    logging.error(f"Unexpected error during upload: {e}", exc_info=True)


    st.subheader("2. Trigger Model Retraining")
    st.info("Retraining can take a few minutes depending on the data size. Check your FastAPI console for progress.")
    if st.button("Trigger Retraining Now", key="trigger_retrain_btn"):
        with st.spinner("Sending retraining request..."):
            try:
                response = requests.post(TRIGGER_RETRAIN_ENDPOINT)
                if response.status_code == 200:
                    status_info = response.json()
                    st.success(f"Retraining triggered: {status_info.get('message')}")
                    logging.info(f"Retraining trigger successful: {status_info.get('message')}")
                elif response.status_code == 429: # Too Many Requests
                    status_info = response.json()
                    st.warning(f"Retraining service busy: {status_info.get('message')}")
                    logging.warning(f"Retraining service busy: {status_info.get('message')}")
                else:
                    error_detail = response.json().get("detail", "Unknown error during retraining trigger")
                    st.error(f"Error triggering retraining: {response.status_code} - {error_detail}")
                    logging.error(f"Retraining trigger failed: {response.status_code} - {error_detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error triggering retraining: {e}")
                logging.error(f"Network error triggering retraining: {e}", exc_info=True)
            except Exception as e:
                st.error(f"An unexpected error occurred triggering retraining: {e}")
                logging.error(f"Unexpected error triggering retraining: {e}", exc_info=True)


elif page == "Data Insights":
    st.header("Data Insights & Visualizations")
    st.write("Explore some insights about the dataset used for training the model.")

    st.warning("Note: These insights are based on a placeholder. For real-time insights, integration with a live database or data store would be required.")

    # Placeholder Data (replace with actual data analysis if available)
    data = {
        'Pest Type': ['Aphids', 'Army Worm', 'Beetle', 'Bollworm', 'Grasshopper'],
        'Number of Samples': [1200, 950, 800, 1100, 750],
        'Prevalence (%)': [26, 21, 18, 24, 11]
    }
    df = pd.DataFrame(data)

    st.subheader("Pest Distribution in Dataset")
    st.dataframe(df, use_container_width=True)

    # Example 1: Bar Chart of Number of Samples
    st.subheader("Number of Samples per Pest Type")
    st.bar_chart(df.set_index('Pest Type')['Number of Samples'])
    st.markdown("This chart shows the absolute count of images available for each pest type in the dataset.")

    # Example 2: Pie Chart of Prevalence
    st.subheader("Prevalence of Pest Types")
    # This line now correctly uses px because it's imported at the top
    fig = px.pie(df, values='Prevalence (%)', names='Pest Type', title='Prevalence of Pest Types in Dataset')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("This pie chart illustrates the proportional representation of each pest type, highlighting which pests are more common in the collected data.")

    # Example 3: Hypothetical Model Performance Over Time (Placeholder)
    st.subheader("Hypothetical Model Performance Trend")
    performance_data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        'Accuracy': [0.85, 0.86, 0.84, 0.87, 0.88, 0.89, 0.90]
    }
    performance_df = pd.DataFrame(performance_data)
    st.line_chart(performance_df.set_index('Month')['Accuracy'])
    st.markdown("This shows a hypothetical trend of model accuracy over recent months, suggesting continuous improvement (or degradation) with retraining efforts.")