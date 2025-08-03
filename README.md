🌾 Automated Crop Pest Classifier: An End-to-End ML Pipeline
Project Overview
This project showcases a complete Machine Learning pipeline designed to automatically classify agricultural pests from images. It demonstrates the entire ML lifecycle: from offline model development to scalable cloud deployment, interactive user interface (UI), and continuous monitoring. The solution specifically handles non-tabular (image) data.

As a researcher in agricultural pest management, I have created and made available an Agricultural Pest Image Dataset containing images of 12 different types of agricultural pests (Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and Weevils). These images were obtained from Flickr using its API and resized to a maximum width or height of 300px. This project utilizes an agricultural pest dataset for its development and evaluation, providing a robust solution for pest management.

✨ Key Features
Image Classification Model: A deep learning model, trained via transfer learning (MobileNetV2), capable of identifying 12 different agricultural pest species.

Robust Data Pipeline: Automated acquisition, preprocessing, and management of image datasets.

Scalable API Endpoint: A Python-based API to serve real-time predictions, deployable on any cloud platform using Docker containers for scalability.

Dynamic Model Retraining: Functionality to trigger model retraining with newly uploaded bulk data, ensuring continuous model improvement.

Interactive Streamlit UI: A user-friendly web interface for:

Live Predictions: Upload an image and get instant pest classification.

Data Visualizations: Understand the dataset's characteristics and model performance.

Pipeline Control: Buttons to initiate training and retraining processes.

System Monitoring: Display of model up-time and status.

Performance Validation: Load testing using Locust to simulate high traffic, evaluate API responsiveness, and demonstrate scalability.

📺 Video Demo: https://youtu.be/cO6982hT7VI
Watch a comprehensive walkthrough of the entire pipeline, demonstrating its functionalities, the Streamlit UI, and the deployment architecture.

Watch the Demo on YouTube

🌐 Live Application
Access the interactive Streamlit application deployed on the cloud: https://mlopspestclassifier-7ksxnvfahqf9zpmbwpdwyj.streamlit.app/

Access the Live App Here

🚀 Setup & Local Execution
To set up and run this project on your local machine, follow these steps:

Clone the Repository:

Bash

git clone https:https://github.com/nmaketh/mlops_pest_classifier
cd project-name
Create & Activate Virtual Environment (Recommended):

Bash

python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
Install Dependencies:

Bash

pip install -r requirements.txt
(Ensure requirements.txt includes: tensorflow, keras, numpy, streamlit, flask or fastapi, locust, scikit-learn, matplotlib, seaborn, kagglehub, Pillow)

Execute Jupyter Notebook:
Navigate to the notebook/ directory and run 01_image_model_training_evaluation.ipynb. This notebook handles:

Automated download of the vencerlanz09/agricultural-pests-image-dataset via kagglehub (or your dataset if you've integrated it).

Data preprocessing, model training (MobileNetV2 transfer learning), and evaluation.

Saving the trained model to models/pest_classifier1.h5.

Bash

jupyter notebook
Start the Prediction API:
This will launch the backend API server responsible for model inference.

Bash

python src/app.py  # Adjust if your main API file has a different name
Launch the Streamlit UI:
Open your web browser and navigate to the local address provided by Streamlit (usually http://localhost:8501).


Bash


streamlit run ui/app.py # Assuming your Streamlit app is in ui/app.py

📂 Project Structure

Project_name/
│
├── README.md                          # Project documentation
│
├── notebook/

│   └── 01_image_model_training_evaluation.ipynb # Detailed model development & evaluation
│
├── src/
│   ├── preprocessing.py               # Functions for image data preparation
│   ├── model.py                       # Model architecture definition
│   ├── prediction.py                  # Model loading and inference logic
│   └── app.py                         # Main API application (e.g., Flask/FastAPI)
│
├── ui/
│   └── app.py                         # Streamlit user interface
│
├── data/
│   ├── train/                         # Directory for training images (downloaded/prepared)
│   └── test/                          # Directory for testing images (downloaded/prepared)
│
└── models/
    └── pest_classifier1.h5            # Saved Keras/TensorFlow trained model
    
📊 Model & Evaluation Highlights
The 01_image_model_training_evaluation.ipynb notebook comprehensively details:

Dataset: agricultural-pests-image-dataset (12 classes).

Preprocessing: Images resized to 150x150, batched, and split into 80% training and 20% validation sets.

Model Architecture: Utilizes transfer learning with MobileNetV2 as a base, fine-tuned with custom classification layers.

Training Strategy: Initial training with a frozen base, followed by fine-tuning of deeper layers.

Evaluation Metrics: Comprehensive assessment using accuracy, precision, recall, f1-score, and confusion matrices to provide a holistic view of model performance.

Model Persistence: The final fine-tuned model is saved in .h5 format for deployment.

📈 Flood Request Simulation Results (Locust)
(NOTE TO USER: Please replace this section with your actual test results once you have performed the Locust simulations. This section is a placeholder demonstrating how to present your findings.)

To evaluate the scalability and resilience of our deployed API, we performed load testing using Locust. We simulated varying numbers of concurrent users and observed the system's performance metrics, including latency and throughput, both with single and multiple Docker containers.







(Consider adding graphs here to visually represent your RPS and latency over time or across different scaling configurations.)

🛠️ Requirements
Python 3.8+

tensorflow

keras

numpy

streamlit

flask (or fastapi)

locust

scikit-learn

matplotlib

seaborn

kagglehub

Pillow

