# Smartciti
# TrafficSense


# Overview

TrafficSense is a machine learning project designed to predict traffic situations (e.g., low, normal, heavy) based on vehicle counts, time of day, and day of the week. The project employs a neural network model built with TensorFlow to analyze traffic patterns and provide predictions. It includes data preprocessing, model training, evaluation, and prediction capabilities, with a front-end application deployed on Vercel for real-time traffic situation predictions.


# Dataset

The project uses a traffic dataset with the following attributes:

- Date: Date of the observation (dropped during preprocessing).
- Time: Time of the observation (e.g., 12:30:00 PM).
- Day of the week: Day of the week (e.g., Monday).
- CarCount: Number of cars observed.
- BikeCount: Number of bikes observed.
- BusCount: Number of buses observed.
- TruckCount: Number of trucks observed.
- Total: Total number of vehicles.
- Traffic Situation: Target label (e.g., low, medium, high, congested).

# Obtaining the Dataset

- Public Datasets: Search for traffic datasets on platforms like Kaggle or UCI Machine Learning Repository using terms like "traffic vehicle count" or "traffic prediction dataset."
- Synthetic Data: Generate synthetic data by running the TrafficSense.ipynb notebook.
- Custom Data: Collect your own data using traffic sensors or manual observations, matching the required format.

# Project Structure

# The project is organized as follows:
SmartCiti/
├── data/
│   ├── train/
│   │   └── traffic_data_train.csv
│   └── test/
│       └── retrain.csv
|
├── notebook/
│   └── TrafficSense.ipynb
|
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
|
├── models/
│   ├── traffic_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
|
├── README.md


# Installation

## Clone the repository:

- git clone https://github.com/kalisa21/smartciti.git

- cd SmartCiti

## Create a virtual environment and activate it:

- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required dependencies:

pip install -r requirements.txt

# Ensure dataset availability:

Place your training and testing datasets in data/train/ and data/test/, respectively, or generate synthetic data using the notebook.
Usage

# Preprocess the data:

python src/preprocessing.py

# Train the model:

python src/model.py

# Make predictions:

python src/prediction.py

# Models
This project uses the following models, trained on the traffic dataset:

- traffic_model.keras: Saved TensorFlow neural network model.
- scaler.pkl: Saved scaler for feature scaling.
- label_encoder.pkl: Saved label encoder for target labels.


# Frontend

The frontend is built with React and deployed on Vercel, offering the following features:

- Data Input: Collect vehicle counts, hour, and day of the week from users.
- Prediction Display: Show real-time traffic situation predictions.
- Integration: Communicates with a FastAPI backend deployed on Render.

# Technologies

- Frontend: React
- Backend: Python, FastAPI
- Machine Learning: TensorFlow, Scikit-Learn
- Visualization: Matplotlib, Seaborn



SmartCiti Frontend Repository :

Live Link to the Project: https://smart-citi-frontend.vercel.app/ 

SmartCiti Live Website: https://smart-citi-frontend.vercel.app/ 

Render API URL: https://fastapi-fezf.onrender.com 

# Author

Kalisa
