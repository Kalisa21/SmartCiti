from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
from typing import List
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Traffic Prediction API",
    description="API for predicting traffic situations with retraining capability",
    version="1.0.0"
)

# Global variables to hold model and preprocessing objects
class ModelPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.last_trained = None

model_pipeline = ModelPipeline()

# Input data model
class TrafficInput(BaseModel):
    car_count: int
    bike_count: int
    bus_count: int
    truck_count: int

# Training data model
class TrainingData(BaseModel):
    data: List[dict]
    retrain: bool = False

# Load initial model and preprocessing objects
def load_pipeline():
    try:
        model_pipeline.model = load_model('traffic_model.keras')
        model_pipeline.scaler = joblib.load('scaler.pkl')
        model_pipeline.label_encoder = joblib.load('label_encoder.pkl')
        model_pipeline.last_trained = datetime.now()
        logger.info("Model pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        raise

# Retrain model function
def retrain_model(new_data: pd.DataFrame):
    try:
        # Prepare data
        X = new_data.drop(columns=['Traffic Situation'])
        y = model_pipeline.label_encoder.fit_transform(new_data['Traffic Situation'])
        
        # Scale features
        X_scaled = model_pipeline.scaler.fit_transform(X)
        
        # Retrain model
        model_pipeline.model.fit(
            X_scaled,
            y,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        # Update timestamp
        model_pipeline.last_trained = datetime.now()
        
        # Save updated pipeline
        model_pipeline.model.save('traffic_model.keras')
        joblib.dump(model_pipeline.scaler, 'scaler.pkl')
        joblib.dump(model_pipeline.label_encoder, 'label_encoder.pkl')
        
        logger.info("Model retrained successfully")
        return True
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_pipeline()

# Prediction endpoint
@app.post("/predict")
async def predict_traffic(input_data: TrafficInput):
    try:
        # Create input array
        total_count = input_data.car_count + input_data.bike_count + \
                     input_data.bus_count + input_data.truck_count
        
        input_array = np.array([[
            input_data.car_count,
            input_data.bike_count,
            input_data.bus_count,
            input_data.truck_count,
            total_count
        ]])
        
        # Scale input
        input_scaled = model_pipeline.scaler.transform(input_array)
        
        # Make prediction
        prediction = model_pipeline.model.predict(input_scaled)
        predicted_class = prediction.argmax(axis=1)[0]
        traffic_situation = model_pipeline.label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            "traffic_situation": traffic_situation,
            "confidence": float(prediction[0][predicted_class]),
            "last_trained": model_pipeline.last_trained.isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Retraining endpoint
@app.post("/retrain")
async def retrain(training_data: TrainingData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(training_data.data)
        
        # Validate required columns
        required_columns = {'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Traffic Situation'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in training data")
        
        # Retrain if requested or if trigger condition met
        retrain_needed = training_data.retrain or \
                        (datetime.now() - model_pipeline.last_trained).days > 30  # Example trigger: 30 days
        
        if retrain_needed:
            success = retrain_model(df)
            if success:
                return {
                    "status": "success",
                    "message": "Model retrained successfully",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail="Model retraining failed")
        else:
            return {
                "status": "skipped",
                "message": "Retraining not needed at this time",
                "last_trained": model_pipeline.last_trained.isoformat()
            }
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "last_trained": model_pipeline.last_trained.isoformat(),
        "timestamp": datetime.now().isoformat()
    }

