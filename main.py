import os
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Initialize FastAPI app
app = FastAPI()

# Load model, scaler, and label encoder
MODEL_PATH = "traffic_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = "Traffic.csv"  # Original dataset for retraining

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def preprocess_input(data: list):
    input_data = np.array([data])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

class TrafficData(BaseModel):
    car_count: int
    bike_count: int
    bus_count: int
    truck_count: int

@app.post("/predict")
def predict_traffic(data: TrafficData):
    total_count = data.car_count + data.bike_count + data.bus_count + data.truck_count
    input_scaled = preprocess_input([data.car_count, data.bike_count, data.bus_count, data.truck_count, total_count])
    pred_class = model.predict(input_scaled).argmax(axis=1)[0]
    traffic_label = label_encoder.inverse_transform([pred_class])[0]
    return {"Predicted Traffic Situation": traffic_label}

@app.post("/retrain")
def retrain_model():
    try:
        df = pd.read_csv(DATA_PATH)
        df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])
        df = df.drop(columns=['Time', 'Date', 'Day of the week'])
        X = df.drop(columns=['Traffic Situation'])
        y = df['Traffic Situation']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        new_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        new_model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        new_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=16, callbacks=[early_stopping], verbose=1)
        
        new_model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)
        
        global model
        model = new_model  # Reload model into memory
        return {"message": "Model retrained and updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
