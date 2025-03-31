import pandas as pd
from .preprocessing import preprocess_data  

def predict_traffic(model, scaler, label_encoder, input_data):
    total = input_data['CarCount'] + input_data['BikeCount'] + input_data['BusCount'] + input_data['TruckCount']
    vehicle_density = total / 4
    heavy_vehicle_ratio = (input_data['BusCount'] + input_data['TruckCount']) / total if total > 0 else 0
    input_df = pd.DataFrame([{
        'CarCount': input_data['CarCount'],
        'BikeCount': input_data['BikeCount'],
        'BusCount': input_data['BusCount'],
        'TruckCount': input_data['TruckCount'],
        'Total': total,
        'Hour': input_data['Hour'],
        'Day of the week': input_data['Day of the week'],
        'Vehicle_Density': vehicle_density,
        'Heavy_Vehicle_Ratio': heavy_vehicle_ratio
    }])
    feature_order = scaler.feature_names_in_
    input_df = input_df[feature_order]
    X_scaled, _, _, _ = preprocess_data(input_df, fit_scaler=False, scaler=scaler, label_encoder=label_encoder, is_prediction=True)
    pred = model.predict(X_scaled).argmax(axis=1)
    return label_encoder.inverse_transform(pred)[0]
