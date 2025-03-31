import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, fit_scaler=True, scaler=None, label_encoder=None, is_prediction=False):
    """Preprocess the dataset: feature engineering, encoding, scaling."""
    df_processed = df.copy()

    if not is_prediction:
        df_processed['Hour'] = pd.to_datetime(df_processed['Time'], format='%I:%M:%S %p').dt.hour
        df_processed = df_processed.drop(columns=['Time', 'Date'])

    # Feature engineering
    if 'Vehicle_Density' not in df_processed.columns:
        df_processed['Vehicle_Density'] = df_processed['Total'] / 4
    if 'Heavy_Vehicle_Ratio' not in df_processed.columns:
        df_processed['Heavy_Vehicle_Ratio'] = (df_processed['BusCount'] + df_processed['TruckCount']) / df_processed['Total'].replace(0, 1)

    # Convert 'Day of the week' to numerical values
    if not is_prediction and df_processed['Day of the week'].dtype == 'object':
        day_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df_processed['Day of the week'] = df_processed['Day of the week'].map(day_mapping)

    if not is_prediction:
        if label_encoder is None:
            label_encoder = LabelEncoder()
            df_processed['Traffic Situation'] = label_encoder.fit_transform(df_processed['Traffic Situation'])
        else:
            df_processed['Traffic Situation'] = label_encoder.transform(df_processed['Traffic Situation'])
        X = df_processed.drop(columns=['Traffic Situation'])
        y = df_processed['Traffic Situation']
    else:
        X = df_processed
        y = None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, label_encoder