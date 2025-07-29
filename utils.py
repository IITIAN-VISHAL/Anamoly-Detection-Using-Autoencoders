
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

def load_pipeline(path='pipeline.pkl'):
    return joblib.load(path)

def preprocess_input(df, pipeline):
    # Combine high/low sensor values
    # if 'Power_High' in df.columns and 'Power_Low' in df.columns:
    #     df['Power'] = (df['Power_High'] + df['Power_Low']) / 2
    #     df.drop(['Power_High', 'Power_Low'], axis=1, inplace=True)

    # if 'Current_High' in df.columns and 'Current_Low' in df.columns:
    #     df['Current'] = (df['Current_High'] + df['Current_Low']) / 2
    #     df.drop(['Current_High', 'Current_Low'], axis=1, inplace=True)

    # if 'RPM_High' in df.columns and 'RPM_Low' in df.columns:
    #     df['RPM'] = (df['RPM_High'] + df['RPM_Low']) / 2
    #     df.drop(['RPM_High', 'RPM_Low'], axis=1, inplace=True)

    # if 'S.no' in df.columns:
    #     df.drop(['S.no'], axis=1, inplace=True)

    scaled = pipeline.transform(df)
    return scaled

def load_model(model_path='Air_Cooler.keras'):
    return tf.keras.models.load_model(model_path,compile=False)

def get_reconstruction_error(model, X_scaled):
    reconstructed = model.predict(X_scaled)
    return np.mean(np.square(X_scaled - reconstructed), axis=1)

def predict_anomalies(errors, threshold=0.00842):
    return ["Anomaly" if e > threshold else "Normal" for e in errors]
