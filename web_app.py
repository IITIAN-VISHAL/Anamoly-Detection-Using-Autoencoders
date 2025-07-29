
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

THRESHOLD = 0.00842 

# --- UI Elements ---
st.title("Air Cooler Anomaly Detection (TensorFlow) 🏭")
st.write("Enter the sensor values for the Air Cooler to check for anomalies.")

# Sidebar for user input
with st.sidebar:
    st.header("Input Features")
    rpm = st.number_input("RPM ", value=0.49)
    power = st.number_input("Power ", value=0.73)
    current = st.number_input("Current ", value=0.45)
# Predict button
from utils import load_model, load_pipeline, preprocess_input, get_reconstruction_error, predict_anomalies

pipeline = load_pipeline('pipeline.pkl')
model = load_model('Air_Cooler.keras')


if st.button("Detect Anomaly"):
    # 1. Create a DataFrame from the inputs
    input_data = pd.DataFrame([[power, current, rpm ]],
                              columns=['Power', 'Current', 'RPM'])

    X_scaled = preprocess_input(input_data, pipeline)
    errors = get_reconstruction_error(model, X_scaled)

    # 2. Display result
    st.subheader("Prediction Result")
    if (errors<THRESHOLD):
        st.success(f"Result: Normal ✅")
    else:
        st.error(f"Result: Anomaly Detected! ⚠️")