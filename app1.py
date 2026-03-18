import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models
model = joblib.load('ckd_deep_model.pkl')
scaler = joblib.load('scaler3.pkl')
label_encoder = joblib.load('target_encoder.pkl')

# Define feature columns (must match training)
FEATURE_COLUMNS = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
]

st.title("CKD Stage Prediction")
st.write("Enter the patient features to predict the CKD stage.")

# Collect inputs
inputs = {}
for col in FEATURE_COLUMNS:
    # Handle different types of inputs: numeric and categorical
    # For simplicity, assuming all are numeric; adjust as needed
    inputs[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# When button clicked, make prediction
if st.button("Predict CKD Stage"):
    try:
        # Scale features
        input_scaled = scaler.transform(input_df)
        # Predict
        pred = model.predict(input_scaled)
        pred_label = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted CKD Stage: {pred_label}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
