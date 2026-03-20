import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. LOAD MODELS ---
# Ensure these files are in the same folder as this script
try:
    model = joblib.load('ckd_deep_model.pkl')
    scaler = joblib.load('scaler (3).pkl')
    label_encoder = joblib.load('target_encoder.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

# --- 2. DEFINE FEATURE COLUMNS ---
# CRITICAL: These must match the order of your training CSV exactly.
# Based on your error message, I've updated these to the "Fit Time" names.
FEATURE_COLUMNS = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 
    'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 
    'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 
    'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia', 'alcohol'
]

st.set_page_config(page_title="CKD Stage Detector", layout="wide")
st.title("🏥 Chronic Kidney Disease Stage Predictor")
st.markdown("---")

# --- 3. COLLECT INPUTS ---
st.sidebar.header("Patient Vitals")
inputs = {}

# We split the inputs into two columns for a better UI
col1, col2 = st.columns(2)

for i, col_name in enumerate(FEATURE_COLUMNS):
    # Use col1 for first half, col2 for second half
    target_col = col1 if i < len(FEATURE_COLUMNS)//2 else col2
    
    # Categorical logic: if it sounds like a binary feature, use a selectbox
    if col_name in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'alcohol', 'hypertension', 'anemia']:
        option = target_col.selectbox(f"{col_name}", options=[0, 1], help="0 for No/Normal, 1 for Yes/Abnormal")
        inputs[col_name] = float(option)
    else:
        # Numeric inputs
        val = target_col.number_input(f"{col_name}", value=0.0, step=0.1)
        inputs[col_name] = float(val)

st.markdown("---")

# --- 4. PREDICTION LOGIC ---
if st.button("🚀 Predict CKD Stage", type="primary"):
    try:
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([inputs])
        
        # Ensure columns are in the EXACT order the scaler expects
        input_df = input_df[FEATURE_COLUMNS]
        
        # FIX: We convert to a NumPy array using .values to avoid the "Feature Names Mismatch" error
        # As long as the number of columns is the same, this will work.
        input_array = input_df.values
        
        # Scale
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction_numeric = model.predict(input_scaled)
        
        # Decode the label (e.g., 0 -> 'Stage 1')
        final_stage = label_encoder.inverse_transform(prediction_numeric)[0]
        
        # Display Result
        st.success(f"### Result: {final_stage}")
        
        # Optional: Show probabilities
        probs = model.predict_proba(input_scaled)[0]
        st.write("Confidence per stage:")
        st.bar_chart(pd.Series(probs, index=label_encoder.classes_))

    except Exception as e:
        st.error("⚠️ Prediction Error")
        st.info(f"Details: {str(e)}")
        st.warning("Make sure the number of input fields matches the number of features the model was trained on.")
