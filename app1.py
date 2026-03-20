import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. LOAD MODELS (Ensure names match your files)
model = joblib.load('ckd_deep_model.pkl')
scaler = joblib.load('scaler (3).pkl')
label_encoder = joblib.load('target_encoder.pkl')

# 2. THE EXACT 26 FEATURES FROM YOUR TRAINING
FEATURE_COLUMNS = [
    'serum_creatinine', 'bun', 'serum_calcium', 'ana', 'c3_c4', 'hematuria', 
    'oxalate_levels', 'urine_ph', 'blood_pressure', 'water_intake', 'months', 
    'cluster', 'physical_activity_rarely', 'physical_activity_weekly', 
    'diet_high protein', 'diet_low salt', 'smoking_yes', 'alcohol_never', 
    'alcohol_occasionally', 'painkiller_usage_yes', 'family_history_yes', 
    'weight_changes_loss', 'weight_changes_stable', 'stress_level_low', 
    'stress_level_moderate', 'ckd_pred_No CKD'
]

st.set_page_config(page_title="CKD Stage Predictor", layout="wide")
st.title("🏥 Professional CKD Stage Analysis")
st.write("Fill in the patient data below to generate a Deep Learning prediction.")

# 3. UI LAYOUT: Split inputs into 3 columns
inputs = {}
cols = st.columns(3)

for i, col_name in enumerate(FEATURE_COLUMNS):
    with cols[i % 3]:
        # Handle Categorical/Binary features with 0/1 selections
        if any(keyword in col_name for keyword in ['yes', 'never', 'occasionally', 'rarely', 'weekly', 'high', 'low', 'stable', 'moderate', 'No CKD', 'loss']):
            inputs[col_name] = st.selectbox(f"{col_name}", options=[0, 1], help="0 = No/False, 1 = Yes/True")
        else:
            # Handle Numeric features (like creatinine, blood pressure)
            inputs[col_name] = st.number_input(f"{col_name}", value=0.0, format="%.2f")

st.markdown("---")

# 4. PREDICTION LOGIC
if st.button("🚀 Run Deep Learning Diagnosis", type="primary"):
    try:
        # Create DataFrame and FORCE the exact column order
        input_df = pd.DataFrame([inputs])[FEATURE_COLUMNS]
        
        # Convert to numpy array to bypass scikit-learn's name-check warning
        input_array = input_df.values
        
        # Scale the data using your trained scaler
        input_scaled = scaler.transform(input_array)
        
        # Get prediction
        prediction = model.predict(input_scaled)
        
        # Convert numeric prediction back to Stage string (e.g., 'Stage 3')
        final_label = label_encoder.inverse_transform(prediction)[0]
        
        # Output result
        st.success(f"### Predicted Clinical Status: {final_label}")
        
        # Optional: Probability visualization
        probs = model.predict_proba(input_scaled)[0]
        st.write("#### Probability Distribution:")
        st.bar_chart(pd.Series(probs, index=label_encoder.classes_))

    except Exception as e:
        st.error("🚨 Prediction Failure")
        st.info(f"Technical Detail: {str(e)}")
