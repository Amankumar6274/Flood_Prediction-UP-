import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model, scaler, and feature list
model = joblib.load("flood_xgb_model.pkl")
scaler = joblib.load("flood_scaler.pkl")

with open("model_features.json", "r") as f:
    expected_features = json.load(f)

st.title("Flood Prediction App")

# User inputs
district = st.selectbox("District", options=[
    'District_1', 'District_2', 'District_3', 'District_4', 'District_5',
    'District_6', 'District_7', 'District_8', 'District_9', 'District_10',
    'District_11', 'District_12', 'District_13', 'District_14', 'District_15',
    'District_16', 'District_17', 'District_18', 'District_19', 'District_20'
])

river = st.selectbox("River", options=[
    'River_1', 'River_2', 'River_3', 'River_4', 'River_5'
])

season = st.selectbox("Season", options=['Monsoon', 'Winter', 'Summer', 'Pre-monsoon'])

rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
temperature_c = st.number_input("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.1)
water_level_m = st.number_input("Water Level (m)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
soil_saturation = st.slider("Soil Saturation", min_value=0.0, max_value=1.0, value=0.5)
urbanization_index = st.slider("Urbanization Index", min_value=0.0, max_value=1.0, value=0.5)
day_of_year = st.number_input("Day of Year (1-365)", min_value=1, max_value=365, value=180)

if st.button("Predict Flood"):
    # Build input dataframe
    input_dict = {
        'rainfall_mm': [rainfall_mm],
        'temperature_c': [temperature_c],
        'water_level_m': [water_level_m],
        'soil_saturation': [soil_saturation],
        'urbanization_index': [urbanization_index],
        'day_of_year': [day_of_year],
        'district': [district],
        'river': [river],
        'season': [season]
    }
    input_df = pd.DataFrame(input_dict)

    # One-hot encode categorical features like in training
    input_df = pd.get_dummies(input_df, columns=['district', 'river', 'season'], drop_first=True)

    # Cyclical encode day_of_year
    input_df['day_sin'] = np.sin(2 * np.pi * input_df['day_of_year'] / 365)
    input_df['day_cos'] = np.cos(2 * np.pi * input_df['day_of_year'] / 365)
    input_df.drop(['day_of_year'], axis=1, inplace=True)

    # Add missing columns with 0 and reorder to match training features
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    # Scale input
    X_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(X_scaled)[0]

    if pred == 1:
        st.error("⚠️ Flood likely in this scenario.")
    else:
        st.success("✅ Flood unlikely in this scenario.")
