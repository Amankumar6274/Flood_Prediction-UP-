import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("flood_xgb_model.pkl")
scaler = joblib.load("flood_scaler.pkl")

st.set_page_config(page_title="Flood Prediction App", layout="centered")

st.title("🌊 Flood Event Prediction")

st.markdown("Enter environmental and weather details to predict the likelihood of a flood.")

# Input fields for each feature
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
river_level = st.number_input("River Level (m)", min_value=0.0, value=5.0)
temperature = st.number_input("Temperature (°C)", value=28.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0)
soil_saturation = st.number_input("Soil Saturation (%)", min_value=0.0, max_value=100.0, value=60.0)
past_flood = st.selectbox("Was there a flood in the past week?", ["No", "Yes"])
elevation = st.number_input("Elevation (meters)", value=150.0)

# Convert categorical input
past_flood_binary = 1 if past_flood == "Yes" else 0

# Prepare input for model
input_features = np.array([[rainfall, river_level, temperature, humidity,
                            wind_speed, soil_saturation, past_flood_binary, elevation]])
input_scaled = scaler.transform(input_features)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Flood likely! (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ No flood expected. (Confidence: {prob:.2%})")
