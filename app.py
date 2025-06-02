import numpy as np
import pandas as pd
import streamlit as st
import joblib

# === Load trained model and scaler ===
model = joblib.load("flood_rf_model.pkl")
scaler = joblib.load("flood_scaler.pkl")

# === Define training columns exactly as in training ===
training_columns = [
    'rainfall_mm', 'temperature_c', 'water_level_m', 'soil_saturation', 'urbanization_index',
    # Districts (District_2 to District_20)
    *['district_District_' + str(i) for i in range(2, 21)],
    # Rivers (River_2 to River_5)
    *['river_River_' + str(i) for i in range(2, 6)],
    # Seasons (exclude Monsoon)
    'season_Pre-monsoon', 'season_Summer', 'season_Winter',
    'day_sin', 'day_cos'
]

# === Preprocessing function ===
def preprocess_inputs(input_df):
    districts = ['District_' + str(i) for i in range(1, 21)]
    rivers = ['River_' + str(i) for i in range(1, 6)]
    seasons = ['Monsoon', 'Winter', 'Summer', 'Pre-monsoon']

    district_dummies = pd.get_dummies(input_df['district'], prefix='district')
    river_dummies = pd.get_dummies(input_df['river'], prefix='river')
    season_dummies = pd.get_dummies(input_df['season'], prefix='season')

    district_cols = ['district_' + d for d in districts[1:]]
    river_cols = ['river_' + r for r in rivers[1:]]
    season_cols = ['season_' + s for s in seasons if s != 'Monsoon']

    district_dummies = district_dummies.reindex(columns=district_cols, fill_value=0)
    river_dummies = river_dummies.reindex(columns=river_cols, fill_value=0)
    season_dummies = season_dummies.reindex(columns=season_cols, fill_value=0)

    input_df = input_df.drop(['district', 'river', 'season'], axis=1)
    df_final = pd.concat([input_df, district_dummies, river_dummies, season_dummies], axis=1)

    df_final['day_sin'] = np.sin(2 * np.pi * df_final['day_of_year'] / 365)
    df_final['day_cos'] = np.cos(2 * np.pi * df_final['day_of_year'] / 365)
    df_final.drop(['day_of_year'], axis=1, inplace=True)

    df_final = df_final.reindex(columns=training_columns, fill_value=0)
    return df_final

# === Streamlit UI ===
st.title("🌊 Flood Risk Predictor")

input_mode = st.radio("Choose input mode:", ["Single Input", "Bulk CSV"])

if input_mode == "Single Input":
    district = st.selectbox("District", ['District_' + str(i) for i in range(1, 21)])
    river = st.selectbox("River", ['River_' + str(i) for i in range(1, 6)])
    season = st.selectbox("Season", ['Monsoon', 'Winter', 'Summer', 'Pre-monsoon'])
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)
    temperature = st.slider("Temperature (°C)", 10, 45, 28)
    water_level = st.slider("Water Level (m)", 0, 15, 5)
    soil_sat = st.slider("Soil Saturation", 0.0, 1.0, 0.5)
    urban_idx = st.slider("Urbanization Index", 0.0, 1.0, 0.5)
    day_of_year = st.slider("Day of Year", 1, 365, 180)

    input_dict = {
        'district': [district],
        'river': [river],
        'season': [season],
        'rainfall_mm': [rainfall],
        'temperature_c': [temperature],
        'water_level_m': [water_level],
        'soil_saturation': [soil_sat],
        'urbanization_index': [urban_idx],
        'day_of_year': [day_of_year]
    }

    input_df = pd.DataFrame(input_dict)
    processed = preprocess_inputs(input_df)
    scaled = scaler.transform(processed)
    pred = model.predict(scaled)[0]

    st.success("🚨 Flood Expected!" if pred == 1 else "✅ No Flood Risk")

else:
    uploaded_file = st.file_uploader("Upload CSV with required columns", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        processed = preprocess_inputs(df)
        scaled = scaler.transform(processed)
        predictions = model.predict(scaled)
        df['Predicted Flood Event'] = predictions
        st.write(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
