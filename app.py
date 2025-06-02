import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

st.set_page_config(page_title="Flood Prediction Dashboard", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("flood_dataset_realistic_final.csv", parse_dates=['date'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load("flood_rf_model.pkl")
    scaler = joblib.load("flood_scaler.pkl")
    return model, scaler

df = load_data()
model, scaler = load_model()

# -------- Sidebar Input --------
st.sidebar.title("🛠️ Predict Flood from Inputs")
st.sidebar.markdown("Enter environmental parameters:")

district = st.sidebar.selectbox("District", sorted(df['district'].unique()))
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 28.0)
river_level = st.sidebar.slider("River Level (m)", 50.0, 110.0, 85.0)
dam_level = st.sidebar.slider("Dam Level (m)", 30.0, 100.0, 70.0)
soil_moisture = st.sidebar.slider("Soil Moisture", 0.0, 1.0, 0.5)

import numpy as np
import pandas as pd
from datetime import datetime

# Inside your Streamlit app, replace your current sidebar prediction code with this:

if st.sidebar.button("Predict Flood"):
    # Step 1: Base input features from user
    input_dict = {
        'rainfall_mm': rainfall,
        'temperature_C': temperature,
        'river_level_m': river_level,
        'current_dam_level_m': dam_level,
        'soil_moisture': soil_moisture,
    }
    
    # Step 2: Add engineered date features (day_sin, day_cos)
    today = datetime.today()
    day_of_year = today.timetuple().tm_yday
    input_dict['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    input_dict['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    # Step 3: Prepare one-hot encoding for district to match model's expected features
    # Get all district columns model expects from scaler
    expected_features = scaler.feature_names_in_
    district_cols = [col for col in expected_features if col.startswith('district_')]

    # Initialize all district one-hot columns as 0
    for col in district_cols:
        input_dict[col] = 0

    # Set the user's district column to 1 if it exists
    district_col_name = f'district_{district}'
    if district_col_name in district_cols:
        input_dict[district_col_name] = 1

    # Step 4: If model expects other one-hot columns (e.g. river, season), add them here similarly
    # For example (adjust according to your feature names):
    # for river_col in [col for col in expected_features if col.startswith('river_')]:
    #     input_dict[river_col] = 0
    # river_col_name = f'river_{user_selected_river}'
    # if river_col_name in expected_features:
    #     input_dict[river_col_name] = 1

    # Step 5: Create DataFrame in the exact feature order
    input_df = pd.DataFrame([input_dict], columns=expected_features)

    # Step 6: Scale and predict
    scaled = scaler.transform(input_df)
    result = model.predict(scaled)[0]

    # Show result
    st.sidebar.success("🌊 Flood Expected!" if result == 1 else "✅ No Flood Risk")


# if st.sidebar.button("Predict Flood"):
#     input_df = pd.DataFrame([{
#         'rainfall_mm': rainfall,
#         'temperature_C': temperature,
#         'river_level_m': river_level,
#         'current_dam_level_m': dam_level,
#         'soil_moisture': soil_moisture
#     }])
#     scaled = scaler.transform(input_df)
#     result = model.predict(scaled)[0]
#     st.sidebar.success("🌊 Flood Expected!" if result == 1 else "✅ No Flood Risk")

# -------- Main Tabs --------
tab1, tab2, tab3 = st.tabs(["📊 Trends", "🗂 Bulk Prediction", "🗺 Map View"])

# --------- Tab 1: Trends ---------
with tab1:
    st.subheader("📈 Time Series Trend of Floods")
    flood_by_month = df[df['flood_event'] == 1].groupby(df['date'].dt.to_period("M")).size()
    st.line_chart(flood_by_month)

    st.subheader("📍 District-wise Flood Frequency")
    flood_dist = df[df['flood_event'] == 1]['district'].value_counts()
    st.bar_chart(flood_dist)

# --------- Tab 2: Bulk Prediction ---------
with tab2:
    st.subheader("📤 Upload CSV for Bulk Flood Prediction")

    uploaded_file = st.file_uploader("Upload CSV with required columns", type=["csv"])
    if uploaded_file:
        csv_df = pd.read_csv(uploaded_file)
        try:
            input_data = csv_df[[
                'rainfall_mm', 'temperature_C', 'river_level_m',
                'current_dam_level_m', 'soil_moisture'
            ]]
            input_scaled = scaler.transform(input_data)
            csv_df['flood_prediction'] = model.predict(input_scaled)
            st.success("✅ Prediction complete!")
            st.dataframe(csv_df.head())

            csv_download = csv_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv_download, "flood_predictions.csv", "text/csv")
        except Exception as e:
            st.error("CSV missing required columns. Please include: rainfall_mm, temperature_C, river_level_m, current_dam_level_m, soil_moisture")

# --------- Tab 3: Map ---------
with tab3:
    st.subheader("🗺 Map of Historical Flood Events")

    df_map = df.copy()
    # Fake coordinates for illustration (you should replace with actual lat/lon per district)
    district_coords = {
        'Gorakhpur': (26.76, 83.37),
        'Ballia': (25.76, 84.15),
        'Bahraich': (27.57, 81.59),
        'Gonda': (27.13, 81.96),
        'Lakhimpur Kheri': (27.94, 80.78),
        'Faizabad': (26.77, 82.15),
        'Basti': (26.79, 82.73),
        'Mau': (25.95, 83.56),
        'Sitapur': (27.57, 80.68),
        'Barabanki': (26.93, 81.20)
    }
    df_map['lat'] = df_map['district'].map(lambda x: district_coords.get(x, (0, 0))[0])
    df_map['lon'] = df_map['district'].map(lambda x: district_coords.get(x, (0, 0))[1])
    df_map = df_map[df_map['flood_event'] == 1]

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=26.8,
            longitude=82.5,
            zoom=6,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_map,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=5000,
                pickable=True
            )
        ],
        tooltip={"text": "District: {district}\nDate: {date}\nRainfall: {rainfall_mm}mm"}
    ))
