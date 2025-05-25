# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import joblib
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # --- Configuration for Districts and Rivers (MUST match your training data) ---
# # These lists are crucial for the OneHotEncoder within your preprocessor.
# # They define all possible categories that the model expects.
# all_districts_rivers = [
#     ("Agra", "Yamuna"), ("Aligarh", "Ganga"), ("Ambedkar Nagar", "Ghaghara"), ("Amethi", "Gomti"),
#     ("Azamgarh", "Ghaghara"), ("Bahraich", "Ghaghara"), ("Ballia", "Ganga"), ("Balrampur", "Rapti"),
#     ("Barabanki", "Ghaghara"), ("Basti", "Rapti"), ("Budaun", "Ganga"), ("Faizabad (Ayodhya)", "Ghaghara"),
#     ("Farrukhabad", "Ganga"), ("Fatehpur", "Ganga"), ("Ghazipur", "Ganga"), ("Gonda", "Ghaghara"),
#     ("Gorakhpur", "Rapti"), ("Hardoi", "Ganga"), ("Jalaun", "Betwa"), ("Jaunpur", "Gomti"),
#     ("Kannauj", "Ganga"), ("Kanpur Nagar", "Ganga"), ("Kushinagar", "Gandak"), ("Lakhimpur Kheri", "Sharda"),
#     ("Lucknow", "Gomti"), ("Maharajganj", "Rapti"), ("Mainpuri", "Yamuna"), ("Mau", "Ghaghara"),
#     ("Meerut", "Hindon"), ("Mirzapur", "Ganga"), ("Mirzapur", "Son"), # Added Son for Mirzapur if present in original data
#     ("Muzaffarnagar", "Ganga"), ("Pilibhit", "Sharda"),
#     ("Pratapgarh", "Ganga"), ("Prayagraj", "Ganga"), ("Rae Bareli", "Ganga"), ("Rampur", "Ramganga"),
#     ("Saharanpur", "Yamuna"), ("Sant Kabir Nagar", "Rapti"), ("Shahjahanpur", "Ramganga"), ("Shrawasti", "Rapti"),
#     ("Siddharthnagar", "Rapti"), ("Sitapur", "Sharda"), ("Sultanpur", "Gomti"), ("Unnao", "Ganga"),
#     ("Varanasi", "Ganga")
# ]
# unique_districts = sorted(list(set([d for d, r in all_districts_rivers])))
# unique_rivers = sorted(list(set([r for d, r in all_districts_rivers])))


# # --- Load Model and Preprocessor ---
# @st.cache_resource # Cache the model and preprocessor to avoid reloading on every rerun
# def load_assets():
#     try:
#         model = load_model('gru_flood_model.keras') # Or cnn_lstm_flood_model.keras if you prefer
#         preprocessor = joblib.load('preprocessor.joblib')
#         return model, preprocessor
#     except Exception as e:
#         st.error(f"Error loading model or preprocessor. Make sure 'gru_flood_model.keras' and 'preprocessor.joblib' are in the same directory. Error: {e}")
#         return None, None

# model, preprocessor = load_assets()

# # --- Define N_TIMESTEPS (MUST match your GRU model's input sequence length) ---
# N_TIMESTEPS = 5 # As used in your GRU training script

# # --- Define Feature Names (MUST match the exact order and names used during training) ---
# # This list defines the full set of features that the preprocessor expects,
# # including numerical, cyclical, and placeholders for one-hot encoded categories.
# # It's generated here for robustness within the app.
# if preprocessor:
#     numerical_features = ['rainfall_mm', 'temperature_C', 'soil_moisture', 'river_level_m',
#                           'current_dam_level_m', 'danger_level_m', 'release_status',
#                           'rainfall_mm_lag1', 'rainfall_mm_lag2', 'rainfall_mm_lag3',
#                           'river_level_m_lag1', 'river_level_m_lag2', 'river_level_m_lag3',
#                           'temperature_C_lag1', 'temperature_C_lag2', 'temperature_C_lag3',
#                           'soil_moisture_lag1', 'soil_moisture_lag2', 'soil_moisture_lag3',
#                           'current_dam_level_m_lag1', 'current_dam_level_m_lag2', 'current_dam_level_m_lag3',
#                           'release_status_lag1',
#                           'rainfall_7day_sum', 'rainfall_15day_sum', 'rainfall_30day_sum',
#                           'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
#     categorical_features = ['district', 'river']

#     # Dynamically get feature names from preprocessor (more robust)
#     # Create a dummy dataframe to pass through the preprocessor to get column names
#     dummy_data = pd.DataFrame(
#         np.zeros((1, len(numerical_features) + len(categorical_features))),
#         columns=numerical_features + categorical_features
#     )
#     dummy_data['district'] = 'Agra' # Use a valid default category
#     dummy_data['river'] = 'Yamuna' # Use a valid default category
    
#     # Preprocessor's categories must be explicitly set if not fitted on full data here
#     # This ensures consistency even if your original training data had different specific values
#     preprocessor.named_transformers_['cat'].categories_ = [unique_districts, unique_rivers]

#     transformed_dummy = preprocessor.transform(dummy_data)
#     ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
#     all_feature_names = numerical_features + list(ohe_feature_names)
# else:
#     # Fallback if preprocessor fails to load
#     all_feature_names = [] # This will cause issues if preprocessor is None

# # --- Streamlit App UI ---
# st.title("Uttar Pradesh Flood Risk Predictor 🌊")
# st.markdown("---")
# st.write("Enter current weather and river conditions for a specific location and date to predict the likelihood of a flood event.")

# # --- Input Section ---
# st.header("1. Select Location & Date")
# col1, col2, col3 = st.columns(3)
# with col1:
#     selected_district = st.selectbox("Select District:", unique_districts, key='district_input')
# with col2:
#     selected_river = st.selectbox("Select River:", unique_rivers, key='river_input')
# with col3:
#     selected_date = st.date_input("Date for Prediction:", value=datetime.date.today(), key='date_input')

# st.header("2. Enter Current Day's Conditions")
# # Using columns for a cleaner layout of numerical inputs
# col_a, col_b, col_c = st.columns(3)

# with col_a:
#     rainfall_mm = st.number_input("Rainfall (mm):", min_value=0.0, value=10.0, step=1.0, key='rainfall_input')
#     soil_moisture = st.number_input("Soil Moisture (0-1):", min_value=0.0, value=0.5, step=0.01, key='soil_moisture_input')
#     current_dam_level_m = st.number_input("Current Dam Level (m):", min_value=0.0, value=15.0, step=0.1, key='dam_level_input')

# with col_b:
#     temperature_C = st.number_input("Temperature (°C):", min_value=-20.0, value=25.0, step=0.1, key='temp_input')
#     river_level_m = st.number_input("River Level (m):", min_value=0.0, value=5.0, step=0.1, key='river_level_input')
#     release_status_input = st.selectbox("Dam Release Status:", ["No Release", "Release"], key='release_input')
#     release_status = 1 if release_status_input == "Release" else 0


# st.markdown("---")
# predict_button = st.button("Predict Flood Risk", type="primary")

# # --- Prediction Logic (on button click) ---
# if predict_button and model and preprocessor:
#     # 1. Create a dictionary for the current day's raw features
#     # Ensure all original feature names are present, even if some are derived.
#     input_data = {
#         'rainfall_mm': rainfall_mm,
#         'temperature_C': temperature_C,
#         'soil_moisture': soil_moisture,
#         'river_level_m': river_level_m,
#         'current_dam_level_m': current_dam_level_m,
#         'release_status': release_status,
#         'district': selected_district,
#         'river': selected_river
#     }

#     # Handle 'danger_level_m' - since it's not user input and no historical data, use a default
#     # You might want to assign a reasonable default based on your training data's average or common value
#     # Or based on specific district/river if you have a lookup table (not covered in basic app)
#     input_data['danger_level_m'] = 10.0 # Placeholder value for danger_level_m


#     # 2. Generate simplified lagged and aggregated features based on current input
#     # As we have no historical data, we simplify by using current day's values for lags.
#     # This is a major simplification and means the model won't truly leverage
#     # its sequence-learning for historical trends, but allows the app to run.
#     for i in range(1, N_TIMESTEPS -1): # Lags 1, 2, 3 (assuming N_TIMESTEPS=5)
#         input_data[f'rainfall_mm_lag{i}'] = rainfall_mm
#         input_data[f'temperature_C_lag{i}'] = temperature_C
#         input_data[f'soil_moisture_lag{i}'] = soil_moisture
#         input_data[f'river_level_m_lag{i}'] = river_level_m
#         input_data[f'current_dam_level_m_lag{i}'] = current_dam_level_m
#     input_data['release_status_lag1'] = release_status # Only lag1 for release_status

#     # Simplified rainfall sums
#     input_data['rainfall_7day_sum'] = rainfall_mm * 7
#     input_data['rainfall_15day_sum'] = rainfall_mm * 15
#     input_data['rainfall_30day_sum'] = rainfall_mm * 30

#     # Cyclical features from the selected date
#     day_of_year = selected_date.timetuple().tm_yday
#     month = selected_date.month
#     input_data['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.0)
#     input_data['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.0)
#     input_data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
#     input_data['month_cos'] = np.cos(2 * np.pi * month / 12.0)

#     # 3. Create a DataFrame from the input data, ensuring column order matches training
#     # Create the DataFrame with all numerical features and categorical features.
#     # The OneHotEncoder in preprocessor will expand categorical features later.
#     # Ensure the order of columns matches the numerical_features + categorical_features list used during training.
    
#     # First, gather all features in the order expected by the ColumnTransformer's parts
#     current_features_raw = {k: input_data[k] for k in numerical_features + categorical_features}
    
#     # Create a DataFrame for a single sample.
#     input_df = pd.DataFrame([current_features_raw])

#     # 4. Preprocess the input data
#     try:
#         processed_input = preprocessor.transform(input_df)
#         # Ensure it's a 2D array: (1, num_features_after_preprocessing)
#         processed_input_df = pd.DataFrame(processed_input, columns=all_feature_names)
#     except Exception as e:
#         st.error(f"Error during preprocessing. This might happen if feature names or their types don't match the preprocessor. Error: {e}")
#         st.stop()


#     # 5. Reshape for GRU model (N_TIMESTEPS sequence)
#     # Repeat the single processed feature vector N_TIMESTEPS times to form the sequence
#     # Shape: (1, N_TIMESTEPS, num_features_after_preprocessing)
#     num_features_after_preprocessing = processed_input_df.shape[1]
#     model_input = np.repeat(processed_input_df.values, N_TIMESTEPS, axis=0) # Repeats the row N_TIMESTEPS times
#     model_input = model_input.reshape(1, N_TIMESTEPS, num_features_after_preprocessing) # Reshape to (1, N_TIMESTEPS, features)

#     # 6. Make prediction
#     try:
#         prediction_proba = model.predict(model_input)[0][0] # Get the probability
#         prediction_class = (prediction_proba > 0.5).astype(int) # Get the binary class (0 or 1)

#         # 7. Display Results
#         st.header("3. Prediction Result")
#         if prediction_class == 1:
#             st.error("🚨 **HIGH FLOOD RISK PREDICTED!**")
#             st.metric(label="Predicted Flood Probability", value=f"{prediction_proba * 100:.2f}%", delta_color="off")
#         else:
#             st.success("✅ **LOW FLOOD RISK PREDICTED.**")
#             st.metric(label="Predicted Flood Probability", value=f"{prediction_proba * 100:.2f}%", delta_color="off")

#     except Exception as e:
#         st.error(f"Error during model prediction. Please check inputs or model integrity. Error: {e}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from datetime import datetime

# # Constants
# N_TIMESTEPS = 1  # Use 1 if you're predicting from a single timestep

# # Load the GRU model and preprocessor
# @st.cache_resource
# def load_model():
#     try:
#         model = tf.keras.models.load_model("gru_flood_model.h5")
#         preprocessor = joblib.load("preprocessor.pkl")
#         return model, preprocessor
#     except Exception as e:
#         st.error(f"Failed to load model or preprocessor: {e}")
#         return None, None

# model, preprocessor = load_model()

# # Streamlit UI
# st.set_page_config(page_title="Flood Risk Predictor", layout="centered")
# st.title("🌊 Flood Risk Prediction App")
# st.markdown("Enter weather and environmental conditions to assess flood risk.")

# # User Inputs
# col1, col2 = st.columns(2)
# with col1:
#     temperature = st.number_input("Temperature (°C)", value=30.0)
#     humidity = st.number_input("Humidity (%)", value=70.0)
#     wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
#     pressure = st.number_input("Pressure (hPa)", value=1010.0)
#     elevation = st.number_input("Elevation (m)", value=50.0)
# with col2:
#     rainfall = st.number_input("Rainfall (mm)", value=20.0)
#     soil_moisture = st.number_input("Soil Moisture (%)", value=45.0)
#     river_level = st.number_input("River Water Level (m)", value=5.0)
#     vegetation = st.number_input("Vegetation Index (NDVI)", value=0.5)
#     current_time = st.time_input("Time", value=datetime.now().time())
#     current_date = st.date_input("Date", value=datetime.today())

# predict_button = st.button("Predict Flood Risk")

# if predict_button and model is not None and preprocessor is not None:
#     # 1. Create input dictionary
#     input_data = {
#         'temperature': temperature,
#         'humidity': humidity,
#         'wind_speed': wind_speed,
#         'pressure': pressure,
#         'rainfall': rainfall,
#         'soil_moisture': soil_moisture,
#         'river_level': river_level,
#         'vegetation_index': vegetation,
#         'elevation': elevation,
#     }

#     # 2. Add cyclical time features
#     day_of_year = current_date.timetuple().tm_yday
#     month = current_date.month
#     input_data['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.0)
#     input_data['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.0)
#     input_data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
#     input_data['month_cos'] = np.cos(2 * np.pi * month / 12.0)

#     # 3. Create DataFrame
#     input_df = pd.DataFrame([input_data])

#     # 4. Preprocess and reshape
#     try:
#         transformed_input = preprocessor.transform(input_df)
#         reshaped_input = transformed_input.reshape((1, N_TIMESTEPS, -1))

#         # 5. Predict
#         prediction = model.predict(reshaped_input)
#         flood_risk = prediction[0][0]

#         # 6. Display result
#         st.subheader("🔍 Prediction Result")
#         if flood_risk > 0.5:
#             st.error(f"⚠️ High Flood Risk Detected: **{flood_risk:.2f}**")
#         else:
#             st.success(f"✅ Low Flood Risk: **{flood_risk:.2f}**")

#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

# elif predict_button:
#     st.warning("Please ensure the model and preprocessor files are present in the app folder.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from datetime import datetime

# # Constants
# N_TIMESTEPS = 1  # Use 1 if you're predicting from a single timestep

# # Load the GRU model and preprocessor
# @st.cache_resource
# def load_model():
#     try:
#         model = tf.keras.models.load_model("gru_flood_model.h5")  # <-- load .h5 here
#         preprocessor = joblib.load("preprocessor.pkl")
#         return model, preprocessor
#     except Exception as e:
#         st.error(f"Failed to load model or preprocessor: {e}")
#         return None, None

# model, preprocessor = load_model()

# # Streamlit UI
# st.set_page_config(page_title="Flood Risk Predictor", layout="centered")
# st.title("🌊 Flood Risk Prediction App")
# st.markdown("Enter weather and environmental conditions to assess flood risk.")

# # User Inputs
# col1, col2 = st.columns(2)
# with col1:
#     temperature = st.number_input("Temperature (°C)", value=30.0)
#     humidity = st.number_input("Humidity (%)", value=70.0)
#     wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
#     pressure = st.number_input("Pressure (hPa)", value=1010.0)
#     elevation = st.number_input("Elevation (m)", value=50.0)
# with col2:
#     rainfall = st.number_input("Rainfall (mm)", value=20.0)
#     soil_moisture = st.number_input("Soil Moisture (%)", value=45.0)
#     river_level = st.number_input("River Water Level (m)", value=5.0)
#     vegetation = st.number_input("Vegetation Index (NDVI)", value=0.5)
#     current_time = st.time_input("Time", value=datetime.now().time())
#     current_date = st.date_input("Date", value=datetime.today())

# predict_button = st.button("Predict Flood Risk")

# if predict_button and model is not None and preprocessor is not None:
#     # 1. Create input dictionary
#     input_data = {
#         'temperature': temperature,
#         'humidity': humidity,
#         'wind_speed': wind_speed,
#         'pressure': pressure,
#         'rainfall': rainfall,
#         'soil_moisture': soil_moisture,
    #     'river_level': river_level,
    #     'vegetation_index': vegetation,
    #     'elevation': elevation,
    # }

    # # 2. Add cyclical time features
    # day_of_year = current_date.timetuple().tm_yday
    # month = current_date.month
    # input_data['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.0)
    # input_data['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.0)
    # input_data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    # input_data['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # # 3. Create DataFrame
    # input_df = pd.DataFrame([input_data])

    # # 4. Preprocess and reshape
    # try:
    #     transformed_input = preprocessor.transform(input_df)
    #     reshaped_input = transformed_input.reshape((1, N_TIMESTEPS, -1))

    #     # 5. Predict
    #     prediction = model.predict(reshaped_input)
    #     flood_risk = prediction[0][0]

    #     # 6. Display result
    #     st.subheader("🔍 Prediction Result")
    #     if flood_risk > 0.5:
    #         st.error(f"⚠️ High Flood Risk Detected: **{flood_risk:.2f}**")
    #     else:
    #         st.success(f"✅ Low Flood Risk: **{flood_risk:.2f}**")

    # except Exception as e:
    #     st.error(f"Prediction failed: {e}")




# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import joblib
# from tensorflow.keras.models import load_model

# # --- Configuration for Districts and Rivers (must match training data) ---
# all_districts_rivers = [
#     ("Agra", "Yamuna"), ("Aligarh", "Ganga"), ("Ambedkar Nagar", "Ghaghara"), ("Amethi", "Gomti"),
#     ("Azamgarh", "Ghaghara"), ("Bahraich", "Ghaghara"), ("Ballia", "Ganga"), ("Balrampur", "Rapti"),
#     ("Barabanki", "Ghaghara"), ("Basti", "Rapti"), ("Budaun", "Ganga"), ("Faizabad (Ayodhya)", "Ghaghara"),
#     ("Farrukhabad", "Ganga"), ("Fatehpur", "Ganga"), ("Ghazipur", "Ganga"), ("Gonda", "Ghaghara"),
#     ("Gorakhpur", "Rapti"), ("Hardoi", "Ganga"), ("Jalaun", "Betwa"), ("Jaunpur", "Gomti"),
#     ("Kannauj", "Ganga"), ("Kanpur Nagar", "Ganga"), ("Kushinagar", "Gandak"), ("Lakhimpur Kheri", "Sharda"),
#     ("Lucknow", "Gomti"), ("Maharajganj", "Rapti"), ("Mainpuri", "Yamuna"), ("Mau", "Ghaghara"),
#     ("Meerut", "Hindon"), ("Mirzapur", "Ganga"), ("Mirzapur", "Son"),
#     ("Muzaffarnagar", "Ganga"), ("Pilibhit", "Sharda"),
#     ("Pratapgarh", "Ganga"), ("Prayagraj", "Ganga"), ("Rae Bareli", "Ganga"), ("Rampur", "Ramganga"),
#     ("Saharanpur", "Yamuna"), ("Sant Kabir Nagar", "Rapti"), ("Shahjahanpur", "Ramganga"), ("Shrawasti", "Rapti"),
#     ("Siddharthnagar", "Rapti"), ("Sitapur", "Sharda"), ("Sultanpur", "Gomti"), ("Unnao", "Ganga"),
#     ("Varanasi", "Ganga")
# ]
# unique_districts = sorted(list(set([d for d, r in all_districts_rivers])))
# unique_rivers = sorted(list(set([r for d, r in all_districts_rivers])))

# # --- Load Model and Preprocessor ---
# @st.cache_resource
# def load_assets():
#     try:
#         model = load_model('gru_flood_model.h5')
#         preprocessor = joblib.load('preprocessor.joblib')
#         return model, preprocessor
#     except Exception as e:
#         st.error(f"Error loading model or preprocessor: {e}")
#         return None, None

# model, preprocessor = load_assets()

# # --- Constants ---
# N_TIMESTEPS = 5

# numerical_features = [
#     'rainfall_mm', 'temperature_C', 'soil_moisture', 'river_level_m',
#     'current_dam_level_m', 'danger_level_m', 'release_status',
#     'rainfall_mm_lag1', 'rainfall_mm_lag2', 'rainfall_mm_lag3',
#     'river_level_m_lag1', 'river_level_m_lag2', 'river_level_m_lag3',
#     'temperature_C_lag1', 'temperature_C_lag2', 'temperature_C_lag3',
#     'soil_moisture_lag1', 'soil_moisture_lag2', 'soil_moisture_lag3',
#     'current_dam_level_m_lag1', 'current_dam_level_m_lag2', 'current_dam_level_m_lag3',
#     'release_status_lag1',
#     'rainfall_7day_sum', 'rainfall_15day_sum', 'rainfall_30day_sum',
#     'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos'
# ]
# categorical_features = ['district', 'river']

# # if preprocessor:
# #     # Set categories explicitly for OneHotEncoder to avoid mismatch
# #     preprocessor.named_transformers_['cat'].categories_ = [unique_districts, unique_rivers]
# #     # Get OneHotEncoder feature names
# #     ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
# #     all_feature_names = numerical_features + list(ohe_feature_names)
# # else:
# #     all_feature_names = []

# if preprocessor:
#     cat_transformer = preprocessor.named_transformers_['cat']
#     # Don't override categories manually
#     # Just use the original fitted categories and get feature names
#     ohe_feature_names = cat_transformer.get_feature_names_out(categorical_features)
#     all_feature_names = numerical_features + list(ohe_feature_names)
# else:
#     all_feature_names = []


# # --- Streamlit UI ---
# st.title("Uttar Pradesh Flood Risk Predictor 🌊")
# st.markdown("---")
# st.write("Enter current weather and river conditions for a specific location and date to predict flood risk.")

# st.header("1. Select Location & Date")
# col1, col2, col3 = st.columns(3)
# with col1:
#     selected_district = st.selectbox("Select District:", unique_districts)
# with col2:
#     selected_river = st.selectbox("Select River:", unique_rivers)
# with col3:
#     selected_date = st.date_input("Date for Prediction:", value=datetime.date.today())

# st.header("2. Enter Current Day's Conditions")
# col_a, col_b = st.columns(2)
# with col_a:
#     rainfall_mm = st.number_input("Rainfall (mm):", min_value=0.0, value=10.0, step=1.0)
#     soil_moisture = st.number_input("Soil Moisture (0-1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
#     current_dam_level_m = st.number_input("Current Dam Level (m):", min_value=0.0, value=15.0, step=0.1)
#     release_status_input = st.selectbox("Dam Release Status:", ["No Release", "Release"])
#     release_status = 1 if release_status_input == "Release" else 0

# with col_b:
#     temperature_C = st.number_input("Temperature (°C):", min_value=-20.0, max_value=50.0, value=25.0, step=0.1)
#     river_level_m = st.number_input("River Level (m):", min_value=0.0, value=5.0, step=0.1)

# st.markdown("---")
# predict_button = st.button("Predict Flood Risk")

# # if predict_button:
# #     if model is None or preprocessor is None:
# #         st.error("Model or preprocessor not loaded properly.")
# #     else:
# #         # Prepare input features
# #         input_data = {
# #             'rainfall_mm': rainfall_mm,
# #             'temperature_C': temperature_C,
# #             'soil_moisture': soil_moisture,
# #             'river_level_m': river_level_m,
# #             'current_dam_level_m': current_dam_level_m,
# #             'danger_level_m': 10.0,  # default placeholder
# #             'release_status': release_status,
# #             'district': selected_district,
# #             'river': selected_river
# #         }

# #         # Generate lagged features (simplified: replicate current value)
# #         for i in range(1, N_TIMESTEPS - 1):
# #             input_data[f'rainfall_mm_lag{i}'] = rainfall_mm
# #             input_data[f'temperature_C_lag{i}'] = temperature_C
# #             input_data[f'soil_moisture_lag{i}'] = soil_moisture
# #             input_data[f'river_level_m_lag{i}'] = river_level_m
# #             input_data[f'current_dam_level_m_lag{i}'] = current_dam_level_m
# #         input_data['release_status_lag1'] = release_status

# #         # Aggregate rainfall sums
# #         input_data['rainfall_7day_sum'] = rainfall_mm * 7
# #         input_data['rainfall_15day_sum'] = rainfall_mm * 15
# #         input_data['rainfall_30day_sum'] = rainfall_mm * 30

# #         # Cyclical date features
# #         day_of_year = selected_date.timetuple().tm_yday
# #         month = selected_date.month
# #         input_data['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
# #         input_data['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
# #         input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
# #         input_data['month_cos'] = np.cos(2 * np.pi * month / 12)

# #         # Create DataFrame with feature order matching training
# #         current_features_raw = {k: input_data[k] for k in numerical_features + categorical_features}
# #         input_df = pd.DataFrame([current_features_raw])

# #         try:
# #             processed_input = preprocessor.transform(input_df)
# #             processed_input_df = pd.DataFrame(processed_input, columns=all_feature_names)
# #         except Exception as e:
# #             st.error(f"Error during preprocessing: {e}")
# #             st.stop()

# #         # Reshape for GRU model: (1, N_TIMESTEPS, num_features)
# #         num_features_after_preprocessing = processed_input_df.shape[1]
# #         model_input = np.repeat(processed_input_df.values, N_TIMESTEPS, axis=0).reshape(1, N_TIMESTEPS, num_features_after_preprocessing)

# #         # Predict flood risk probability
# #         pred_prob = model.predict(model_input)[0][0]
# #         pred_class = int(pred_prob >= 0.5)

# #         st.markdown("---")
# #         st.subheader("Prediction Result:")
# #         risk_label = "High Flood Risk" if pred_class == 1 else "Low Flood Risk"
# #         risk_color = "red" if pred_class == 1 else "green"
# #         st.markdown(f"<h2 style='color:{risk_color};'>{risk_label}</h2>", unsafe_allow_html=True)
# #         st.write(f"Flood Risk Probability: {pred_prob:.2%}")
# if predict_button:
#     if model is None or preprocessor is None:
#         st.error("Model or preprocessor not loaded properly.")
#     else:
#         # Prepare input features dictionary
#         input_data = {
#             'rainfall_mm': rainfall_mm,
#             'temperature_C': temperature_C,
#             'soil_moisture': soil_moisture,
#             'river_level_m': river_level_m,
#             'current_dam_level_m': current_dam_level_m,
#             'danger_level_m': 10.0,  # placeholder, ideally user input or computed
#             'release_status': release_status,
#             'district': selected_district,
#             'river': selected_river
#         }

#         # Correct lag feature generation for lag1, lag2, lag3 explicitly
#         for i in range(1, 4):  # lag1 to lag3 inclusive
#             input_data[f'rainfall_mm_lag{i}'] = rainfall_mm
#             input_data[f'temperature_C_lag{i}'] = temperature_C
#             input_data[f'soil_moisture_lag{i}'] = soil_moisture
#             input_data[f'river_level_m_lag{i}'] = river_level_m
#             input_data[f'current_dam_level_m_lag{i}'] = current_dam_level_m
#         # release_status_lag1 only
#         input_data['release_status_lag1'] = release_status

#         # Aggregate rainfall sums
#         input_data['rainfall_7day_sum'] = rainfall_mm * 7
#         input_data['rainfall_15day_sum'] = rainfall_mm * 15
#         input_data['rainfall_30day_sum'] = rainfall_mm * 30

#         # Cyclical date features
#         day_of_year = selected_date.timetuple().tm_yday
#         month = selected_date.month
#         input_data['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
#         input_data['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
#         input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
#         input_data['month_cos'] = np.cos(2 * np.pi * month / 12)

#         # Ensure order and keys match exactly the preprocessor expected columns
#         current_features_raw = {k: input_data[k] for k in numerical_features + categorical_features}
#         input_df = pd.DataFrame([current_features_raw])

#         st.write("Input DataFrame columns:", input_df.columns.tolist())
#         st.write("Input DataFrame shape:", input_df.shape)

#         try:
#             processed_input = preprocessor.transform(input_df)
#             processed_input_df = pd.DataFrame(processed_input, columns=all_feature_names)
#         except Exception as e:
#             st.error(f"Error during preprocessing: {e}")
#             st.stop()

#         st.write("Processed DataFrame shape:", processed_input_df.shape)

#         # Prepare input for GRU model: repeat row for N_TIMESTEPS to create 3D input
#         num_features_after_preprocessing = processed_input_df.shape[1]
#         model_input = np.repeat(processed_input_df.values, N_TIMESTEPS, axis=0).reshape(1, N_TIMESTEPS, num_features_after_preprocessing)

#         st.write("Model input shape:", model_input.shape)

#         # Prediction
#         pred_prob = model.predict(model_input)[0][0]
#         pred_class = int(pred_prob >= 0.5)

#         st.markdown("---")
#         st.subheader("Prediction Result:")
#         risk_label = "High Flood Risk" if pred_class == 1 else "Low Flood Risk"
#         risk_color = "red" if pred_class == 1 else "green"
#         st.markdown(f"<h2 style='color:{risk_color};'>{risk_label}</h2>", unsafe_allow_html=True)
#         st.write(f"Flood Risk Probability: {pred_prob:.2%}")


import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
from tensorflow.keras.models import load_model

# --- Configuration for Districts and Rivers ---
all_districts_rivers = [
    ("Agra", "Yamuna"), ("Aligarh", "Ganga"), ("Ambedkar Nagar", "Ghaghara"), ("Amethi", "Gomti"),
    ("Azamgarh", "Ghaghara"), ("Bahraich", "Ghaghara"), ("Ballia", "Ganga"), ("Balrampur", "Rapti"),
    ("Barabanki", "Ghaghara"), ("Basti", "Rapti"), ("Budaun", "Ganga"), ("Faizabad (Ayodhya)", "Ghaghara"),
    ("Farrukhabad", "Ganga"), ("Fatehpur", "Ganga"), ("Ghazipur", "Ganga"), ("Gonda", "Ghaghara"),
    ("Gorakhpur", "Rapti"), ("Hardoi", "Ganga"), ("Jalaun", "Betwa"), ("Jaunpur", "Gomti"),
    ("Kannauj", "Ganga"), ("Kanpur Nagar", "Ganga"), ("Kushinagar", "Gandak"), ("Lakhimpur Kheri", "Sharda"),
    ("Lucknow", "Gomti"), ("Maharajganj", "Rapti"), ("Mainpuri", "Yamuna"), ("Mau", "Ghaghara"),
    ("Meerut", "Hindon"), ("Mirzapur", "Ganga"), ("Mirzapur", "Son"),
    ("Muzaffarnagar", "Ganga"), ("Pilibhit", "Sharda"),
    ("Pratapgarh", "Ganga"), ("Prayagraj", "Ganga"), ("Rae Bareli", "Ganga"), ("Rampur", "Ramganga"),
    ("Saharanpur", "Yamuna"), ("Sant Kabir Nagar", "Rapti"), ("Shahjahanpur", "Ramganga"), ("Shrawasti", "Rapti"),
    ("Siddharthnagar", "Rapti"), ("Sitapur", "Sharda"), ("Sultanpur", "Gomti"), ("Unnao", "Ganga"),
    ("Varanasi", "Ganga")
]
unique_districts = sorted(list(set([d for d, r in all_districts_rivers])))
unique_rivers = sorted(list(set([r for d, r in all_districts_rivers])))

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_assets():
    try:
        model = load_model('gru_flood_model.h5')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None

model, preprocessor = load_assets()

# --- Constants ---
N_TIMESTEPS = 5
numerical_features = [
    'rainfall_mm', 'temperature_C', 'soil_moisture', 'river_level_m',
    'current_dam_level_m', 'danger_level_m', 'release_status',
    'rainfall_mm_lag1', 'rainfall_mm_lag2', 'rainfall_mm_lag3',
    'river_level_m_lag1', 'river_level_m_lag2', 'river_level_m_lag3',
    'temperature_C_lag1', 'temperature_C_lag2', 'temperature_C_lag3',
    'soil_moisture_lag1', 'soil_moisture_lag2', 'soil_moisture_lag3',
    'current_dam_level_m_lag1', 'current_dam_level_m_lag2', 'current_dam_level_m_lag3',
    'release_status_lag1',
    'rainfall_7day_sum', 'rainfall_15day_sum', 'rainfall_30day_sum',
    'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos'
]
categorical_features = ['district', 'river']

if preprocessor:
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
else:
    all_feature_names = []

# --- Streamlit UI ---
st.title("Uttar Pradesh Flood Risk Predictor 🌊")
st.markdown("---")
st.write("Enter current weather and river conditions for a specific location and date to predict flood risk.")

st.header("1. Select Location & Date")
col1, col2, col3 = st.columns(3)
with col1:
    selected_district = st.selectbox("Select District:", unique_districts)
with col2:
    selected_river = st.selectbox("Select River:", unique_rivers)
with col3:
    selected_date = st.date_input("Date for Prediction:", value=datetime.date.today())

st.header("2. Enter Current Day's Conditions")
col_a, col_b = st.columns(2)
with col_a:
    rainfall_mm = st.number_input("Rainfall (mm):", min_value=0.0, value=10.0, step=1.0)
    soil_moisture = st.number_input("Soil Moisture (0-1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    current_dam_level_m = st.number_input("Current Dam Level (m):", min_value=0.0, value=15.0, step=0.1)
    release_status_input = st.selectbox("Dam Release Status:", ["No Release", "Release"])
    release_status = 1 if release_status_input == "Release" else 0

with col_b:
    temperature_C = st.number_input("Temperature (°C):", min_value=-20.0, max_value=50.0, value=25.0, step=0.1)
    river_level_m = st.number_input("River Level (m):", min_value=0.0, value=5.0, step=0.1)

st.markdown("---")
predict_button = st.button("Predict Flood Risk")

# --- Prediction Logic ---
if predict_button:
    if model is None or preprocessor is None:
        st.error("Model or preprocessor not loaded properly.")
    else:
        # Base input
        input_data = {
            'rainfall_mm': rainfall_mm,
            'temperature_C': temperature_C,
            'soil_moisture': soil_moisture,
            'river_level_m': river_level_m,
            'current_dam_level_m': current_dam_level_m,
            'danger_level_m': 10.0,  # assumed constant
            'release_status': release_status,
            'district': selected_district,
            'river': selected_river
        }

        # Add lag features
        for i in range(1, 4):
            input_data[f'rainfall_mm_lag{i}'] = rainfall_mm
            input_data[f'temperature_C_lag{i}'] = temperature_C
            input_data[f'soil_moisture_lag{i}'] = soil_moisture
            input_data[f'river_level_m_lag{i}'] = river_level_m
            input_data[f'current_dam_level_m_lag{i}'] = current_dam_level_m
        input_data['release_status_lag1'] = release_status

        # Rolling sum features
        input_data['rainfall_7day_sum'] = rainfall_mm * 7
        input_data['rainfall_15day_sum'] = rainfall_mm * 15
        input_data['rainfall_30day_sum'] = rainfall_mm * 30

        # Cyclical date features
        day_of_year = selected_date.timetuple().tm_yday
        month = selected_date.month
        input_data['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        input_data['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        input_data['month_cos'] = np.cos(2 * np.pi * month / 12)

        # DataFrame and preprocessing
        raw_df = pd.DataFrame([input_data])
        try:
            processed_input = preprocessor.transform(raw_df)
            processed_df = pd.DataFrame(processed_input, columns=all_feature_names)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # Reshape for GRU model
        model_input = np.repeat(processed_df.values, N_TIMESTEPS, axis=0).reshape(1, N_TIMESTEPS, -1)

        # Predict
        pred_prob = model.predict(model_input)[0][0]
        pred_class = int(pred_prob >= 0.5)

        st.markdown("---")
        st.subheader("Prediction Result:")
        risk_label = "High Flood Risk" if pred_class == 1 else "Low Flood Risk"
        risk_color = "red" if pred_class == 1 else "green"
        st.markdown(f"<h2 style='color:{risk_color};'>{risk_label}</h2>", unsafe_allow_html=True)
        st.write(f"Flood Risk Probability: {pred_prob:.2%}")

