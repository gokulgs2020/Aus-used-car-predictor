

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from brand_dict import brand_dict

# Load the trained model
with open("rf_lasso.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚗 Australian Used Car Price Estimator")
st.markdown("Kindly fill in the details to get an estimated car price.")

# --- INPUTS ---
# Basic inputs
kilometres = st.number_input("Kilometres Driven", min_value=0, value=50000)
# doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7])
fuel_eff = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, value=8.0)


# New or used
car_condition = st.radio("Condition", ["Used", "New"])
used_0_new_1 = 1 if car_condition == "New" else 0

# Transmission
transmission = st.radio("Transmission", ["Automatic", "Manual"])
transmission_auto = 1 if transmission == "Automatic" else 0

# Colors
color = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Red", "Others"])
color_dict = {
    'color_black': int(color == 'Black'),
    'color_white': int(color == 'White'),
    'color_gray': int(color == 'Gray'),
    'color_silver': int(color == 'Silver'),
    'color_red': int(color == 'Red'),
}

# Fuel type
fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
fuel_dict = {
    'fuel_cat_Gasoline': int(fuel_type == 'Gasoline'),
    'fuel_cat_Diesel': int(fuel_type == 'Diesel'),
    'fuel_cat_Electric': int(fuel_type == 'Electric'),
    'fuel_cat_Hybrid': int(fuel_type == 'Hybrid'),
}

# Brand category
brand_dict
brand_bucket_map = {}
for bucket, brands in brand_dict.items():
    for brand in brands:
        brand_bucket_map[brand] = bucket

input_brand = st.selectbox("Select Brand", sorted(brand_bucket_map.keys()))
brand_bucket = brand_bucket_map.get(input_brand, 'Economy')  # Default to 'Economy' if not found

# Initialize all buckets as 0
brand_dummies = {
    'brand_cat_Economy': 0,
    'brand_cat_Premium': 0,
    'brand_cat_Luxury': 0,
    'brand_cat_Ultra Luxury': 0
}

# Set the corresponding bucket to 1
brand_dummies[f'brand_cat_{brand_bucket}'] = 1

# Year of manufacture
year = st.slider("Year of Manufacture", min_value=1990, max_value=datetime.now().year, value=2018)
age_squared = (datetime.now().year - year) ** 2

# --- BUILD INPUT ---
input_data = {
    'Kilometres': kilometres,
    'seats_int': seats,
    'LitresPer100km': fuel_eff,
    'used_0_new_1': used_0_new_1,
    'transmission_auto': transmission_auto,
    'age_squared': age_squared
}

# Merge all encoded inputs
input_data.update(color_dict)
input_data.update(fuel_dict)
input_data.update(brand_dummies)

# Ensure all columns are present
required_columns = [
    'Kilometres', 'doors_int', 'seats_int', 'LitresPer100km',
    'used_0_new_1', 'transmission_auto', 'color_black', 'color_white',
    'color_gray', 'color_silver', 'color_red', 'fuel_cat_Gasoline',
    'fuel_cat_Diesel', 'fuel_cat_Electric', 'fuel_cat_Hybrid',
    'brand_cat_Economy', 'brand_cat_Luxury', 'brand_cat_Premium',
    'brand_cat_Ultra Luxury', 'age_squared'
]

# Fill missing keys (if any) with 0
for col in required_columns:
    input_data[col] = input_data.get(col, 0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])


# --- PREDICTION ---
if st.button("Predict Price"):
    # Ensure input_df matches model's expected feature order
    expected_features = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_features, fill_value=0)


    # 1. Predict in log space
    log_pred = model.predict(input_df)[0]

    # 2. Set residual std dev (already in log space)
    residual_std = 0.213  

    # 3. Z-score for 95% CI
    z_score = norm.ppf(0.975)

    # 4. Compute lower and upper bounds in log space
    lower_log = log_pred - z_score * residual_std
    upper_log = log_pred + z_score * residual_std

    # 5. Convert to price space
    lower_price = np.exp(lower_log)
    upper_price = np.exp(upper_log)

    # 6. Round to nearest 100 and convert to int
    def round_to_100(x):
        return int(round(x / 100.0) * 100)

    lower_price_rounded = round_to_100(lower_price)
    upper_price_rounded = round_to_100(upper_price)

    # Display
    st.markdown(
    f"🔍 <span style='font-weight:bold; font-size:24px;'>Estimated Price Range (95% CI):</span><br>"
    f"<span style='font-weight:bold; font-size:28px; color:#2E8B57;'>${lower_price_rounded:,} - ${upper_price_rounded:,}</span>",
    unsafe_allow_html=True
)




