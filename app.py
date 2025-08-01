
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
kilometres = st.number_input("Kilometres Driven", min_value=0, value=50000)
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7])
fuel_eff = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, value=8.0)

car_condition = st.radio("Condition", ["Used", "New"])
used_0_new_1 = 1 if car_condition == "New" else 0

transmission = st.radio("Transmission", ["Automatic", "Manual"])
transmission_auto = 1 if transmission == "Automatic" else 0

color = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Red", "Others"])
color_dict = {
    'color_black': int(color == 'Black'),
    'color_white': int(color == 'White'),
    'color_gray': int(color == 'Gray'),
    'color_silver': int(color == 'Silver'),
    'color_red': int(color == 'Red'),
}

fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
fuel_dict = {
    'fuel_cat_Gasoline': int(fuel_type == 'Gasoline'),
    'fuel_cat_Diesel': int(fuel_type == 'Diesel'),
    'fuel_cat_Electric': int(fuel_type == 'Electric'),
    'fuel_cat_Hybrid': int(fuel_type == 'Hybrid'),
}

brand_bucket_map = {}
for bucket, brands in brand_dict.items():
    for brand in brands:
        brand_bucket_map[brand] = bucket

input_brand = st.selectbox("Select Brand", sorted(brand_bucket_map.keys()))
brand_bucket = brand_bucket_map.get(input_brand, 'Economy')

brand_dummies = {
    'brand_cat_Economy': 0,
    'brand_cat_Premium': 0,
    'brand_cat_Luxury': 0,
    'brand_cat_Ultra Luxury': 0
}
brand_dummies[f'brand_cat_{brand_bucket}'] = 1

year = st.slider("Year of Manufacture", min_value=1990, max_value=datetime.now().year, value=2018)
age_squared = (datetime.now().year - year) ** 2

input_data = {
    'Kilometres': kilometres,
    'seats_int': seats,
    'LitresPer100km': fuel_eff,
    'used_0_new_1': used_0_new_1,
    'transmission_auto': transmission_auto,
    'age_squared': age_squared
}
input_data.update(color_dict)
input_data.update(fuel_dict)
input_data.update(brand_dummies)

required_columns = [
    'Kilometres', 'doors_int', 'seats_int', 'LitresPer100km',
    'used_0_new_1', 'transmission_auto', 'color_black', 'color_white',
    'color_gray', 'color_silver', 'color_red', 'fuel_cat_Gasoline',
    'fuel_cat_Diesel', 'fuel_cat_Electric', 'fuel_cat_Hybrid',
    'brand_cat_Economy', 'brand_cat_Luxury', 'brand_cat_Premium',
    'brand_cat_Ultra Luxury', 'age_squared'
]

for col in required_columns:
    input_data[col] = input_data.get(col, 0)

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    expected_features = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    log_pred = model.predict(input_df)[0]
    residual_std = 0.213
    z_score = norm.ppf(0.95)

    lower_price = np.exp(log_pred) - z_score * np.exp(residual_std)
    upper_price = np.exp(log_pred) + z_score * np.exp(residual_std)

    def round_to_1000(x):
        return int(round(x / 1000.0) * 1000)

    lower_price_rounded = round_to_1000(lower_price*0.9)
    upper_price_rounded = round_to_1000(upper_price*1.1)

    st.markdown(
        f"""
        <div style='font-family: "sans-serif";'>
            <span style='font-weight: bold; font-size: 20px;'>🔍 Estimated Price Range (90% CI):</span><br>
            <span style='color: #2E8B57; font-weight: 600; font-size: 28px;'>
                ${lower_price_rounded:,} – ${upper_price_rounded:,}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
