import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from brand_dict import brand_dict
from car_groups import make_model_dict

# -----------------------------
# Load trained model and feature columns
# -----------------------------
with open("rf.pkl", "rb") as f:
    model = pickle.load(f)

feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Load dataset to get dropdown options
df = pd.read_csv("car_data_cleaned.csv")

# Keep only brands with >=100 rows
brand_counts = df['Brand'].value_counts()
valid_brands = brand_counts[brand_counts >= 100].index
df = df[df['Brand'].isin(valid_brands)]

# Create brand → model mapping
brand_model_mapping = df.groupby('Brand')['Model'].unique().apply(list).to_dict()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Ver 2 Australian Used Car Price Predictor")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Car Brand", sorted(valid_brands))
    model_choice = st.selectbox("Car Model", sorted(brand_model_mapping[brand]))
    year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2020)
    kms = st.number_input("Kilometers Covered", min_value=1000, max_value=200000, value=50000, step=5000)
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"], index=0)
    body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Wagon", "Hatchback"])

with col2:
    fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_bucket'].unique()))
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1, max_value=20, value=8, step=1)
    cylinders = st.selectbox("Engine Cylinders", [2, 4, 6, 8], index=1)
    litres = st.number_input("Engine Litres", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    color = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Red", "Others"])
    seats = st.selectbox("Seats (Optional)", [5, 6, 7], index=0)
    used_new = st.selectbox("Car Condition", ["Used", "New"], index=0)

# -----------------------------
# Prepare input for prediction
# -----------------------------
input_dict = {
    "Kilometres": kms,
    "doors_int": 5,
    "seats_int": seats,
    "LitresPer100km": fuel_consumption,
    "transmission_auto": 1 if transmission == "Automatic" else 0,
    "color_black": int(color == "Black"),
    "color_white": int(color == "White"),
    "color_gray": int(color == "Gray"),
    "color_silver": int(color == "Silver"),
    "color_red": int(color == "Red"),
    "fuel_cat_Gasoline": int(fuel_type == "Gasoline"),
    "fuel_cat_Hybrid": int(fuel_type == "Hybrid"),
    "fuel_cat_Diesel": int(fuel_type == "Diesel"),
    "fuel_cat_Electric": int(fuel_type == "Electric"),
    "brand_cat_Economy": int(brand in brand_dict.get("Economy", [])),
    "brand_cat_Luxury": int(brand in brand_dict.get("Luxury", [])),
    "brand_cat_Premium": int(brand in brand_dict.get("Premium", [])),
    "brand_cat_Ultra Luxury": int(brand in brand_dict.get("Ultra Luxury", [])),
    "cylinders": cylinders,
    "engine_l": litres,
    "age": 2025 - year,
    "Body_type_Other": int(body_type not in ["Sedan", "SUV", "Wagon", "Hatchback"]),
    "Body_type_SUV": int(body_type == "SUV"),
    "Body_type_Sedan": int(body_type == "Sedan"),
    "Body_type_Wagon": int(body_type == "Wagon"),
    "used_0_new_1": int(used_new == "New"),
    "Make_Model_cat_economy": int(model_choice in make_model_dict.get("Economy", [])),
    "Make_Model_cat_premium": int(model_choice in make_model_dict.get("Premium", []))
}

input_data = pd.DataFrame([input_dict])

# Ensure all features the model expects exist
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder to match model
input_data = input_data[feature_columns]

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Price"):
    try:
        price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${round(price/500,0)*500:,.0f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
features = np.array(model.feature_names_in_)
sorted_idx = np.argsort(importances)[-5:]

with st.expander("ℹ️ Top 5 Features determining car price"):
    top_features = features[sorted_idx][::-1]
    top_importances = importances[sorted_idx][::-1]

    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), top_importances, align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

# -----------------------------
# Model Info and Metrics
# -----------------------------
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    - **Model**: Random Forest Regressor  
    - **R² Score**: 0.79 on test set  
    - **Data**: Scraped from Australian car listings  
    - **Features**: Brand, year, fuel type, color, transmission, etc.  
    """)

# Market metrics
yoy_price_growth = 4.6  # %
avg_days_to_sell = 49.7  # days

st.metric("YoY Used Car Price Growth (May 2025)", f"{yoy_price_growth:.1f}%")
st.metric("Average Days to Sell a Used Vehicle (April 2025)", f"{avg_days_to_sell:.1f} days")

st.write("\nSource: Datium Insights – Moody’s Analytics Used Vehicle Price Index (May 2025), [Economy.com]. AADA April 2025 Used Vehicle Sales Report")
