import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from brand_dict import brand_dict
from car_groups import make_model_dict
import base64

# ----------------------------
# --- Background image with opacity ---
# ----------------------------
def add_bg_with_opacity(image_file, opacity=0.3):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            position: relative;
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255, 255, 255, {1-opacity});
            z-index: 0;
        }}
        .stApp .main {{
            position: relative;
            z-index: 1;  /* ensures widgets are above overlay */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
add_bg_with_opacity("used_car_app_background.jpg", opacity=0.3)

# ----------------------------
# --- Load model and dataset ---
# ----------------------------
with open("rf_lasso.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("car_data_cleaned.csv")
brand_counts = df['Brand'].value_counts()
valid_brands = brand_counts[brand_counts >= 50].index
df = df[df['Brand'].isin(valid_brands)]
brand_model_mapping = df.groupby('Brand')['Model'].unique().apply(list).to_dict()

# Default selections
default_brand = "Hyundai"
default_model = "Tucson"

st.title("🚗 Australian Used Car Price Prediction App")

# ----------------------------
# --- User Inputs ---
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Car Brand", sorted(valid_brands), index=sorted(valid_brands).index(default_brand))
    model_choice = st.selectbox("Car Model", sorted(brand_model_mapping[brand]), index=sorted(brand_model_mapping[brand]).index(default_model))
    year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2023)
    kms = st.number_input("Kilometers Covered", min_value=1000, max_value=200000, value=25000, step=5000)
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"], index=0)
    body_type = st.selectbox("Car Body Type", ["SUV", "Sedan","Hatchback","Wagon"], index=0)

with col2:
    fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_bucket'].unique()))
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1, max_value=20, value=8, step=1)
    cylinders = st.selectbox("Engine Cylinders", [2,4,6,8], index=1)
    litres = st.number_input("Engine Litres", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    color = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Red", "Others"])
    seats = st.selectbox("Seats (Optional)", [5,6,7], index=0)

# ----------------------------
# --- Prepare features ---
# ----------------------------
input_data = pd.DataFrame([{
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
    "brand_cat_Economy": int(brand in brand_dict["Economy"]),
    "brand_cat_Luxury": int(brand in brand_dict["Luxury"]),
    "cylinders": cylinders,
    "engine_l": litres,
    "age_squared": (2025 - year)**2,
    "Body_type_Other": int(body_type not in ["Sedan", "SUV", "Wagon","Hatchback"]),
    "Body_type_SUV": int(body_type == "SUV"),
    "Body_type_Sedan": int(body_type == "Sedan"),
    "Body_type_Wagon": int(body_type == "Wagon"),
    "Make_Model_cat_economy": int(model_choice in make_model_dict["Economy"]),
    "Make_Model_cat_premium": int(model_choice in make_model_dict["Premium"])
}])

# ----------------------------
# --- Prediction ---
# ----------------------------
if st.button("Predict Price"):
    try:
        price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${round(np.exp(price)/500,0)*500:,.0f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ----------------------------
# --- Feature Importance ---
# ----------------------------
importances = model.feature_importances_
features = np.array(model.feature_names_in_)
sorted_idx = np.argsort(importances)[-5:]

with st.expander("ℹ️ Top 5 Features determining the price of used cars"):
    st.subheader("📊 Top 5 Features")
    top_features = features[sorted_idx][::-1]
    top_importances = importances[sorted_idx][::-1]
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), top_importances, align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

# ----------------------------
# --- Model Info ---
# ----------------------------
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    - **Model**: Random Forest Regressor with Lasso feature selection  
    - **R² Score**: 0.88 on test set  
    - **Data**: Scraped from Australian car listing sites (pre-cleaned)  
    - **Features**: Brand, year, fuel type, color, transmission, etc.  
    """)

# ----------------------------
# --- Market Data Metrics ---
# ----------------------------
st.write("\n\n\nMarket data shows slight increase in YoY prices of used cars in Australia (avg time for sale: 7 weeks)")
yoy_price_growth = 4.6
avg_days_to_sell = 49.7
st.metric("YoY Used Car Price Growth (May 2025)", f"{yoy_price_growth:.1f}%")
st.metric("Average Days to Sell a Used Vehicle (April 2025)", f"{avg_days_to_sell:.1f} days")
st.write("\nSource: Datium Insights – Moody’s Analytics Used Vehicle Price Index (May 2025), [Economy.com] \n AADA April 2025 Used Vehicle Sales Report")
