import streamlit as st
import pickle
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
from brand_dict import brand_dict
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("rf_lasso.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get dropdown options
df = pd.read_csv("car_data_cleaned.csv")

# Drop brands with < 50 rows
brand_counts = df['Brand'].value_counts()
valid_brands = brand_counts[brand_counts >= 50].index
df = df[df['Brand'].isin(valid_brands)]

# Create brand → model mapping
brand_model_mapping = df.groupby('Brand')['Model'].unique().apply(list).to_dict()

st.title("🚗 Car Price Prediction App")

# Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(valid_brands))
    model_choice = st.selectbox("Model", sorted(brand_model_mapping[brand]))
    year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2015)
    kms = st.number_input("Kilometers Covered", min_value=1000, max_value=200000, value=50000)
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"],index=0])
    

with col2:
    fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_bucket'].unique()))
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=20.0, value=8.0, step=1)
    cylinders = st.selectbox("Engine Cylinders Eg: Engine type - **4** cylinders - 2 Litres", [2,4,6,8], index=1)
    litres = st.number_input("Engine Litres Eg: Engine type - 4 cylinders **2** Litres", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
    color = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Red", "Others"])
    color_dict = {
   	'color_black': int(color == 'Black'),
    	'color_white': int(color == 'White'),
    	'color_gray': int(color == 'Gray'),
    	'color_silver': int(color == 'Silver'),
    	'color_red': int(color == 'Red'),
    }
    seats = st.number_input("Seats (Optional)", [5,6,7], index=0)

# Prepare features for prediction
input_data = pd.DataFrame([{
    "Brand": brand,
    "Model": model_choice,
    "Year of Manufacture": year,
    "Transmission": transmission,
    "Fuel Type": fuel_type,
    "Kilometers covered": kms,
    "Fuel Consumption": fuel_consumption,
    "Cylinders": cylinders,
    "Litres": litres,
    "Color": color,
    "Seats": seats
}])

# Predict button
if st.button("Predict Price"):
    try:
        price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

#----------------------------------------------------------------

    # --- Feature Importance ---
    importances = model.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[::-1][:5]

    st.subheader("📊 Top 5 Features")
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([features[i] for i in sorted_idx][::-1])
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

# --- Model Info ---
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    - **Model**: Random Forest Regressor with Lasso feature selection  
    - **R² Score**: 0.88 on test set  
    - **Data**: Scraped from Australian car listing sites (pre-cleaned)  
    - **Features**: Brand, year, fuel type, color, transmission, etc.  
    """)


#-----------------------------------------------

st.write("\n\nMarket data shows slight increase in the YoY prices of used cars implying growing demand in Australia with the avg time for a sale about 7 weeks\n")

yoy_price_growth = 4.6  # YoY growth (%) for Australia’s Used Vehicle Price Index (May 2025)
avg_days_to_sell = 49.7  # Average days to sell a used vehicle (April 2025)

# Display metrics
st.metric(
    label="YoY Used Car Price Growth (May 2025)",
    value=f"{yoy_price_growth:.1f}%",
)
st.metric(
    label="Average Days to Sell a Used Vehicle (April 2025)",
    value=f"{avg_days_to_sell:.1f} days",
)

# Display data sources
st.write(
    "Source: Datium Insights – Moody’s Analytics Used Vehicle Price Index (May 2025), "
    "[Economy.com] \n AADA April 2025 Used Vehicle Sales Report"
)

