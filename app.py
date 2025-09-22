import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from brand_dict import brand_dict
from car_groups import make_model_dict

# ------------------- Google Analytics -------------------
GA_MEASUREMENT_ID = "G-D3FBZQ2V5H"  

st.markdown(f"""
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}', {{
      'send_page_view': true
  }});
</script>
""", unsafe_allow_html=True)

# ------------------- Background -------------------
def set_background(local_file, opacity=0.3):
    """Set a local image as Streamlit background with transparency."""
    with open(local_file, "rb") as f:
        img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,{1-opacity}), rgba(255,255,255,{1-opacity})),
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("used_car_app_background.jpg", opacity=0.3)

# ------------------- Load Model and Data -------------------
with open("rf_lasso.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("car_data_cleaned.csv")

# Filter brands with sufficient data
brand_counts = df['Brand'].value_counts()
valid_brands = brand_counts[brand_counts >= 100].index
df = df[df['Brand'].isin(valid_brands)]

# Temporarily remove luxury brands due to pricing error compared to actuals - to be fixed
exclude_brands = ["BMW", "Mercedes-Benz", "Audi", "Lexus","Land"]  
valid_brands = [b for b in valid_brands if b not in exclude_brands]
df = df[df['Brand'].isin(valid_brands)]

# Brand -> Model mapping
brand_model_mapping = df.groupby('Brand')['Model'].unique().apply(list).to_dict()

default_brand = "Hyundai"
default_model = "Tucson"

st.title("🚗 Australian Used Car Price Prediction App")

# ------------------- User Inputs -------------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Car Brand",
        sorted(valid_brands),
        index=sorted(valid_brands).index(default_brand) if default_brand in valid_brands else 0
    )

    # Safely select models for the chosen brand
    models = sorted(brand_model_mapping[brand])
    if default_model in models:
        model_index = models.index(default_model)
    else:
        model_index = 0  # fallback to first available model

    model_choice = st.selectbox("Car Model", models, index=model_index)

    st.markdown(f"""
    <script>
      gtag('event', 'select_brand', {{
        'event_category': 'input',
        'event_label': '{brand}'
      }});
      gtag('event', 'select_model', {{
        'event_category': 'input',
        'event_label': '{model_choice}'
      }});
    </script>
    """, unsafe_allow_html=True)

    year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2023)
    kms = st.number_input("Kilometers Covered", min_value=1000, max_value=200000, value=25000, step=5000)
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"], index=0)
    body_type = st.selectbox("Car Body Type", ["SUV", "Sedan", "Hatchback", "Wagon"], index=0)

with col2:
    fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_bucket'].unique()))
    st.markdown(f"""
    <script>
      gtag('event', 'select_fuel_type', {{
        'event_category': 'input',
        'event_label': '{fuel_type}'
      }});
    </script>
    """, unsafe_allow_html=True)

    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1, max_value=20, value=8, step=1)
    cylinders = st.selectbox("Engine Cylinders", [2, 4, 6, 8], index=1)
    litres = st.number_input("Engine Litres", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    color = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Red", "Others"])
    seats = st.selectbox("Seats (Optional)", [5, 6, 7], index=0)

# ------------------- Prepare Input -------------------
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
    "Body_type_Other": int(body_type not in ["Sedan", "SUV", "Wagon", "Hatchback"]),
    "Body_type_SUV": int(body_type == "SUV"),
    "Body_type_Sedan": int(body_type == "Sedan"),
    "Body_type_Wagon": int(body_type == "Wagon"),
    "Make_Model_cat_economy": int(model_choice in make_model_dict["Economy"]),
    "Make_Model_cat_premium": int(model_choice in make_model_dict["Premium"])
}])

# ------------------- Prediction -------------------
if st.button("Predict Price"):
    try:
        price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${round(np.exp(price)/500,0)*500:,.0f}")
        st.markdown(f"""
        <script>
          gtag('event', 'predict_button_click', {{
            'event_category': 'engagement',
            'event_label': 'Predict Price Button'
          }});
        </script>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ------------------- Feature Importance -------------------
importances = model.feature_importances_
features = np.array(model.feature_names_in_)
sorted_idx = np.argsort(importances)[-5:]

with st.expander("ℹ️ Top 5 Features Determining Price"):
    top_features = features[sorted_idx][::-1]
    top_importances = importances[sorted_idx][::-1]

    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), top_importances, align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

# ------------------- Model Info -------------------
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    - **Model**: Random Forest Regressor with Lasso feature selection  
    - **R² Score**: 0.88 on test set  
    - **Data**: Scraped from Australian car listing sites (pre-cleaned)  
    - **Features**: Brand, year, fuel type, color, transmission, etc.  
    """)

# ------------------- Market Insights -------------------
st.write("\n\nMarket data shows slight increase in the YoY prices of used cars in Australia, with average sale time about 7 weeks.")

yoy_price_growth = 4.6  # %
avg_days_to_sell = 49.7  # days

st.metric(label="YoY Used Car Price Growth (May 2025)", value=f"{yoy_price_growth:.1f}%")
st.metric(label="Average Days to Sell a Used Vehicle (April 2025)", value=f"{avg_days_to_sell:.1f} days")

st.write(
    "\nSource: Datium Insights – Moody’s Analytics Used Vehicle Price Index (May 2025), "
    "[Economy.com] \nAADA April 2025 Used Vehicle Sales Report"
)
