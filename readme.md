🚗 Car Price Prediction Dashboard
A Streamlit-based web application that predicts the price of a car based on user inputs such as brand, fuel type, transmission, colour, and other features.
The prediction model uses a Lasso Regression and Random Forest Regressor trained on cleaned car listings data.

🔍 Features
Predicts car prices in Australia using model trained on real-world dataset

Dynamic input form: dropdowns and sliders for intuitive use

Categorizes:

Brands into Economy, Premium, Luxury, and Ultra Luxury

Fuel type into Gasoline, Diesel, Electric, and Hybrid

Preprocessing logic includes:

One-hot encoding of categorical variables

Feature engineering (e.g. age squared, fuel efficiency conversion)

Scaling for model compatibility

🧠 ML Models Used
RandomForestRegressor (grid search optimized)

Lasso Regression

MinMaxScaler for feature scaling

Models are saved using pickle and loaded in the app.

🛠 How to Run Locally
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/car-price-predictor.git
cd car-price-predictor
2. Install dependencies
Make sure you have Python 3.8+ installed.

bash
Copy
Edit
pip install -r requirements.txt
3. Run the app
bash
Copy
Edit
streamlit run app.py
🧾 Input Parameters
Feature	Description
Brand	Make of the car (e.g. Honda, BMW)
Fuel Type	Fuel type: Gasoline, Diesel, Electric, Hybrid
Transmission	Manual or Automatic
Number of Doors	Extracted from text like "4 Doors"
Kilometres Driven	Car mileage in KM
Colour	Primary colour from string like "White / Black"
Age of Car	Derived from year of manufacture
Fuel Efficiency	Extracted from "8.7 L/100 km" and converted to KM/L

📁 Project Structure
bash
Copy
Edit
├── app.py                # Streamlit app
├── model_lasso.pkl       # Pickled Lasso model
├── model_rf.pkl          # Pickled Random Forest model
├── scaler.pkl            # Pickled MinMaxScaler
├── brand_dict.py         # Brand → Category mapping
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
🚀 Deployment
You can deploy this app for free using Streamlit Cloud:

Push your project to a public GitHub repo

Go to https://streamlit.io/cloud

Click "New app", connect your repo, and deploy!

📊 Sample Screenshot

Add a screenshot of your working app here for better appeal

🧠 Future Enhancements
Add support for multiple models to compare results

Use SHAP or LIME for feature importance explanation

Add file upload for batch predictions

👨‍💻 Author
Gokul GS
LinkedIn | GitHub

