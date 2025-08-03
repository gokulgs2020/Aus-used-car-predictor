🚗 Car Price Prediction Dashboard
A Streamlit-based web application that predicts the price of a car based on user inputs such as brand, fuel type, transmission, colour, and other features.
The prediction model uses a Lasso Regression and Random Forest Regressor trained on cleaned car listings data.

🔍 Features
Predicts car prices in Australia using model trained on real-world dataset

Makes use of Dynamic input form: dropdowns and sliders for intuitive use

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

Models have been saved using pickle and loaded in the app.

📊 Sample Screenshot


<img width="522" height="715" alt="image" src="https://github.com/user-attachments/assets/d80de2ab-d77d-44f5-ad71-2fd5d9474767" />


🧠 Future Enhancements
Add support for multiple models to compare results


Add file upload for batch predictions

👨‍💻 Author
Gokul GS
LinkedIn | GitHub

