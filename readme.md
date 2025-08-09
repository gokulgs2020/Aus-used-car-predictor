🚗 ML-Based Used Car Price Prediction Dashboard – Australian Market
📌 Overview
A Streamlit-based web application that predicts the fair market price of used cars in Australia based on brand, model, year, mileage, fuel type, and transmission.
Designed for:

Car buyers → Negotiate better deals.

Dealerships → Set competitive yet profitable prices.

Live Demo: Streamlit App
Dataset: Kaggle – Australian Vehicle Prices

🎯 Business Problem
The Australian used car market is valued at $60B+ annually, influenced by:

Regional demand

Fuel preferences

Seasonal trends

Without accurate pricing tools:

Overpricing → Longer inventory cycles

Underpricing → Profit loss

Solution: A data-driven model providing brand & model-specific price estimates for AU market players.

🛠 Technical Approach
1. Data Collection & Cleaning
Used open-source datasets from Carsales, Gumtree, and Kaggle.

Cleaned & merged multiple sources for richer feature coverage.

2. Feature Engineering
Numerical: mileage, year

Categorical: brand, model, fuel type, transmission

Derived: car age, price per kilometre

3. Modelling
Baseline: Linear Regression

Improved: Lasso Regression + Random Forest Regressor (GridSearchCV optimized)

Performance: R² = 0.82, MAE = $1,850

4. Deployment
Interactive Streamlit UI with dropdowns & sliders

Real-time prediction output

📊 Key Results
Metric	Value
R² Score	0.82
MAE	$1,850
Dataset Size	45,000+ rows

Sample Prediction:
Toyota Corolla, 2018, 45,000 km, Petrol, Automatic, Sydney → $18,500

💡 Business Impact
Dealers: Price competitively, improve turnover.

Buyers: Negotiate with confidence.

Analysts: Track brand & regional trends.

🖼 Sample Screenshot
<img width="522" height="715" alt="image" src="https://github.com/user-attachments/assets/d80de2ab-d77d-44f5-ad71-2fd5d9474767" />
🚀 Future Enhancements
Compare multiple models in-app ✅

Confidence intervals in predictions ✅

Real-time listings scraping

Seasonal trend adjustments

Expansion to New Zealand market

Batch file upload for bulk predictions

👨‍💻 Author
Gokul GS
LinkedIn | GitHub
