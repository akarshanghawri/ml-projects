## Used Car Price Predictor 🚗

This machine learning project predicts the selling price of used cars based on features like brand, year, fuel type, kilometers driven, transmission type, and more. Built with **CatBoost Regressor**, deployed using **Streamlit**, and trained on real-world vehicle data.

## 📊 Features

- Extracts car brand from the full name
- Handles categorical and numerical features
- Uses CatBoost for better performance with categorical data
- Removes the outliers to have a better score 
- Web app built using Streamlit
- Model saved and loaded via Joblib

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- CatBoost
- Joblib
- Streamlit

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-projects.git
   cd ml-projects/used_car_price_predictor

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run app.py

## 🗂️ Folder Structure
<pre>
used_car_price_predictor/
├── car_data.csv
├── app.py
├── main.py
├── model.pkl
├── requirements.txt
└── README.md
</pre>

## 🧪 Model Evaluation
- MAE: ~₹0.96 Lakh
- RMSE: ~₹1.29 Lakh
- R² Score: 0.73

