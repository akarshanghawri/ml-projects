import streamlit as st
import pandas as pd
import joblib
import os
# from sklearn.preprocessing import LabelEncoder

# Get the absolute path of the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

st.title("🚗 Used Car Price Predictor")

# Manual encoding maps (same order as in training)
fuel_map = {'Petrol': 3, 'Diesel': 1, 'CNG': 0, 'LPG': 2, 'Electric': 4}
seller_map = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
trans_map = {'Manual': 1, 'Automatic': 0}
owner_map = {
    'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2,
    'Fourth & Above Owner': 3, 'Test Drive Car': 4
}
brand_map = {
    'Maruti': 8, 'Hyundai': 6, 'Honda': 5, 'Toyota': 13, 'Mahindra': 7,
    'Ford': 3, 'Chevrolet': 1, 'Tata': 12, 'BMW': 0, 'Audi': 14,
    'Mercedes': 9, 'Volkswagen': 15, 'Renault': 11, 'Skoda': 10,
    'Datsun': 2, 'Nissan': 16, 'Kia': 4, 'MG': 17, 'Other': 18
}

# Input form
brand = st.selectbox("Car Brand", [
    "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Ford", "Chevrolet",
    "Tata", "BMW", "Audi", "Mercedes", "Volkswagen", "Renault", "Skoda", 
    "Datsun", "Nissan", "Kia", "MG", "Other"
])
year = st.number_input("Year", 1990, 2025, step=1)
km_driven = st.number_input("Kilometers Driven", 0)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

#prediction
if st.button("Predict Price"):
    # Encode inputs similarly to training
    input_df = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel_map[fuel]],
    'seller_type': [seller_map[seller_type]],
    'transmission': [trans_map[transmission]],
    'owner': [owner_map[owner]],
    'brand': [brand_map[brand]]
    })

    prediction = model.predict(input_df)
    st.subheader(f"Estimated Selling Price: ₹{int(prediction[0]):,}")

# To run the app use terminal command: streamlit run  ml-projects/used_car_price_predictor/app.py 
