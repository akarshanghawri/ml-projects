import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gets the absolute path of the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'car_data.csv')
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Load data
df = pd.read_csv(data_path)

# Extracting the brand name 
brand_list = [
    "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Ford", "Chevrolet",
    "Tata", "BMW", "Audi", "Mercedes", "Volkswagen", "Renault", "Skoda", 
    "Datsun", "Nissan", "Kia", "MG"
]

def extract_brand(name):
    for brand in brand_list:
        if brand.lower() in name.lower():
            return brand
    return "Other"

df['brand'] = df['name'].apply(extract_brand)
df.drop('name', axis=1, inplace=True)   #drop full name 

# Removing outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from 'selling_price' and 'km_driven'
df = remove_outliers_iqr(df, 'selling_price')
df = remove_outliers_iqr(df, 'km_driven')


#split data
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Encode categorical features
le = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = le.fit_transform(X[col])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_train,y_train)

# predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#  Metric explanations -
#  MAE: Average absolute difference between predicted and actual values
#  MSE: Average squared difference (penalizes large errors more)
#  R² Score: Proportion of variance explained by the model (1 = perfect)
print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): ₹{mae:,.0f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.0f}")
print(f"R² Score: {r2:.2f}")

# Save the model
if not os.path.exists(model_path):
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
else:
    print("⚠️ Model already exists. Skipping save.")
