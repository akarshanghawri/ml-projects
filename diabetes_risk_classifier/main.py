import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Gets the absolute path of the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data/diabetes.csv')
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Load data
df = pd.read_csv(data_path)

# Data cleaning and Feature engineering 
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())

# Feature Scaling 
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Outcome', axis=1))

# Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(" Model Evaluation:")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
if not os.path.exists(model_path):
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
else:
    print("⚠️ Model already exists. Skipping save.")