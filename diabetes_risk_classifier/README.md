# 🩺 Diabetes Risk Classifier

This machine learning project predicts whether a person is at risk of diabetes based on medical features such as glucose level, BMI, insulin, age, and more. Built using CatBoostClassifier with a user-friendly web interface for real-time predictions.

## 📂 Project Structure
diabetes_risk_classifier/
├── data/
│   └── diabetes.csv
├── templates/
│   └── index.html
├── model.pkl
├── app.py
├── main.py
├── requirements.txt
└── README.md

## 🔍 Features
- Handles missing or zero medical values smartly
- Predicts diabetes risk using CatBoost
- Includes clean visualizations in EDA
- Deployable with Streamlit or Flask

## 🛠️ How to Run

1. **Install requirements**  
   ```bash
   pip install -r app/requirements.txt

Run the web app:
    run app.py

## 📊 Model Performance - 
- Accuracy: ~0.75 (can be improved with tuning)
- Classifier Used: CatBoostClassifier
- Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix\

## 🧪 Dataset
- Source: Pima Indians Diabetes Dataset
- Features include glucose, insulin, pregnancies, etc.



