# 🩺 Diabetes Risk Classifier

This project uses machine learning to predict the risk of diabetes based on medical attributes like glucose levels, BMI, insulin, age, and more. It includes a simple web app built with Flask that allows users to input health metrics and receive real-time predictions.

---

## 🚀 Deployment

This Flask-based diabetes risk classifier has been deployed on Render.
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-green)](https://diabetes-risk-classifier.onrender.com)

## 📁 Project Structure

<pre>
diabetes_risk_classifier/
├── data/
│   └── diabetes.csv           # Dataset used for training
│
├── templates/
│   └── index.html             # HTML template for Flask app
│
├── model.pkl                  # Trained machine learning model
├── app.py                     # Flask web application
├── main.py                    # Model training and evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and instructions
</pre>

---

## 🔍 Features
- Handles missing or zero medical values smartly
- Predicts diabetes risk using CatBoost
- Includes clean visualizations in EDA
- Deployable with Streamlit or Flask

## 🛠️ How to Run

1. **Install requirements**  
   ```bash
   pip install -r app/requirements.txt

2. Run the web app:
    ```bash
   run app.py

## 📊 Model Performance - 
- Accuracy: ~0.75 (can be improved with tuning)
- Classifier Used: CatBoostClassifier
- Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix\

## 🧪 Dataset
- Source: Pima Indians Diabetes Dataset
- Features include glucose, insulin, pregnancies, etc.

🔍 Note: This model is based on data from Pima Indian women only. Results may not generalize to other populations or genders.


