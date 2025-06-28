# 🎓 Student Social Media Addiction Predictor

This is a simple machine learning project that predicts whether a student is addicted to social media based on their daily usage, academic performance, mental health, and other factors. I built this as part of my learning journey to understand real-world applications of ML, and also added a user-friendly interface using Streamlit.

---

## 📁 What’s Inside

The dataset contains information like:
- Age, gender, academic level
- Average daily time spent on social media
- Most-used platform
- Impact on academics
- Sleep patterns and mental health score
- Relationship status and conflicts due to social media

The final label (target) is an **addiction score**, which I’ve used to classify students as addicted or not.

---

## 🧠 Model Info

I used a basic machine learning pipeline with:
- Label Encoding for categorical features
- Random Forest Classifier (or any other you used)
- Accuracy and other metrics for evaluation

---

## 🖥 App Features

The Streamlit app allows you to:
- Upload a CSV file for batch predictions
- Enter details for one student to predict in real-time
- Visualize the relationship between screen time and mental health
- Toggle between light and dark mode
- Download the prediction results as a CSV file

---

## ▶️ How to Run the App

Make sure you have all the required packages installed:

```bash
pip install -r requirements.txt

#Then start the app:
streamlit run ml-projects/student_social_media_addiction/app.py

Folder Structure :
student-social-addiction/
├── app.py                 # Streamlit web app
├── main.py                # Model training script
├── model.pkl              # Saved trained model
├── student_data.csv       # Sample dataset
├── requirements.txt       # Project dependencies
└── README.md              # This file

