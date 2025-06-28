import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('ml-projects/student_social_media_addiction/model.pkl')

st.title("🎓 Student Social Media Addiction Predictor")

# User inputs
age = st.number_input("Age", min_value=10, max_value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
country = st.text_input("Country")
usage_hours = st.slider("Average Daily Usage Hours", 0.0, 12.0, 1.0)
platform = st.selectbox("Most Used Platform", ["Instagram", "Facebook", "TikTok", "YouTube", "Twitter"])
affects_academics = st.selectbox("Affects Academic Performance", ["Yes", "No"])
sleep = st.slider("Sleep Hours per Night", 0.0, 12.0, 6.0)
mental_score = st.slider("Mental Health Score (1-10)", 1, 10, 5)
relationship = st.selectbox("Relationship Status", ["Single", "In Relationship", "Complicated"])
conflicts = st.slider("Conflicts Over Social Media", 0, 10, 0)

# Encode inputs (must match training)
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Academic_Level': [academic_level],
    'Country': [country],
    'Avg_Daily_Usage_Hours': [usage_hours],
    'Most_Used_Platform': [platform],
    'Affects_Academic_Performance': [affects_academics],
    'Sleep_Hours_Per_Night': [sleep],
    'Mental_Health_Score': [mental_score],
    'Relationship_Status': [relationship],
    'Conflicts_Over_Social_Media': [conflicts]
})

# Label encoding (simplified, must match your training)
from sklearn.preprocessing import LabelEncoder
for col in input_df.select_dtypes(include='object'):
    le = LabelEncoder()
    input_df[col] = le.fit_transform(input_df[col])

# Predict
if st.button("Predict Addiction"):
    prediction = model.predict(input_df)
    result = "🛑 Addicted" if prediction[0] == 1 else "✅ Not Addicted"
    st.subheader(f"Prediction: {result}")


# to run the app use terminal command : streamlit run ml-projects/student_social_media_addiction/app.py