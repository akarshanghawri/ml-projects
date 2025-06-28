from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get data from form and convert to float
            features = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"]),
            ]

            # Reshape for prediction
            input_data = np.array(features).reshape(1, -1)
            pred = model.predict(input_data)[0]
            prediction = "🛑 Diabetic" if pred == 1 else "✅ Not Diabetic"
        except:
            prediction = "⚠️ Please enter valid numeric values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides a PORT env variable
    app.run(host='0.0.0.0', port=port)
