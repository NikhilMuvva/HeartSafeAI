from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load dataset and train model
heart = pd.read_csv("heart.csv")
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
X = heart[features]
Y = heart['output']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier()
knn.fit(X_scaled, Y)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Safe AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #DFFFD6;
            text-align: center;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        form {
            background: white;
            padding: 20px;
            border-radius: 8px;
            display: inline-block;
            text-align: left;
        }
        label {
            display: block;
            margin: 10px 0;
        }
        input {
            width: 100%;
            padding: 5px;
            margin-top: 5px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: red;
        }
        .error {
            color: red;
        }
    </style>
</head>
<body>
    <h1>Heart Safe AI - Prediction</h1>
    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
    <form action="/" method="post">
        <label>Age (Years): <input type="number" name="age" required></label>
        <label>Sex (0: Female, 1: Male): <input type="number" name="sex" min="0" max="1" required></label>
        <label>Chest Pain Type (0-3): <input type="number" name="cp" min="0" max="3" required></label>
        <label>Resting BP (mm Hg): <input type="number" name="trtbps" required></label>
        <label>Cholesterol (mg/dL): <input type="number" name="chol" required></label>
        <label>Fasting Blood Sugar (0-1): <input type="number" name="fbs" min="0" max="1" required></label>
        <label>Resting ECG (0-2): <input type="number" name="restecg" min="0" max="2" required></label>
        <label>Max Heart Rate (bpm): <input type="number" name="thalachh" required></label>
        <label>Exercise Angina (0-1): <input type="number" name="exng" min="0" max="1" required></label>
        <label>Old Peak: <input type="number" step="0.1" name="oldpeak" required></label>
        <label>Slope (0-2): <input type="number" name="slp" min="0" max="2" required></label>
        <label>Number of Vessels (0-4): <input type="number" name="caa" min="0" max="4" required></label>
        <label>Thal (0-2): <input type="number" name="thall" min="0" max="2" required></label>
        <button type="submit">Predict</button>
    </form>

    {% if result %}
        <h2 class="result">{{ result }}</h2>
        <h3>Precautions:</h3>
        <ul>
            {% for precaution in precautions %}
                <li>{{ precaution }}</li>
            {% endfor %}
        </ul>
        <a href="/">Go Back</a>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            data = [float(request.form[key]) for key in features]
            user_input = np.array([data])
            user_input_scaled = scaler.transform(user_input)
            prediction = knn.predict(user_input_scaled)[0]

            if prediction == 1:
                result = "High Risk of Heart Attack"
                precautions = ["Maintain a healthy diet", "Regular exercise", "Avoid smoking and alcohol", "Consult a doctor"]
            else:
                result = "Low Risk of Heart Attack"
                precautions = ["Keep maintaining a healthy lifestyle"]

            return render_template_string(HTML_TEMPLATE, result=result, precautions=precautions)

        except ValueError:
            return render_template_string(HTML_TEMPLATE, error="Please enter valid numbers in all fields!")

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
