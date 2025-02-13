import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset and train model
heart = pd.read_csv("heart.csv")
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
X = heart[features]
Y = heart['output']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier()
knn.fit(X_scaled, Y)

# Streamlit UI
st.set_page_config(page_title="Heart Safe AI", page_icon=":)", layout="centered")
st.title("Heart Safe AI - Heart Attack Prediction")
st.markdown("### Enter the details below to predict the risk of heart attack.")

# Define Input Fields
input_data = {}
labels = {
    "Age (Years)": "age",
    "Sex (0: Female, 1: Male)": "sex",
    "Chest Pain Type (0-3)": "cp",
    "Resting BP (mm Hg)": "trtbps",
    "Cholesterol (mg/dL)": "chol",
    "Fasting Blood Sugar (0-1)": "fbs",
    "Resting ECG (0-2)": "restecg",
    "Max Heart Rate (bpm)": "thalachh",
    "Exercise Angina (0-1)": "exng",
    "Old Peak (ST Depression)": "oldpeak",
    "Slope (0-2)": "slp",
    "Number of Vessels (0-4)": "caa",
    "Thal (0-2)": "thall",
}

for label, key in labels.items():
    input_data[key] = st.number_input(label, min_value=0.0, step=1.0 if key not in ['oldpeak'] else 0.1)

# Prediction
if st.button("Predict"):
    try:
        user_input = np.array([[input_data[key] for key in features]])
        user_input_scaled = scaler.transform(user_input)
        prediction = knn.predict(user_input_scaled)[0]

        if prediction == 1:
            st.error("High Risk of Heart Attack! Take precautions:")
            st.markdown("- Maintain a healthy diet  \n- Regular exercise  \n- Avoid smoking & alcohol  \n- Consult a doctor")
        else:
            st.success(" Low Risk of Heart Attack! Keep maintaining a healthy lifestyle.")

    except ValueError:
        st.error(" Please enter valid numbers in all fields!")
