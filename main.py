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
st.set_page_config(page_title="Heart Safe AI", page_icon="❤️", layout="centered")
st.title("Heart Safe AI - Prediction")

# Form for user input
st.sidebar.header("Enter Patient Details")

def user_input():
    age = st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=40)
    sex = st.sidebar.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
    trtbps = st.sidebar.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    fbs = st.sidebar.radio("Fasting Blood Sugar", [0, 1], format_func=lambda x: "<120 mg/dL" if x == 0 else ">120 mg/dL")
    restecg = st.sidebar.slider("Resting ECG", 0, 2, 1)
    thalachh = st.sidebar.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150)
    exng = st.sidebar.radio("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("Old Peak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slp = st.sidebar.slider("Slope", 0, 2, 1)
    caa = st.sidebar.slider("Number of Vessels (0-4)", 0, 4, 0)
    thall = st.sidebar.slider("Thal (0-2)", 0, 2, 1)

    return np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])

# Predict Button
if st.sidebar.button("Predict"):
    user_data = user_input()
    user_data_scaled = scaler.transform(user_data)
    prediction = knn.predict(user_data_scaled)[0]

    if prediction == 1:
        st.error("High Risk of Heart Attack")
        st.write("### Precautions:")
        st.write("- Maintain a healthy diet")
        st.write("- Regular exercise")
        st.write("- Avoid smoking and alcohol")
        st.write("- Consult a doctor")
    else:
        st.success("Low Risk of Heart Attack")
        st.write("Keep maintaining a healthy lifestyle")

st.sidebar.markdown("Developed by Heart Safe AI")
