import streamlit as st
import joblib
import pandas as pd

st.title("😊 Happiness Score Predictor")

# Load model + scaler
model = joblib.load("svr_happiness_model.pkl")
scaler = joblib.load("scaler.pkl")

st.write("Fill the details below to get your predicted happiness score.")

# Input fields
device_hours = st.number_input("Device hours per day", 0, 20, 5)
physical_activity = st.number_input("Physical activity days per week", 0, 7, 3)
sleep_quality = st.number_input("Sleep quality (1–10)", 1, 10, 7)
stress = st.number_input("Stress level (1–10)", 1, 10, 5)
anxiety = st.number_input("Anxiety score (1–10)", 1, 10, 4)
depression = st.number_input("Depression score (1–10)", 1, 10, 3)
focus = st.number_input("Focus score (1–10)", 1, 10, 6)
productivity = st.number_input("Productivity score (1–100)", 1, 100, 70)
digital_dependence = st.number_input("Digital dependence score (1–100)", 1, 100, 60)
notifications = st.number_input("Notifications per day", 0, 500, 100)

# Feature order
features = [[
    device_hours,
    physical_activity,
    sleep_quality,
    stress,
    anxiety,
    depression,
    focus,
    productivity,
    digital_dependence,
    notifications
]]

if st.button("Predict Happiness Score"):
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    st.success(f"Your Predicted Happiness Score: {prediction:.2f}")
