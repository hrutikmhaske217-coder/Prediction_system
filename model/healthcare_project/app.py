import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

st.title("🏥 Patient Risk Prediction System")

# User Inputs
age = st.number_input("Age", 1, 100)
bp = st.number_input("Blood Pressure")
sugar = st.number_input("Sugar Level")
heart_rate = st.number_input("Heart Rate")

# Predict
if st.button("Predict Risk"):
    input_data = np.array([[age, bp, sugar, heart_rate]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Risk Probability: {round(probability, 2)}")

    # Alert Logic
    if probability > 0.7:
        st.error("🚨 High Risk! Consult Doctor Immediately")
    elif probability > 0.4:
        st.warning("⚠️ Medium Risk! Monitor Health")
    else:
        st.success("✅ Low Risk! Keep Healthy")