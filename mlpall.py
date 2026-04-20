import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load data
data = {
    "age": [25, 45, 50, 30, 60, 35],
    "bp": [120, 140, 160, 130, 170, 125],
    "sugar": [90, 150, 180, 100, 200, 110],
    "heart_rate": [72, 85, 90, 75, 95, 78],
    "risk": [0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['age', 'bp', 'sugar', 'heart_rate']]
y = df['risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# Train All Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(),
}

# Train and Save Models
os.makedirs("model", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Load Models
loaded_models = {}
for name in models.keys():
    loaded_models[name] = pickle.load(open(f"model/{name}.pkl", "rb"))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏥 Patient Risk Prediction System ")

model_choice = st.selectbox("Select Model", list(loaded_models.keys()))

# Inputs
age = st.number_input("Age", 1, 100)
bp = st.number_input("Blood Pressure")
sugar = st.number_input("Sugar Level")
heart_rate = st.number_input("Heart Rate")

# Prediction
if st.button("Predict Risk"):
    input_data = np.array([[age, bp, sugar, heart_rate]])

    model = loaded_models[model_choice]

    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Risk Probability: {round(probability, 2)}")

    # Alert Logic
    if probability > 0.7:
        st.error("🚨 High Risk! Consult Doctor Immediately")
    elif probability > 0.4:
        st.warning("⚠️ Medium Risk! Monitor Health")
    else:
        st.success("✅ Low Risk! Keep Healthy")