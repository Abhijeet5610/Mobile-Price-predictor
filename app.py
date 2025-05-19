import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📱 Mobile Price Range Predictor")

features = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_deep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]

user_input = []
for feat in features:
    val = st.number_input(f"{feat}:", value=0.0 if feat not in ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"] else 0, step=1.0)
    user_input.append(val)

if st.button("Predict Price Range"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    price_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    st.success(f"Predicted Price Range: **{price_map.get(pred, pred)}**")
