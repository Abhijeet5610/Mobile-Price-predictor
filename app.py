import streamlit as st
import joblib
import numpy as np

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="📱 Mobile Price Range Predictor", layout="centered")
st.title("Mobile Price Range Predictor")
st.markdown("Predict the price category of a smartphone based on its features.")

int_features = {
    "battery_power": "mAh",
    "blue": "Yes(1)/No(0)",
    "dual_sim": "Yes(1)/No(0)",
    "fc": "MP",
    "four_g": "Yes(1)/No(0)",
    "int_memory": "GB",
    "mobile_wt": "grams",
    "n_cores": "cores",
    "pc": "MP",
    "px_height": "pixels",
    "px_width": "pixels",
    "ram": "MB",
    "sc_h": "cm",
    "sc_w": "cm",
    "talk_time": "hours",
    "three_g": "Yes(1)/No(0)",
    "touch_screen": "Yes(1)/No(0)",
    "wifi": "Yes(1)/No(0)",
}

float_features = {
    "clock_speed": "GHz",
    "m_deep": "cm",
}

feature_ranges = {
    "battery_power": (2000, 7000),
    "blue": (0, 1),
    "clock_speed": (1.0, 3.5),
    "dual_sim": (0, 1),
    "fc": (0, 50),
    "four_g": (0, 1),
    "int_memory": (16, 512),
    "m_deep": (0.1, 1.0),
    "mobile_wt": (120, 250),
    "n_cores": (2, 12),
    "pc": (5, 200),
    "px_height": (720, 3200),
    "px_width": (1280, 1440),
    "ram": (512, 24576),
    "sc_h": (12, 19),
    "sc_w": (6, 10),
    "talk_time": (5, 35),
    "three_g": (0, 1),
    "touch_screen": (0, 1),
    "wifi": (0, 1)
}

user_input = []
st.markdown("### Enter Mobile Specifications")
for feat, unit in int_features.items():
    min_val, max_val = feature_ranges.get(feat, (0, 100))
    val = st.number_input(
        f"{feat.replace('_', ' ').title()} ({unit})", 
        min_value=min_val, 
        max_value=max_val, 
        value=(min_val + max_val) // 2, 
        step=1
    )
    user_input.append(val)

for feat, unit in float_features.items():
    min_val, max_val = feature_ranges.get(feat, (0.0, 1.0))
    val = st.number_input(
        f"{feat.replace('_', ' ').title()} ({unit})", 
        min_value=min_val, 
        max_value=max_val, 
        value=round((min_val + max_val) / 2, 1), 
        step=0.1,
        format="%.1f"
    )
    user_input.append(val)

if st.button("Predict Price Range"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    price_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    st.success(f" Predicted Price Range: **{price_map.get(pred, pred)}**")


