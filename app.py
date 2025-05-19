import streamlit as st
import joblib
import numpy as np

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Mobile Price Predictor", layout="centered")
st.title("📱 Mobile Price Range Predictor")
st.markdown("Fill in the specifications below to predict the mobile's price range.")

features = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_deep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]

int_features = [
    "battery_power", "blue", "dual_sim", "fc", "four_g",
    "int_memory", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]

feature_ranges = {
    "battery_power": (500, 2000),
    "clock_speed": (0.5, 3.5),
    "fc": (0, 20),
    "int_memory": (2, 128),
    "m_deep": (0.1, 1.0),
    "mobile_wt": (80, 250),
    "n_cores": (1, 8),
    "pc": (0, 20),
    "px_height": (0, 1960),
    "px_width": (0, 2000),
    "ram": (256, 4096),
    "sc_h": (5, 20),
    "sc_w": (0, 18),
    "talk_time": (2, 20),
}

def get_range(feat, is_int):
    if feat in feature_ranges:
        return feature_ranges[feat]
    return (0, 10) if is_int else (0.0, 10.0)

# Group inputs
user_input = []
with st.expander("📋 Enter Mobile Specifications", expanded=True):
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                feat = features[i + j]
                min_val, max_val = get_range(feat, feat in int_features)

                label = feat.replace('_', ' ').title()

                if feat in int_features:
                    min_val, max_val = int(min_val), int(max_val)
                    val = cols[j].number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        step=1
                    )
                else:
                    min_val, max_val = float(min_val), float(max_val)
                    val = cols[j].number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        step=0.1
                    )

                user_input.append(val)


if st.button("🔍 Predict Price Range"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    price_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    st.success(f"💰 **Predicted Price Range: {price_map.get(pred, pred)}**")

