import requests, streamlit as st

st.title("🌫 Karachi AQI Predictor")

if st.button("Predict Next 3 Days"):
    res = requests.get("http://localhost:8000/predict")
    for i, v in enumerate(res.json()["forecast"], 1):
        st.metric(f"Day {i}", f"{v}")
