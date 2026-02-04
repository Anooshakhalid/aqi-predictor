import requests
import streamlit as st

st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("🌍 AQI Prediction Dashboard")

if st.button("Get 3-Day AQI Forecast"):
    res = requests.get("http://localhost:8000/predict")
    preds = res.json()["next_3_days_aqi"]

    for i, val in enumerate(preds, 1):
        st.metric(f"Day {i}", f"{val} AQI")
