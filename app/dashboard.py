import streamlit as st

st.title("🌫 Karachi AQI Predictor")

# Connect to Hopsworks
import hopsworks
project = hopsworks.login(api_key_value="YOUR_API_KEY", project="AQIPred")
fs = project.get_feature_store()

# Read predictions
pred_fg = fs.get_feature_group("aqi_forecast_fg", version=1)
df = pred_fg.read()  # latest predictions

for i, row in df.iterrows():
    st.metric(f"Day {i+1}", f"{row['pred_aqi']}")
