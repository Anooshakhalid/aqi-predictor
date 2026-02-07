import streamlit as st
import hopsworks
import pandas as pd
import os

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# --------------------------------
# Advanced CSS Styling
# --------------------------------
st.markdown("""
<style>
body {
    background-color: #f9fafb;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: #111827;
}

.sub {
    text-align: center;
    color: #6b7280;
    margin-bottom: 40px;
    font-size: 16px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}

.center {
    text-align: center;
}

.aqi-value {
    font-size: 70px;
    font-weight: 800;
    margin: 0;
}

.aqi-label {
    font-size: 18px;
    color: #6b7280;
}

.forecast-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------
st.markdown("<div class='main-title'>Karachi AQI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Live Air Quality Monitoring & Forecast</div>", unsafe_allow_html=True)

# --------------------------------
# Config
# --------------------------------
PROJECT = "AQIPred"
AQI_FG = "karachi_aqishine_fg"
FORECAST_FG = "aqi_forecast_fg"

# --------------------------------
# Login to Hopsworks
# --------------------------------
project = hopsworks.login(
    project=PROJECT,
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

# --------------------------------
# AQI Status Helper
# --------------------------------
def aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#16a34a"
    elif aqi <= 100:
        return "Moderate", "#facc15"
    elif aqi <= 150:
        return "Unhealthy", "#fb923c"
    else:
        return "Severe", "#dc2626"

# ==================================
# CURRENT AQI (MAIN FOCUS)
# ==================================
try:
    aqi_fg = fs.get_feature_group(AQI_FG, version=1)
    df_aqi = aqi_fg.read().sort_values("date", ascending=False)

    if not df_aqi.empty:
        latest_aqi = int(df_aqi.iloc[0]["aqi"])
        status, color = aqi_status(latest_aqi)

        left, center, right = st.columns([1, 2, 1])

        with center:
            st.markdown(f"""
            <div class="card center">
                <div class="aqi-label">Current AQI</div>
                <div class="aqi-value" style="color:{color}">{latest_aqi}</div>
                <div style="color:{color}; font-size:20px; font-weight:600">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        # AQI Bar
        st.progress(min(latest_aqi / 300, 1.0))

        # Trend
        st.markdown("### AQI Trend (Recent)")
        st.line_chart(df_aqi["aqi"].head(30))

    else:
        st.warning("No AQI data available.")

except Exception as e:
    st.error(f"Failed to load AQI data: {e}")

# ==================================
# FORECAST SECTION
# ==================================
st.markdown("## 3-Day AQI Forecast")

try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df_forecast = forecast_fg.read()

    if not df_forecast.empty:
        cols = st.columns(3)

        for col, (_, row) in zip(cols, df_forecast.tail(3).iterrows()):
            forecast_aqi = int(row["pred_aqi"])
            status, color = aqi_status(forecast_aqi)

            with col:
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="aqi-label">Predicted AQI</div>
                    <h2 style="color:{color}; margin:10px 0">{forecast_aqi}</h2>
                    <div style="color:{color}; font-weight:600">{status}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("No forecast data available.")

except Exception as e:
    st.error(f"Failed to load forecast data: {e}")

# --------------------------------
# Footer
# --------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='footer'>AQI Forecasting System • Powered by Hopsworks & Streamlit</div>", unsafe_allow_html=True)
