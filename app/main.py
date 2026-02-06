import streamlit as st
import hopsworks
import pandas as pd
import os

# --------------------------------
# Page Config (wide is IMPORTANT)
# --------------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Karachi AQI Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Live Air Quality & 3-Day Forecast</div>", unsafe_allow_html=True)

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
        return "Good", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "#facc15"
    elif aqi <= 150:
        return "Unhealthy", "#fb923c"
    else:
        return "Severe", "#ef4444"

# =================================
# ROW 1 — CURRENT AQI (HORIZONTAL)
# =================================
st.markdown("Current AQI")

try:
    aqi_fg = fs.get_feature_group(AQI_FG, version=1)
    df_aqi = aqi_fg.read().sort_values("date", ascending=False)

    if not df_aqi.empty:
        latest_aqi = int(df_aqi.iloc[0]["aqi"])
        status, color = aqi_status(latest_aqi)

        col1, col2, col3 = st.columns([1, 1, 2])

        # AQI Value
        with col1:
            st.metric("AQI", latest_aqi)

        # Status
        with col2:
            st.markdown(
                f"<h3 style='color:{color}'>{status}</h3>",
                unsafe_allow_html=True
            )

        # Trend (horizontal space)
        with col3:
            st.caption("Recent Trend")
            st.line_chart(df_aqi["aqi"].head(20))

    else:
        st.warning("No AQI data available yet.")

except Exception as e:
    st.error(f"Failed to load AQI data: {e}")

# =================================
# ROW 2 — FORECAST (HORIZONTAL)
# =================================
st.markdown("---")
st.markdown("3-Day Forecast")

try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df_forecast = forecast_fg.read()

    if not df_forecast.empty:
        fcols = st.columns(3)

        for col, (_, row) in zip(fcols, df_forecast.tail(3).iterrows()):
            forecast_aqi = int(row["pred_aqi"])
            status, color = aqi_status(forecast_aqi)

            with col:
                st.metric("Predicted AQI", forecast_aqi)
                st.markdown(
                    f"<span style='color:{color}; font-weight:600'>{status}</span>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("No forecast available yet.")

except Exception as e:
    st.error(f"Failed to load forecast data: {e}")


