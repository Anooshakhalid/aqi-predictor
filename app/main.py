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
    layout="centered"
)

# --------------------------------
# Custom CSS (Aesthetic)
# --------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        color: #1f2937;
    }
    .sub {
        text-align: center;
        color: #6b7280;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Karachi AQI Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Real-time Air Quality Insights & Forecast</div>", unsafe_allow_html=True)

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
        return "🟢 Good", "#16a34a"
    elif aqi <= 100:
        return "🟡 Moderate", "#facc15"
    elif aqi <= 150:
        return "🟠 Unhealthy", "#fb923c"
    else:
        return "🔴 Severe", "#dc2626"

# --------------------------------
# Current AQI Section
# --------------------------------
st.markdown("### 🌫️ Current AQI")

try:
    aqi_fg = fs.get_feature_group(AQI_FG, version=1)
    df_aqi = aqi_fg.read().sort_values("date", ascending=False)

    if not df_aqi.empty:
        latest_aqi = int(df_aqi.iloc[0]["aqi"])
        status, color = aqi_status(latest_aqi)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("AQI Value", latest_aqi)

        with col2:
            st.markdown(
                f"<h3 style='color:{color}'>{status}</h3>",
                unsafe_allow_html=True
            )

        # AQI Trend Chart (no date labels)
        st.markdown("#### AQI Trend")
        st.line_chart(df_aqi["aqi"].head(30))

    else:
        st.warning("No AQI data available yet.")

except Exception as e:
    st.error(f"Failed to load AQI data: {e}")

# --------------------------------
# Forecast Section
# --------------------------------
st.markdown("---")
st.markdown("### 📈 AQI Forecast")

try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df_forecast = forecast_fg.read()

    if not df_forecast.empty:
        cols = st.columns(len(df_forecast.tail(3)))

        for col, (_, row) in zip(cols, df_forecast.tail(3).iterrows()):
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

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("🚀 AQI Forecasting System • Powered by Hopsworks & Streamlit")
