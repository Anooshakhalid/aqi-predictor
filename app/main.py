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
# Custom CSS
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
    .card {
        background:#f9fafb;
        padding:30px;
        border-radius:18px;
        text-align:center;
        box-shadow:0 4px 10px rgba(0,0,0,0.08);
        margin-bottom:20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Karachi AQI Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Real-time Air Quality & Forecast</div>", unsafe_allow_html=True)

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
# AQI Status + Health Tips
# --------------------------------
def aqi_info(aqi):
    if aqi <= 50:
        return "🟢 Good", "#16a34a", "Air quality is ideal. Enjoy outdoor activities freely."
    elif aqi <= 100:
        return "🟡 Moderate", "#facc15", "Sensitive people should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "🟠 Unhealthy", "#fb923c", "Children, elderly, and patients should avoid outdoor activity."
    else:
        return "🔴 Severe", "#dc2626", "Everyone should stay indoors and wear masks if going outside."

# --------------------------------
# Current AQI Section
# --------------------------------
st.markdown("### >> Current Air Quality in Karachi")

try:
    aqi_fg = fs.get_feature_group(AQI_FG, version=1)
    df_aqi = aqi_fg.read().sort_values("date", ascending=False)

    if not df_aqi.empty:
        latest = df_aqi.iloc[0]
        latest_aqi = int(latest["aqi"])
        status, color, tip = aqi_info(latest_aqi)

        st.markdown(
            f"""
            <div class="card">
                <h1 style="font-size:60px; margin:0;">{latest_aqi}</h1>
                <h3 style="color:{color}; margin:10px 0;">{status}</h3>
                <p style="color:#6b7280;">Last Updated: {latest['date']}</p>
                <hr>
                <p style="font-size:16px;"><b>Health Tip:</b> {tip}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Summary
        st.markdown("### 📊 Last 24 Readings Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Min AQI", int(df_aqi.head(24)["aqi"].min()))
        with col2:
            st.metric("Max AQI", int(df_aqi.head(24)["aqi"].max()))
        with col3:
            st.metric("Avg AQI", int(df_aqi.head(24)["aqi"].mean()))

    else:
        st.warning("No AQI data available yet.")

except Exception as e:
    st.error(f"Failed to load AQI data: {e}")

# --------------------------------
# Forecast Section
# --------------------------------
st.markdown("---")
st.markdown("### >> AQI Forecast")

try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df_forecast = forecast_fg.read().sort_values("date")

    if not df_forecast.empty:
        cols = st.columns(len(df_forecast.tail(3)))

        for col, (_, row) in zip(cols, df_forecast.tail(3).iterrows()):
            aqi = int(row["pred_aqi"])
            status, color, tip = aqi_info(aqi)

            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <h2>{aqi}</h2>
                        <p style="color:{color}; font-weight:600;">{status}</p>
                        <small>{row['date']}</small>
                        <hr>
                        <small>{tip}</small>
                    </div>
                    """,
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
st.caption("AQI Monitoring System • Powered by Hopsworks & Streamlit")
