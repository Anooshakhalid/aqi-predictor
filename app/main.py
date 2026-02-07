import streamlit as st
import hopsworks
import pandas as pd
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Karachi AQI",
    page_icon="🌫️",
    layout="wide"
)

# ----------------------------
# Global CSS
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #e5e7eb;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 40px;
}
.metric-value {
    font-size: 48px;
    font-weight: 800;
}
.small {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='title'>Karachi Air Quality Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Live AQI monitoring & short-term forecast</div>", unsafe_allow_html=True)

# ----------------------------
# Config
# ----------------------------
PROJECT = "AQIPred"
AQI_FG = "karachi_aqishine_fg"
FORECAST_FG = "aqi_forecast_fg"

project = hopsworks.login(
    project=PROJECT,
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

# ----------------------------
# AQI Color Logic
# ----------------------------
def aqi_meta(aqi):
    if aqi <= 50:
        return "Good", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "#facc15"
    elif aqi <= 150:
        return "Unhealthy", "#fb923c"
    else:
        return "Severe", "#ef4444"

# ----------------------------
# Load AQI Data
# ----------------------------
aqi_fg = fs.get_feature_group(AQI_FG, version=1)
df = aqi_fg.read().sort_values("date")

latest = int(df.iloc[-1]["aqi"])
status, color = aqi_meta(latest)

# ----------------------------
# TOP ROW — CURRENT AQI
# ----------------------------
c1, c2, c3 = st.columns([2,1,1])

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="small">Current AQI</div>
        <div class="metric-value" style="color:{color}">{latest}</div>
        <div style="color:{color}; font-weight:600">{status}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="small">24h Average</div>
        <div class="metric-value">{int(df["aqi"].tail(24).mean())}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="small">Peak AQI</div>
        <div class="metric-value">{int(df["aqi"].tail(24).max())}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# TREND VISUALIZATION
# ----------------------------
st.markdown("AQI Trend (Recent Readings)")
trend_df = df.tail(48)[["aqi"]]
st.line_chart(trend_df, height=320)

# ----------------------------
# FORECAST
# ----------------------------
st.markdown("3-Day Forecast")

forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
forecast_df = forecast_fg.read().tail(3)

cols = st.columns(3)

for col, (_, row) in zip(cols, forecast_df.iterrows()):
    val = int(row["pred_aqi"])
    label, c = aqi_meta(val)

    with col:
        st.markdown(f"""
        <div class="card">
            <div class="small">Predicted AQI</div>
            <div class="metric-value" style="color:{c}">{val}</div>
            <div style="color:{c}; font-weight:600">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.caption("Powered by Hopsworks • Streamlit • AQI ML Pipeline")
