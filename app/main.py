import streamlit as st
import hopsworks
import pandas as pd
import os
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    layout="wide"
)

# =====================================================
# GLOBAL STYLES (SaaS / Dashboard Look)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #f4f6fb;
}
.card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #6b7280;
}
.metric-value {
    font-size: 34px;
    font-weight: 700;
    color: #111827;
}
.status {
    font-size: 16px;
    font-weight: 600;
}
.header {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.subheader {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='header'>Karachi AQI Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Live Air Quality Monitoring & Forecast</div>", unsafe_allow_html=True)

# =====================================================
# CONFIG
# =====================================================
PROJECT = "AQIPred"
AQI_FG = "karachi_aqishine_fg"
FORECAST_FG = "aqi_forecast_fg"

# =====================================================
# HOPSWORKS LOGIN
# =====================================================
project = hopsworks.login(
    project=PROJECT,
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

# =====================================================
# AQI STATUS FUNCTION
# =====================================================
def aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#16a34a"
    elif aqi <= 100:
        return "Moderate", "#facc15"
    elif aqi <= 150:
        return "Unhealthy", "#fb923c"
    else:
        return "Severe", "#dc2626"

# =====================================================
# LOAD DATA
# =====================================================
aqi_fg = fs.get_feature_group(AQI_FG, version=1)
forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)

df_aqi = aqi_fg.read().sort_values("date")
df_forecast = forecast_fg.read()

latest_aqi = int(df_aqi.iloc[-1]["aqi"])
status, color = aqi_status(latest_aqi)

# =====================================================
# KPI CARDS
# =====================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Current AQI</div>
        <div class="metric-value">{latest_aqi}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Air Quality</div>
        <div class="status" style="color:{color}">{status}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="metric-title">City</div>
        <div class="metric-value">Karachi</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Forecast Model</div>
        <div class="metric-value">ML-Based</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =====================================================
# AQI TREND (INTERACTIVE)
# =====================================================
st.markdown("### Air Quality Trend")

df_trend = df_aqi.tail(48)  # last 48 readings

fig_trend = px.line(
    df_trend,
    y="aqi",
    markers=True,
    template="plotly_white"
)

fig_trend.update_layout(
    height=420,
    hovermode="x unified",
    showlegend=False,
    xaxis_title=None,
    yaxis_title="AQI",
)

st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# FORECAST SECTION
# =====================================================
st.markdown("### 3-Day AQI Forecast")

df_aqi["aqi"] = pd.to_numeric(df_aqi["aqi"], errors="coerce")
df_aqi = df_aqi.dropna(subset=["aqi"])
df_aqi = df_aqi.sort_values("date")

df_trend = df_aqi.tail(48)

fig_trend = px.line(
    df_trend,
    y="aqi",
    markers=True,
    template="plotly_white"
)

fig_trend.update_layout(
    height=420,
    hovermode="x unified",
    showlegend=False,
    xaxis_visible=False,
    yaxis_title="AQI"
)

st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# LOWER VISUALS (ANALYTICS)
# =====================================================
colA, colB = st.columns(2)

with colA:
    fig_dist = px.histogram(
        df_aqi,
        x="aqi",
        nbins=20,
        title="AQI Distribution",
        template="plotly_white"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with colB:
    df_aqi["severity"] = df_aqi["aqi"].apply(lambda x: aqi_status(x)[0])
    fig_pie = px.pie(
        df_aqi,
        names="severity",
        title="AQI Severity Breakdown"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("AQI Forecasting Platform • Powered by Hopsworks & Streamlit")
