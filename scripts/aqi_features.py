import requests
import os
import pandas as pd
from datetime import datetime, timezone

API_TOKEN = os.getenv("AQICN_API_TOKEN")
CITY = "Karachi"

def run_feature_pipeline():
    # --- Load existing history ---
    try:
        history = pd.read_csv("data/karachi_aqi_last1yr.csv")
        history["timestamp"] = pd.to_datetime(history["timestamp"])
    except FileNotFoundError:
        history = pd.DataFrame(columns=["timestamp", "aqi", "pm25"])

    # --- Fetch current AQI ---
    url = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"
    r = requests.get(url).json()
    data = r.get("data", {})   # safe get

    if not data:
        print("No data returned from API")
        return history

    # Current timestamp (timezone-aware)
    now = datetime.now(timezone.utc)

    # Extract AQI & PM2.5 safely
    aqi = data.get("aqi")
    pm25 = data.get("iaqi", {}).get("pm25", {}).get("v")

    # Append current data
    history.loc[len(history)] = [now, aqi, pm25]

    # --- Feature Engineering ---
    history["aqi_lag_1"] = history["aqi"].shift(1)
    history["aqi_lag_3"] = history["aqi"].shift(3)
    history["aqi_change_rate"] = history["aqi"] - history["aqi_lag_1"]
    history["month"] = history["timestamp"].dt.month
    history["day_of_week"] = history["timestamp"].dt.weekday
    history["is_weekend"] = history["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Drop first rows with NaN lags
    history = history.dropna().reset_index(drop=True)

    # Save updated history
    history.to_csv("data/karachi_aqi_last1yr.csv", index=False)
    print("Feature pipeline done")
    return history

if __name__ == "__main__":
    run_feature_pipeline()
