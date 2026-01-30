import requests
import os
import pandas as pd
from datetime import datetime, timezone

API_TOKEN = os.getenv("AQICN_API_TOKEN")
CITY = "Karachi"

def run_feature_pipeline():
    # Load history
    try:
        history = pd.read_csv("data/karachi_aqi_last1yr.csv")
        history["timestamp"] = pd.to_datetime(history["timestamp"])
    except FileNotFoundError:
        history = pd.DataFrame(columns=["timestamp", "aqi", "pm25"])

    # Fetch current AQI
    url = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"
    r = requests.get(url)

    try:
        r_json = r.json()
    except ValueError:
        print("API did not return valid JSON!")
        return history

    # Ensure 'data' exists
    data = r_json.get("data")
    if not isinstance(data, dict):
        print("API returned data as string or list, cannot parse:")
        print(data)
        return history

    # Current timestamp
    now = datetime.now(timezone.utc)

    # Safe extraction
    aqi = data.get("aqi")
    pm25 = data.get("iaqi", {}).get("pm25", {}).get("v")

    # Append row
    history.loc[len(history)] = [now, aqi, pm25]

    # Features
    history["aqi_lag_1"] = history["aqi"].shift(1)
    history["aqi_lag_3"] = history["aqi"].shift(3)
    history["aqi_change_rate"] = history["aqi"] - history["aqi_lag_1"]
    history["month"] = history["timestamp"].dt.month
    history["day_of_week"] = history["timestamp"].dt.weekday
    history["is_weekend"] = history["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    history = history.dropna().reset_index(drop=True)

    # Save CSV
    history.to_csv("data/karachi_aqi_last1yr.csv", index=False)
    print("Feature pipeline done")
    return history

if __name__ == "__main__":
    run_feature_pipeline()