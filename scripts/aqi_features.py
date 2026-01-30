import pandas as pd
import requests
import os
from datetime import datetime

def run_feature_pipeline():
    API_TOKEN = os.getenv("AQICN_API_TOKEN")
    CITY = "Karachi"

    try:
        history = pd.read_csv("data/karachi_aqi_last1yr.csv")
        history["timestamp"] = pd.to_datetime(history["date"])
        history = history[["timestamp", "aqi", "pm25"]]
    except FileNotFoundError:
        history = pd.DataFrame(columns=["timestamp", "aqi", "pm25"])

    url = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"
    r = requests.get(url).json()
    d = r["data"]

    now = datetime.utcnow()
    aqi = d["aqi"]
    pm25 = d["iaqi"].get("pm25", {}).get("v")

    history.loc[len(history)] = [now, aqi, pm25]

    # Features
    history["aqi_lag_1"] = history["aqi"].shift(1)
    history["aqi_lag_3"] = history["aqi"].shift(3)
    history["aqi_change_rate"] = history["aqi"] - history["aqi_lag_1"]
    history["month"] = history["timestamp"].dt.month
    history["day_of_week"] = history["timestamp"].dt.weekday
    history["is_weekend"] = history["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    history = history.dropna().reset_index(drop=True)

    history.to_csv("data/karachi_aqi_last1yr.csv", index=False)
    print("Feature pipeline done")
    return history

if __name__ == "__main__":
    run_feature_pipeline()
