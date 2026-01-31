# src/features.py
import pandas as pd
from datetime import datetime

def calculate_aqi_pm25(pm25):
    if pm25 <= 12:
        return (50 / 12) * pm25
    elif pm25 <= 35.4:
        return 50 + ((pm25 - 12.1) / (35.4 - 12.1)) * 50
    elif pm25 <= 55.4:
        return 100 + ((pm25 - 35.5) / (55.4 - 35.5)) * 50
    elif pm25 <= 150.4:
        return 150 + ((pm25 - 55.5) / (150.4 - 55.5)) * 100
    else:
        return 300

def run_feature_pipeline():
    df = pd.read_csv("data/karachi_aqi_last1year.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["aqi"] = df["pm25"].apply(calculate_aqi_pm25)

    df.to_csv("data/aqi_features.csv", index=False)
    print("Features generated")

if __name__ == "__main__":
    run_feature_pipeline()
