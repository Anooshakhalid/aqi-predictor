# Feature pipeline
def run_feature_pipeline():
    import os
    import pandas as pd
    import requests
    import hopsworks
    from datetime import datetime

    # Load historical data
    df = pd.read_csv("karachi_aqi_last1year.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()

    # Calculate AQI from PM2.5
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

    df['aqi'] = df['pm25'].apply(calculate_aqi_pm25)
    df = df.sort_values('date')

    # Features
    df['aqi_lag_1'] = df['aqi'].shift(1)
    df['aqi_lag_3'] = df['aqi'].shift(3)
    df['aqi_change_rate'] = df['aqi'].pct_change()
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df = df.dropna()

    df.to_csv("karachi_aqi_features.csv", index=False)
    print("Feature pipeline done")

    # Upload to Hopsworks
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="AQIPred"
    )
    fs = project.get_feature_store()

    feature_group = fs.create_feature_group(
        name="karachi_aqishine_fg",
        description="AQI features for Karachi",
        version=1,
        primary_key=["date"],
        online_enabled=True
    )
    feature_group.insert(df, write_options={"wait_for_job": True})
    print("Inserted features into Feature Store")

if __name__ == "__main__":
    run_feature_pipeline()