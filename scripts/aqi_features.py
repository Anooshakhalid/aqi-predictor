import os
import requests
import pandas as pd
from datetime import datetime
import hopsworks


# -------------------------
# 1. Login to Hopsworks
# -------------------------
def login_hopsworks(api_key: str, project_name: str):
    """
    Login to Hopsworks project and return feature store and feature group.
    """
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqishine_fg", version=1)
    return fg


# -------------------------
# 2. Fetch last rows for lag calculation
# -------------------------
def get_last_rows(fg, n=3):
    """
    Fetch last n rows from Hopsworks feature group.
    """
    try:
        df_last = fg.read()
        df_last = df_last.sort_values("date").tail(n)
        return df_last
    except Exception as e:
        print("No previous rows found:", e)
        return pd.DataFrame()


# -------------------------
# 3. Fetch current AQI (pm25) from API
# -------------------------
def fetch_current_pm25(api_key, city="Karachi"):
    """
    Fetch current PM2.5 value from WAQI API.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={api_key}"
    resp = requests.get(url).json()
    
    pm25 = resp['data']['iaqi']['pm25']['v']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return pd.DataFrame([{"date": timestamp, "pm25": int(pm25)}])


# -------------------------
# 4. Compute features
# -------------------------
def compute_features(new_row, last_rows):
    """
    Compute AQI, lag features, and time features.
    """
    pm25 = new_row.loc[0, "pm25"]
    
    # AQI calculation
    if pm25 <= 12:
        aqi = (50 / 12) * pm25
    elif pm25 <= 35.4:
        aqi = 50 + ((pm25 - 12.1) / (35.4 - 12.1)) * 50
    elif pm25 <= 55.4:
        aqi = 100 + ((pm25 - 35.5) / (55.4 - 35.5)) * 50
    elif pm25 <= 150.4:
        aqi = 150 + ((pm25 - 55.5) / (150.4 - 55.5)) * 100
    else:
        aqi = 300

    new_row["aqi"] = float(aqi)

    # Lag features
    if last_rows.empty:
        new_row["aqi_lag_1"] = float(aqi)
        new_row["aqi_lag_3"] = float(aqi)
        new_row["aqi_change_rate"] = 0.0
    else:
        last_aqi_1 = float(last_rows["aqi"].iloc[-1])
        last_aqi_3 = float(last_rows["aqi"].iloc[-3]) if len(last_rows) >= 3 else float(last_rows["aqi"].iloc[0])
        
        new_row["aqi_lag_1"] = last_aqi_1
        new_row["aqi_lag_3"] = last_aqi_3
        new_row["aqi_change_rate"] = float((aqi - last_aqi_1) / last_aqi_1)

    # Time features
    dt = pd.to_datetime(new_row["date"].iloc[0])
    new_row["month"] = int(dt.month)
    new_row["day_of_week"] = int(dt.dayofweek)
    new_row["is_weekend"] = int(1 if dt.dayofweek >= 5 else 0)

    return new_row


# -------------------------
# 5. Insert new row to Feature Store
# -------------------------
def insert_new_row(fg, new_row):
    fg.insert(new_row, write_options={"wait_for_job": True})
    print("New row inserted successfully!")


# -------------------------
# 6. Run pipeline
# -------------------------
def run_pipeline(hopsworks_api_key, waqi_api_key, project_name="AQIPred"):
    fg = login_hopsworks(hopsworks_api_key, project_name)
    last_rows = get_last_rows(fg, n=3)
    new_row = fetch_current_pm25(waqi_api_key)
    new_row = compute_features(new_row, last_rows)
    insert_new_row(fg, new_row)
    return new_row


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Load keys from environment variables for security
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    WAQI_API_KEY = os.getenv("WAQI_API_KEY")
    
    if not HOPSWORKS_API_KEY or not WAQI_API_KEY:
        raise EnvironmentError("Please set HOPSWORKS_API_KEY and WAQI_API_KEY in your environment.")
    
    df_new = run_pipeline(HOPSWORKS_API_KEY, WAQI_API_KEY)
    print(df_new)
