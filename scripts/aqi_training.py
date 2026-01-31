import os
import pandas as pd
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Hopsworks Login
# -------------------------
def login_hopsworks(api_key, project_name="AQIPred"):
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqishine_fg", version=1)
    mr = project.get_model_registry()
    return fg, mr

# -------------------------
# Fetch all historical data
# -------------------------
def fetch_features(fg):
    df = fg.read()
    df = df.sort_values("date")  # Important for time series
    return df

# -------------------------
# Train models
# -------------------------
def train_models(X_train, X_test, y_train, y_test):
    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["RandomForest"] = {
        "model": rf,
        "mse": mean_squared_error(y_test, rf_pred),
        "r2": r2_score(y_test, rf_pred),
        "mae": mean_absolute_error(y_test, rf_pred)
    }

    # Ridge Regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    ridge_pred = ridge.predict(X_test_s)
    results["Ridge"] = {
        "model": ridge,
        "scaler": scaler,
        "mse": mean_squared_error(y_test, ridge_pred),
        "r2": r2_score(y_test, ridge_pred),
        "mae": mean_absolute_error(y_test, ridge_pred)
    }

    # Neural Network
    nn = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    nn.compile(optimizer="adam", loss="mse")
    nn.fit(X_train_s, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)
    nn_pred = nn.predict(X_test_s).flatten()
    results["NN"] = {
        "model": nn,
        "scaler": scaler,
        "mse": mean_squared_error(y_test, nn_pred),
        "r2": r2_score(y_test, nn_pred),
        "mae": mean_absolute_error(y_test, nn_pred)
    }

    return results

# -------------------------
# Save best model to Hopsworks Model Registry
# -------------------------
def save_best_model(mr, results):
    # Choose model with lowest MSE
    best_model_name = min(results, key=lambda k: results[k]["mse"])
    best_model_info = results[best_model_name]
    print(f"Best Model: {best_model_name} | MSE: {best_model_info['mse']:.2f} | R2: {best_model_info['r2']:.2f}")

    model_name = "AQI_Predictor"
    model_version = mr.get_model(model_name).get_next_version() if mr.exists(model_name) else 1

    if best_model_name == "NN":
        best_model_info["model"].save(f"{model_name}.h5")
        mr_model = mr.python.create_model(
            name=model_name,
            metrics={"mse": best_model_info["mse"], "r2": best_model_info["r2"]},
            model_file=f"{model_name}.h5",
            description="Neural Network for AQI prediction"
        )
    else:
        joblib.dump(best_model_info["model"], f"{model_name}.pkl")
        mr_model = mr.python.create_model(
            name=model_name,
            metrics={"mse": best_model_info["mse"], "r2": best_model_info["r2"]},
            model_file=f"{model_name}.pkl",
            description=f"{best_model_name} model for AQI prediction"
        )

    print(f"Model saved in registry: {model_name} v{model_version}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    if not HOPSWORKS_API_KEY:
        raise EnvironmentError("Please set HOPSWORKS_API_KEY in your environment.")

    fg, mr = login_hopsworks(HOPSWORKS_API_KEY)

    df_fs = fetch_features(fg)
    y = df_fs["aqi"]
    X = df_fs.drop(columns=["aqi", "date"])

    # Train-test split (80% train, 20% test)
    split = int(len(df_fs) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results = train_models(X_train, X_test, y_train, y_test)
    save_best_model(mr, results)
