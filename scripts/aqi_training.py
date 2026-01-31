# -*- coding: utf-8 -*-
"""
AQI Training Pipeline
Includes Random Forest, Ridge, Neural Network
Evaluates models, saves best to Hopsworks Model Registry
Generates SHAP feature importance plots
"""

import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import shap
import matplotlib.pyplot as plt

# -------------------------
# 1. Hopsworks login & feature store
# -------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise EnvironmentError("Please set HOPSWORKS_API_KEY in your environment.")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="AQIPred")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="karachi_aqishine_fg", version=1)

# Read historical data
df_fs = fg.read().sort_values("date")

# -------------------------
# 2. Prepare features & target
# -------------------------
y = df_fs["aqi"]
X = df_fs.drop(columns=["aqi", "date"])

# Time-based split
split = int(len(df_fs) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Scale for Ridge & Neural Network
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------------------------
# 3. Train Random Forest
# -------------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# -------------------------
# 4. Train Ridge Regression
# -------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
ridge_pred = ridge.predict(X_test_s)

# -------------------------
# 5. Train Neural Network
# -------------------------
nn_model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])
nn_model.compile(optimizer="adam", loss="mse")
nn_model.fit(X_train_s, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)
nn_pred = nn_model.predict(X_test_s).flatten()

# -------------------------
# 6. Evaluate models
# -------------------------
metrics = {}
models = {"RandomForest": (rf, rf_pred, False), "Ridge": (ridge, ridge_pred, True), "NeuralNet": (nn_model, nn_pred, True)}

for name, (model, pred, scaled) in models.items():
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    metrics[name] = {"MSE": mse, "R2": r2, "MAE": mae}
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}, MAE: {mae:.2f}")

# Determine best model (highest R2)
best_model_name = max(metrics, key=lambda x: metrics[x]["R2"])
print(f"Best model: {best_model_name}")

# -------------------------
# 7. SHAP explainability
# -------------------------
shap_dir = "shap_plots"
os.makedirs(shap_dir, exist_ok=True)

if "RandomForest" in models:
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(X_test)
    shap.summary_plot(shap_values_rf, X_test, plot_type="bar", feature_names=X_test.columns, show=False)
    plt.savefig(os.path.join(shap_dir, "shap_rf.png"))
    plt.close()

if "Ridge" in models:
    explainer_ridge = shap.LinearExplainer(ridge, X_train_s)
    shap_values_ridge = explainer_ridge.shap_values(X_test_s)
    shap.summary_plot(shap_values_ridge, X_test_s, plot_type="bar", feature_names=X_test.columns, show=False)
    plt.savefig(os.path.join(shap_dir, "shap_ridge.png"))
    plt.close()

if "NeuralNet" in models:
    X_bg = shap.sample(X_train_s, 100)
    X_test_small = shap.sample(X_test_s, 50)
    explainer_nn = shap.KernelExplainer(nn_model.predict, X_bg)
    shap_values_nn = explainer_nn.shap_values(X_test_small)
    shap.summary_plot(shap_values_nn, X_test_small, plot_type="bar", feature_names=X_test.columns, show=False)
    plt.savefig(os.path.join(shap_dir, "shap_nn.png"))
    plt.close()

# -------------------------
# 8. Save best model to Hopsworks Model Registry
# -------------------------
mr = project.get_model_registry()

if best_model_name == "RandomForest":
    joblib.dump(rf, "best_model.pkl")
    best_model = mr.python.create_model(
        name="aqi_rf_model",
        version=1,
        description="Random Forest model for AQI prediction",
        metrics=metrics["RandomForest"]
    )
    best_model.save("best_model.pkl")

elif best_model_name == "Ridge":
    joblib.dump(ridge, "best_model.pkl")
    best_model = mr.python.create_model(
        name="aqi_ridge_model",
        version=1,
        description="Ridge Regression model for AQI prediction",
        metrics=metrics["Ridge"]
    )
    best_model.save("best_model.pkl")

elif best_model_name == "NeuralNet":
    nn_model.save("best_model.keras")
    best_model = mr.tensorflow.create_model(
        name="aqi_nn_model",
        version=1,
        description="Neural Network model for AQI prediction",
        metrics=metrics["NeuralNet"]
    )
    best_model.save("best_model.keras")

print(f"Best model {best_model_name} saved to Hopsworks Model Registry successfully!")
