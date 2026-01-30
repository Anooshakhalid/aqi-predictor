import pandas as pd
from aqi_features import run_feature_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run_training_pipeline():
    history = run_feature_pipeline()
    X = history.drop(columns=["aqi", "timestamp"])
    y = history["aqi"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf_aqi_model.pkl")

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    joblib.dump(ridge, "models/ridge_aqi_model.pkl")

    # Neural Network
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    nn_model.compile(optimizer="adam", loss="mse")
    nn_model.fit(X_train_s, y_train, epochs=50, batch_size=16, verbose=0)
    nn_model.save("models/nn_aqi_model.keras")

    print("Training pipeline done")

    # Metrics
    rf_pred = rf.predict(X_test)
    ridge_pred = ridge.predict(X_test_s)
    nn_pred = nn_model.predict(X_test_s).flatten()

    for name, pred in zip(["RF","Ridge","NN"], [rf_pred, ridge_pred, nn_pred]):
        print(f"{name} MSE: {mean_squared_error(y_test, pred):.2f}, "
              f"R2: {r2_score(y_test, pred):.2f}, "
              f"MAE: {mean_absolute_error(y_test, pred):.2f}")

if __name__ == "__main__":
    run_training_pipeline()
