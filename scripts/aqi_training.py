# src/train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_training():
    df = pd.read_csv("data/karachi_aqi_features.csv")

    X = df[["pm25"]]
    y = df["aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    joblib.dump(model, "models/aqi_model.pkl")
    print("Model trained | RMSE:", rmse)

if __name__ == "__main__":
    run_training()
