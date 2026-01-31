import pandas as pd
from aqi_features import run_feature_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run_training_pipeline():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import joblib

    # Load features from CSV or Feature Store
    df = pd.read_csv("karachi_aqi_features.csv")

    # Features and target
    X = df.drop(columns=['date','aqi'])
    y = df['aqi']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, "rf_aqi_model.pkl")

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    joblib.dump(ridge, "ridge_aqi_model.pkl")

    print("Training pipeline done. Models saved.")

if __name__ == "__main__":
    run_training_pipeline()
