# Pearls AQI Predictor

## Project Overview

Pearls AQI Predictor is an end-to-end serverless machine learning system designed to predict the Air Quality Index (AQI) for the next three days for a given city. The system integrates automated data collection, feature engineering, model training, and real-time predictions through a web dashboard. It demonstrates a full MLOps workflow including feature store usage, automated CI/CD pipelines, model versioning, and explainability.

This system enables users and authorities to plan ahead for air quality concerns, moving beyond static dashboards and real-time monitoring.

## Technology Stack

The project uses the following technologies and tools:

- **Python**: Core programming language
- **Scikit-learn**: Classical machine learning models
- **TensorFlow**: Neural network models
- **Hopsworks**: Feature store and model registry
- **GitHub Actions**: CI/CD automation for pipelines
- **Streamlit**: Web dashboard interface
- **AQICN API**: Air quality data source
- **SHAP**: Model explainability
- **Git**: Version control

## Key Features

### Feature Engineering Pipeline
- Fetches raw pollutant data from AQICN.
- Computes derived features including time-based (hour, day, month), lag features, rolling averages, and AQI change rates.
- Stores processed features in Hopsworks feature store with versioning to ensure consistency between training and inference.

### Historical Data Backfill
- Generates past feature datasets by running the feature pipeline for previous timestamps.
- Creates comprehensive datasets for model training and evaluation.

### Model Training Pipeline
- Trains multiple models: Random Forest, Ridge Regression, and Neural Network.
- Evaluates models using RMSE, MAE, and R² metrics.
- Selects the best model based on highest R² score, while considering generalization, computational efficiency, and feature store compatibility.
- Registers the best-performing model in Hopsworks with automated versioning.

### Automation and CI/CD
- Feature pipeline runs automatically every hour.
- Model training pipeline executes daily to update models.
- GitHub Actions ensures automated scheduling, logging, and error handling.
- Versioned models enable reproducibility and rollback if needed.

### Web Application Dashboard
- Built using Streamlit to provide an interactive interface.
- Displays current AQI values and 3-day predictions for Karachi.

### Model Explainability
- SHAP is used to analyze feature importance for the best-performing model.
- Provides transparency by showing contributions of weather and pollutant features.
- SHAP plots are stored in a dedicated directory (`shap_plots/`).

## Model Selection Criteria

After training multiple models, a systematic evaluation process is followed to determine the best model for registration.  

**Evaluation Metrics:**
- **Root Mean Square Error (RMSE)** – measures average squared difference between predicted and actual AQI values.
- **Mean Absolute Error (MAE)** – measures average absolute prediction error.
- **R² Score** – indicates how well the model explains the variance in AQI values.

The model with the **highest R² score** on the validation dataset is selected for deployment, considering:
- **Generalization** – consistent performance on unseen data.
- **Feature Store Compatibility** – works correctly with Hopsworks features.
- **Computational Efficiency** – suitable for serverless daily retraining.
- **Explainability** – interpretable using SHAP.

Versioning is implemented so each registered model has a unique version and can be rolled back if necessary.

## Folder Structure
```
pearls-aqi-predictor/
│
├── .github/workflows/            # GitHub Actions CI/CD pipelines
│   ├── feature_pipeline.yml
│   ├── run_predictions.yml
│   └── training_pipeline.yml
│
├── app/                          # Web dashboard
│   ├── main.py
│   └── prediction_job.py
│
├── data/                         # Datasets
│   ├── karachi_aqi_features.csv
│   └── karachi_aqi_last1year.csv
│
├── documents/                    # Project documents
│   └── Pearl AQI Predictor.pdf
│
├── pipelines/                    # Pipeline notebooks
│   └── historical_data.ipynb
│
├── scripts/                      # Scripts for features and training
│   ├── aqi_features.py
│   └── aqi_training.py
│
├── shap_plots/                   # SHAP explainability plots
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore
```


## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/Anooshakhalid/pearls-aqi-predictor.git
cd pearls-aqi-predictor

