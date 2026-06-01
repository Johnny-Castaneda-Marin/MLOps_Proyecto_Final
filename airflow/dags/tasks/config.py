import os

DATA_API_URL = os.getenv("DATA_API_URL", "http://data-api-service")
GROUP_NUMBER = 5

POSTGRES_RAW_CONN = os.getenv("POSTGRES_RAW_CONN", "postgresql://mlops_user:mlops1234@postgres-service:5432/raw_db")
POSTGRES_CLEAN_CONN = os.getenv("POSTGRES_CLEAN_CONN", "postgresql://mlops_user:mlops1234@postgres-service:5432/clean_db")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MLFLOW_EXPERIMENT_NAME = "real_estate_price_prediction"
MLFLOW_MODEL_NAME = "real_estate_champion"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MIN_RECORDS_TO_TRAIN = 1000
MIN_VOLUME_INCREASE_PCT = 0.05
DRIFT_THRESHOLD = 0.1
PROMOTION_MAE_IMPROVEMENT_PCT = 0.03
