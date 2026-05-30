import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sqlalchemy import create_engine, text
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
from config import POSTGRES_CLEAN_CONN, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, PROMOTION_MAE_IMPROVEMENT_PCT

logger = logging.getLogger(__name__)

def train_and_promote(**context):
    ti = context['ti']
    train_reason = ti.xcom_pull(key='train_reason', task_ids='decide_training')
    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()

    engine = create_engine(POSTGRES_CLEAN_CONN)
    df_train = pd.read_sql("SELECT * FROM clean_properties WHERE split='train'", engine)
    df_val   = pd.read_sql("SELECT * FROM clean_properties WHERE split='val'", engine)
    df_test  = pd.read_sql("SELECT * FROM clean_properties WHERE split='test'", engine)
    engine.dispose()

    feature_cols = ["brokered_by", "status", "bed", "bath", "acre_lot",
                    "street", "city", "state", "zip_code", "house_size", "prev_sold_year"]
    target = "price"

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target]
    X_val   = df_val[feature_cols].fillna(0)
    y_val   = df_val[target]
    X_test  = df_test[feature_cols].fillna(0)
    y_test  = df_test[target]

    candidates = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    best_run_id = None
    best_val_mae = float("inf")
    best_model_name = None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"batch_{batch_number}_{name}") as run:
            model.fit(X_train, y_train)
            val_preds  = model.predict(X_val)
            test_preds = model.predict(X_test)
            val_mae  = mean_absolute_error(y_val, val_preds)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            val_r2   = r2_score(y_val, val_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            test_rmse= np.sqrt(mean_squared_error(y_test, test_preds))
            test_r2  = r2_score(y_test, test_preds)
            mlflow.log_params({"model": name, "batch_number": batch_number, "train_reason": train_reason})
            mlflow.log_metrics({
                "val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2,
                "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2,
                "train_size": len(X_train), "val_size": len(X_val)
            })
            mlflow.sklearn.log_model(model, "model", registered_model_name=MLFLOW_MODEL_NAME)
            logger.info(f"{name}: val_mae={val_mae:.2f}, val_rmse={val_rmse:.2f}, val_r2={val_r2:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run.info.run_id
                best_model_name = name

    logger.info(f"Best candidate: {best_model_name} with val_mae={best_val_mae:.2f}")

    promoted = False
    promotion_reason = ""
    try:
        champion = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
        champ_run = client.get_run(champion.run_id)
        champ_mae = champ_run.data.metrics.get("val_mae", float("inf"))
        improvement = (champ_mae - best_val_mae) / champ_mae
        if improvement >= PROMOTION_MAE_IMPROVEMENT_PCT:
            promoted = True
            promotion_reason = f"MAE improved {improvement:.2%} over champion ({champ_mae:.2f} -> {best_val_mae:.2f})"
        else:
            promotion_reason = f"No improvement: candidate MAE {best_val_mae:.2f} vs champion {champ_mae:.2f}"
    except Exception:
        promoted = True
        promotion_reason = "No existing champion, promoting first model"

    if promoted:
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        best_version = None
        for v in versions:
            if v.run_id == best_run_id:
                best_version = v.version
                break
        if best_version:
            client.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", best_version)
            logger.info(f"Promoted version {best_version} as champion: {promotion_reason}")

    ti.xcom_push(key='promoted', value=promoted)
    ti.xcom_push(key='promotion_reason', value=promotion_reason)
    ti.xcom_push(key='best_val_mae', value=best_val_mae)
    ti.xcom_push(key='best_model_name', value=best_model_name)

def skip_training(**context):
    ti = context['ti']
    reason = ti.xcom_pull(key='train_reason', task_ids='decide_training')
    logger.info(f"Training skipped: {reason}")

def log_result(**context):
    ti = context['ti']
    engine = create_engine(POSTGRES_CLEAN_CONN.replace("/clean_db", "/mlops_db"))
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS training_history (
                id SERIAL PRIMARY KEY,
                batch_number INTEGER,
                trained BOOLEAN,
                train_reason TEXT,
                promoted BOOLEAN,
                promotion_reason TEXT,
                best_model VARCHAR(50),
                best_val_mae FLOAT,
                logged_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            INSERT INTO training_history
            (batch_number, trained, train_reason, promoted, promotion_reason, best_model, best_val_mae)
            VALUES (:bn, :trained, :tr, :promoted, :pr, :bm, :mae)
        """), {
            "bn": ti.xcom_pull(key='batch_number', task_ids='fetch_batch'),
            "trained": ti.xcom_pull(key='train_decision', task_ids='decide_training') or False,
            "tr": ti.xcom_pull(key='train_reason', task_ids='decide_training') or "",
            "promoted": ti.xcom_pull(key='promoted', task_ids='train_and_promote') or False,
            "pr": ti.xcom_pull(key='promotion_reason', task_ids='train_and_promote') or "",
            "bm": ti.xcom_pull(key='best_model_name', task_ids='train_and_promote') or "",
            "mae": ti.xcom_pull(key='best_val_mae', task_ids='train_and_promote') or 0,
        })
    engine.dispose()
