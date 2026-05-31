from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/tasks")

from fetch_batch import fetch_batch
from store_raw import store_raw_batch
from validate import validate_schema, validate_quality, detect_drift, decide_training
from preprocess import preprocess_data
from train import (
    train_candidates,
    compare_models,
    promote_or_reject,
    skip_training,
    log_result,
    train_and_promote,  # legacy: conservado para compatibilidad
)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="mlops_pipeline",
    default_args=default_args,
    description="MLOps pipeline: ingest, validate, train, promote",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "real_estate"],
) as dag:

    start = EmptyOperator(task_id="start")

    t_fetch = PythonOperator(
        task_id="fetch_batch",
        python_callable=fetch_batch,
    )

    t_store_raw = PythonOperator(
        task_id="store_raw",
        python_callable=store_raw_batch,
    )

    t_validate_schema = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema,
    )

    t_validate_quality = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality,
    )

    t_detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    t_decide = BranchPythonOperator(
        task_id="decide_training",
        python_callable=decide_training,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t_train_candidates = PythonOperator(
        task_id="train_candidates",
        python_callable=train_candidates,
    )

    t_compare = PythonOperator(
        task_id="compare_models",
        python_callable=compare_models,
    )

    t_promote = PythonOperator(
        task_id="promote_or_reject",
        python_callable=promote_or_reject,
    )

    t_skip = PythonOperator(
        task_id="skip_training",
        python_callable=skip_training,
    )

    t_log = PythonOperator(
        task_id="log_result",
        python_callable=log_result,
        trigger_rule="none_failed_min_one_success",
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    start >> t_fetch >> t_store_raw >> t_validate_schema >> t_validate_quality >> t_detect_drift >> t_decide
    t_decide >> t_preprocess >> t_train_candidates >> t_compare >> t_promote >> t_log >> end
    t_decide >> t_skip >> t_log >> end
