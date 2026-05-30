import logging
import pandas as pd
from sqlalchemy import create_engine, text
from config import POSTGRES_RAW_CONN, MIN_RECORDS_TO_TRAIN, DRIFT_THRESHOLD

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["brokered_by", "status", "price", "bed", "bath",
                    "acre_lot", "street", "city", "state", "zip_code",
                    "house_size", "prev_sold_date"]

def validate_schema(**context):
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    if not records:
        ti.xcom_push(key='schema_valid', value=False)
        return
    sample = records[0]
    received_cols = set(sample.keys())
    expected_cols = set(EXPECTED_COLUMNS)
    missing = expected_cols - received_cols
    extra = received_cols - expected_cols
    issues = []
    if missing:
        issues.append(f"Missing columns: {missing}")
    if extra:
        issues.append(f"Extra columns: {extra}")
    if issues:
        logger.warning(f"Schema issues: {issues}")
        ti.xcom_push(key='schema_valid', value=False)
        ti.xcom_push(key='schema_issues', value=str(issues))
    else:
        logger.info("Schema validation passed")
        ti.xcom_push(key='schema_valid', value=True)
        ti.xcom_push(key='schema_issues', value="none")

def validate_quality(**context):
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    if not records:
        ti.xcom_push(key='quality_valid', value=False)
        return
    df = pd.DataFrame(records)
    issues = []
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.5].index.tolist()
    if high_null:
        issues.append(f"High null rate in: {high_null}")
    if "price" in df.columns:
        neg_price = (df["price"] <= 0).sum()
        if neg_price > 0:
            issues.append(f"{neg_price} records with invalid price")
        if df["price"].isnull().mean() > 0.3:
            issues.append("More than 30% null prices")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows")
    total = len(df)
    logger.info(f"Quality check: {total} records, {len(issues)} issues")
    ti.xcom_push(key='quality_valid', value=len(issues) == 0)
    ti.xcom_push(key='quality_issues', value=str(issues) if issues else "none")
    ti.xcom_push(key='total_records', value=total)

def detect_drift(**context):
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    if not records:
        ti.xcom_push(key='drift_detected', value=False)
        return
    df_new = pd.DataFrame(records)
    engine = create_engine(POSTGRES_RAW_CONN)
    try:
        df_hist = pd.read_sql("SELECT price, house_size, bed, bath FROM raw_properties WHERE processed=TRUE LIMIT 10000", engine)
    except Exception:
        df_hist = pd.DataFrame()
    engine.dispose()
    if df_hist.empty:
        logger.info("No historical data, skipping drift detection")
        ti.xcom_push(key='drift_detected', value=False)
        ti.xcom_push(key='drift_details', value="no_history")
        return
    drift_found = False
    drift_details = []
    for col in ["price", "house_size", "bed", "bath"]:
        if col not in df_new.columns or col not in df_hist.columns:
            continue
        new_mean = df_new[col].dropna().mean()
        hist_mean = df_hist[col].dropna().mean()
        if hist_mean == 0:
            continue
        pct_change = abs(new_mean - hist_mean) / abs(hist_mean)
        if pct_change > DRIFT_THRESHOLD:
            drift_found = True
            drift_details.append(f"{col}: {pct_change:.2%} change")
    engine2 = create_engine(POSTGRES_RAW_CONN)
    new_cities = set(df_new["city"].dropna().unique()) if "city" in df_new.columns else set()
    try:
        with engine2.connect() as conn:
            hist_cities = set(row[0] for row in conn.execute(text("SELECT DISTINCT city FROM raw_properties")).fetchall())
    except Exception:
        hist_cities = set()
    engine2.dispose()
    new_cats = new_cities - hist_cities
    if len(new_cats) > 5:
        drift_found = True
        drift_details.append(f"{len(new_cats)} new cities detected")
    logger.info(f"Drift detected: {drift_found}, details: {drift_details}")
    ti.xcom_push(key='drift_detected', value=drift_found)
    ti.xcom_push(key='drift_details', value=str(drift_details))

def decide_training(**context):
    ti = context['ti']
    inserted = ti.xcom_pull(key='inserted_count', task_ids='store_raw')
    drift = ti.xcom_pull(key='drift_detected', task_ids='detect_drift')
    quality_valid = ti.xcom_pull(key='quality_valid', task_ids='validate_quality')
    engine = create_engine(POSTGRES_RAW_CONN)
    with engine.connect() as conn:
        try:
            total = conn.execute(text("SELECT COUNT(*) FROM raw_properties")).fetchone()[0]
        except Exception:
            total = 0
    engine.dispose()
    reasons = []
    should_train = False
    if total < MIN_RECORDS_TO_TRAIN:
        reason = f"Not enough data: {total} records"
        logger.info(reason)
        ti.xcom_push(key='train_decision', value=False)
        ti.xcom_push(key='train_reason', value=reason)
        return 'skip_training'
    if not quality_valid:
        reason = "Data quality issues found"
        logger.info(reason)
        ti.xcom_push(key='train_decision', value=False)
        ti.xcom_push(key='train_reason', value=reason)
        return 'skip_training'
    if drift:
        should_train = True
        reasons.append("drift detected")
    if inserted and inserted > MIN_RECORDS_TO_TRAIN:
        should_train = True
        reasons.append(f"large batch: {inserted} new records")
    if total > 0 and inserted:
        volume_increase = inserted / total
        if volume_increase >= 0.05:
            should_train = True
            reasons.append(f"volume increase: {volume_increase:.1%}")
    if should_train:
        reason = ", ".join(reasons)
        logger.info(f"Decision: TRAIN — {reason}")
        ti.xcom_push(key='train_decision', value=True)
        ti.xcom_push(key='train_reason', value=reason)
        return 'preprocess_data'
    else:
        reason = "No significant changes detected"
        logger.info(f"Decision: SKIP — {reason}")
        ti.xcom_push(key='train_decision', value=False)
        ti.xcom_push(key='train_reason', value=reason)
        return 'skip_training'
