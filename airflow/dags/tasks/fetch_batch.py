import requests
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from config import DATA_API_URL, GROUP_NUMBER, POSTGRES_RAW_CONN, DAYS

logger = logging.getLogger(__name__)

def fetch_batch(**context):
    engine = create_engine(POSTGRES_RAW_CONN)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS batch_control (
                id SERIAL PRIMARY KEY,
                batch_number INTEGER,
                day_used VARCHAR(20),
                records_count INTEGER,
                fetched_at TIMESTAMP,
                status VARCHAR(20) DEFAULT 'fetched'
            )
        """))
        result = conn.execute(text("SELECT COUNT(*) FROM batch_control")).fetchone()
        batch_index = result[0]

    if batch_index >= len(DAYS):
        logger.info("No more days available")
        context['ti'].xcom_push(key='data_exhausted', value=True)
        engine.dispose()
        return

    day = DAYS[batch_index]
    logger.info(f"Fetching batch for day: {day}")

    try:
        response = requests.get(
            f"{DATA_API_URL}/data",
            params={"group_number": GROUP_NUMBER, "day": day},
            timeout=120
        )
        if response.status_code == 400:
            detail = response.json().get("detail", "")
            logger.info(f"API exhausted: {detail}")
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO batch_control (batch_number, day_used, records_count, fetched_at, status)
                    VALUES (:bn, :day, 0, :ts, 'exhausted')
                """), {"bn": -1, "day": day, "ts": datetime.utcnow()})
            context['ti'].xcom_push(key='data_exhausted', value=True)
            engine.dispose()
            return

        response.raise_for_status()
        data = response.json()
        batch_number = data["batch_number"]
        records = data["data"]
        logger.info(f"Received batch {batch_number} with {len(records)} records")

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO batch_control (batch_number, day_used, records_count, fetched_at)
                VALUES (:bn, :day, :cnt, :ts)
            """), {"bn": batch_number, "day": day, "cnt": len(records), "ts": datetime.utcnow()})

        context['ti'].xcom_push(key='batch_number', value=batch_number)
        context['ti'].xcom_push(key='records', value=records)
        context['ti'].xcom_push(key='day', value=day)
        context['ti'].xcom_push(key='data_exhausted', value=False)

    except Exception as e:
        logger.error(f"Error fetching batch: {e}")
        engine.dispose()
        raise

    engine.dispose()
