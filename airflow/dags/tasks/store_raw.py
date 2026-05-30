import logging
import hashlib
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from config import POSTGRES_RAW_CONN

logger = logging.getLogger(__name__)

def store_raw_batch(**context):
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')
    day = ti.xcom_pull(key='day', task_ids='fetch_batch')

    if not records:
        logger.info("No records to store")
        return

    engine = create_engine(POSTGRES_RAW_CONN)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_properties (
                id SERIAL PRIMARY KEY,
                batch_number INTEGER,
                day_used VARCHAR(20),
                brokered_by FLOAT,
                status VARCHAR(50),
                price FLOAT,
                bed FLOAT,
                bath FLOAT,
                acre_lot FLOAT,
                street FLOAT,
                city VARCHAR(100),
                state VARCHAR(100),
                zip_code FLOAT,
                house_size FLOAT,
                prev_sold_date VARCHAR(20),
                row_hash VARCHAR(64),
                loaded_at TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        """))

    # Bulk insert con pandas
    df = pd.DataFrame(records)
    df["batch_number"] = batch_number
    df["day_used"] = day
    df["loaded_at"] = datetime.utcnow()
    df["processed"] = False
    df["prev_sold_date"] = df["prev_sold_date"].astype(str)
    df["row_hash"] = df.apply(
        lambda r: hashlib.md5(json.dumps(r[list(records[0].keys())].to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
        axis=1
    )

    # Eliminar duplicados ya existentes
    with engine.connect() as conn:
        existing = pd.read_sql("SELECT row_hash FROM raw_properties", conn)
    df = df[~df["row_hash"].isin(existing["row_hash"])]

    inserted = len(df)
    if inserted > 0:
        df.to_sql("raw_properties", engine, if_exists="append", index=False,
                  method="multi", chunksize=1000)

    logger.info(f"Stored {inserted} records via bulk insert")
    context['ti'].xcom_push(key='inserted_count', value=inserted)
    engine.dispose()
