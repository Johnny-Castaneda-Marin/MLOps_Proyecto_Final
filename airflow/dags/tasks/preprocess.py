import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import POSTGRES_RAW_CONN, POSTGRES_CLEAN_CONN

logger = logging.getLogger(__name__)

def preprocess_data(**context):
    engine_raw = create_engine(POSTGRES_RAW_CONN)
    engine_clean = create_engine(POSTGRES_CLEAN_CONN)

    df = pd.read_sql("SELECT * FROM raw_properties WHERE processed=FALSE", engine_raw)
    logger.info(f"Processing {len(df)} unprocessed records")

    if df.empty:
        logger.info("No new records to process")
        engine_raw.dispose()
        engine_clean.dispose()
        return

    # Drop unnecessary columns
    df = df.drop(columns=["id", "row_hash", "loaded_at", "processed", "day_used"], errors="ignore")

    # Parse date
    df["prev_sold_date"] = pd.to_datetime(df["prev_sold_date"], errors="coerce")
    df["prev_sold_year"] = df["prev_sold_date"].dt.year.fillna(0).astype(int)
    df = df.drop(columns=["prev_sold_date"], errors="ignore")

    # Drop rows without target
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Fill numeric nulls with median
    num_cols = ["bed", "bath", "acre_lot", "house_size", "zip_code", "brokered_by", "street"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    for col in ["status", "city", "state"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype("category").cat.codes

    # Cap outliers on price (remove top 1%)
    p99 = df["price"].quantile(0.99)
    df = df[df["price"] <= p99]

    # Store to clean_db
    with engine_clean.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clean_properties (
                id SERIAL PRIMARY KEY,
                batch_number INTEGER,
                brokered_by FLOAT,
                status INTEGER,
                price FLOAT,
                bed FLOAT,
                bath FLOAT,
                acre_lot FLOAT,
                street FLOAT,
                city INTEGER,
                state INTEGER,
                zip_code FLOAT,
                house_size FLOAT,
                prev_sold_year INTEGER,
                split VARCHAR(10)
            )
        """))
        conn.commit()

    # Train/val/test split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    df["split"] = "train"
    df.iloc[int(n*0.7):int(n*0.85), df.columns.get_loc("split")] = "val"
    df.iloc[int(n*0.85):, df.columns.get_loc("split")] = "test"

    df.to_sql("clean_properties", engine_clean, if_exists="append", index=False)
    logger.info(f"Stored {len(df)} clean records")

    # Mark raw as processed
    batch_number = df["batch_number"].iloc[0] if "batch_number" in df.columns else None
    with engine_raw.connect() as conn:
        conn.execute(text("UPDATE raw_properties SET processed=TRUE WHERE processed=FALSE"))
        conn.commit()

    engine_raw.dispose()
    engine_clean.dispose()
    context['ti'].xcom_push(key='clean_count', value=len(df))
