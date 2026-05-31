import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

from config import POSTGRES_RAW_CONN, POSTGRES_CLEAN_CONN


# --- Import shim ---------------------------------------------------------
# El DAG sólo añade ``tasks/`` a ``sys.path`` (no la raíz del repositorio),
# por lo que ``mlops_core`` no es importable por defecto. Subimos por los
# directorios padre desde este archivo hasta encontrar el que contiene
# ``mlops_core`` y lo insertamos en ``sys.path``. Esto resuelve la importación
# tanto en el repositorio como dentro del contenedor de Airflow.
def _ensure_mlops_core_on_path() -> None:
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(path, "mlops_core")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent


_ensure_mlops_core_on_path()

from mlops_core.features import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DEFAULT_RANDOM_STATE,
    fit_categorical_encoders,
    split_dataset,
    transform_categorical,
)

logger = logging.getLogger(__name__)

# Columnas de metadatos/control de RAW_DATA que no forman parte de la versión
# procesada. ``batch_number`` se preserva de forma explícita para mantener la
# trazabilidad de extremo a extremo (RF2.5).
_DROP_COLUMNS = ("id", "row_hash", "loaded_at", "processed", "day_used")

# Variables numéricas imputadas por mediana en el preprocesamiento.
_NUMERIC_COLUMNS = ("bed", "bath", "acre_lot", "house_size", "zip_code", "brokered_by", "street")


def preprocess_data(**context):
    engine_raw = create_engine(POSTGRES_RAW_CONN)
    engine_clean = create_engine(POSTGRES_CLEAN_CONN)

    try:
        df = pd.read_sql("SELECT * FROM raw_properties WHERE processed=FALSE", engine_raw)
        logger.info(f"Processing {len(df)} unprocessed records")

        if df.empty:
            logger.info("No new records to process")
            context['ti'].xcom_push(key='clean_count', value=0)
            return

        # --- Limpieza de datos (se conservan los pasos razonables) -----------
        # Se elimina metadata de control pero se preserva ``batch_number`` (RF2.5).
        df = df.drop(columns=list(_DROP_COLUMNS), errors="ignore")

        df["prev_sold_date"] = pd.to_datetime(df["prev_sold_date"], errors="coerce")
        df["prev_sold_year"] = df["prev_sold_date"].dt.year.fillna(0).astype(int)
        df = df.drop(columns=["prev_sold_date"], errors="ignore")

        # Descartar precios nulos o no positivos (objetivo de regresión inválido).
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0]

        # Imputación numérica por mediana.
        for col in _NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Recorte de precios extremos al percentil 99 (reducción de outliers).
        p99 = df["price"].quantile(0.99)
        df = df[df["price"] <= p99]

        if df.empty:
            logger.info("No records left after cleaning")
            context['ti'].xcom_push(key='clean_count', value=0)
            return

        # --- Codificación categórica con manejo robusto de desconocidos ------
        # Se delega a ``mlops_core.features``: las categorías no vistas o nulas
        # se mapean a un código reservado de "otros" sin lanzar excepción (RF3.5),
        # en lugar de los antiguos ``category codes`` de pandas.
        encoders = fit_categorical_encoders(df, CATEGORICAL_COLUMNS)
        df = transform_categorical(df, encoders)

        # --- Particionado determinista train/val/test ------------------------
        # Se delega a ``mlops_core.features.split_dataset`` con ``random_state=42``
        # para garantizar reproducibilidad (RF11.3). Conserva ``batch_number``.
        df = split_dataset(df, random_state=DEFAULT_RANDOM_STATE)

        # --- Persistencia en CLEAN_DATA --------------------------------------
        with engine_clean.begin() as conn:
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

        df.to_sql("clean_properties", engine_clean, if_exists="append", index=False,
                  method="multi", chunksize=1000)
        logger.info(f"Stored {len(df)} clean records")

        # Marcar los registros de origen como procesados (RF2.4).
        with engine_raw.begin() as conn:
            conn.execute(text("UPDATE raw_properties SET processed=TRUE WHERE processed=FALSE"))

        context['ti'].xcom_push(key='clean_count', value=len(df))
    finally:
        engine_raw.dispose()
        engine_clean.dispose()
