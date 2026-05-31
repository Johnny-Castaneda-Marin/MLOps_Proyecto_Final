import logging
import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from config import POSTGRES_RAW_CONN


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

from mlops_core.ingest import deduplicate  # noqa: E402

logger = logging.getLogger(__name__)

# Columnas de metadatos que añade el adaptador (no forman parte del registro
# crudo recibido de la Data_API). El resto de columnas se conservan sin
# modificación para cumplir RF2.1 (preservación de datos crudos).
_METADATA_COLUMNS = ("batch_number", "day_used", "loaded_at", "processed", "row_hash")


def store_raw_batch(**context):
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')
    day = ti.xcom_pull(key='day', task_ids='fetch_batch')

    if not records:
        logger.info("No records to store")
        context['ti'].xcom_push(key='inserted_count', value=0)
        return

    engine = create_engine(POSTGRES_RAW_CONN)
    try:
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

        # Hashes ya presentes en la tabla (clave de deduplicación, RF1.4).
        with engine.connect() as conn:
            existing = pd.read_sql("SELECT row_hash FROM raw_properties", conn)
        existing_hashes = set(existing["row_hash"].dropna().tolist())

        # Delegar la deduplicación a la lógica pura de mlops_core: excluye
        # registros cuyo row_hash ya existe y duplicados dentro del propio lote.
        new_records, new_hashes = deduplicate(records, existing_hashes)

        inserted = len(new_records)
        if inserted == 0:
            logger.info("No new records to store after deduplication")
            context['ti'].xcom_push(key='inserted_count', value=0)
            return

        # Conservar los campos originales sin modificación (RF2.1); sólo se
        # añaden columnas de metadatos.
        df = pd.DataFrame(new_records)
        df["row_hash"] = new_hashes
        df["batch_number"] = batch_number
        df["day_used"] = day
        df["loaded_at"] = datetime.utcnow()
        df["processed"] = False

        df.to_sql(
            "raw_properties",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

        logger.info(f"Stored {inserted} records via bulk insert")
        context['ti'].xcom_push(key='inserted_count', value=inserted)
    finally:
        engine.dispose()
