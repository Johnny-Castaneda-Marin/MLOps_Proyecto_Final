"""Adaptador de Airflow para la ingesta incremental de lotes (RF1, RF12).

Este adaptador conserva la responsabilidad de I/O (lectura del conteo en
``batch_control``, llamada a la ``Data_API`` con ``timeout=120``, persistencia
de metadatos y publicación de XComs) y delega las decisiones de negocio
—selección de día por índice acumulado y detección de agotamiento— a las
funciones puras de :mod:`mlops_core.ingest` (``select_day`` / ``is_exhausted``).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import requests
from sqlalchemy import create_engine, text

from config import DATA_API_URL, GROUP_NUMBER, POSTGRES_RAW_CONN, DAYS


def _ensure_mlops_core_importable() -> None:
    """Garantiza que el paquete ``mlops_core`` sea importable.

    El DAG inserta ``dags/tasks`` en ``sys.path`` (no la raíz del repositorio),
    por lo que ``import mlops_core`` puede fallar al ejecutarse desde Airflow.
    Esta función:

    1. Intenta importar ``mlops_core`` directamente (caso en que el paquete está
       instalado en el contenedor o ya disponible en ``sys.path``).
    2. Si falla, asciende por los directorios ancestros del archivo hasta hallar
       el que contiene ``mlops_core/__init__.py`` (la raíz del repositorio) y lo
       inserta en ``sys.path``.

    Es idempotente y no altera los imports ``from config import ...`` existentes.
    """
    try:
        import mlops_core  # noqa: F401
        return
    except ImportError:
        pass

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "mlops_core" / "__init__.py").exists():
            ancestor_str = str(ancestor)
            if ancestor_str not in sys.path:
                sys.path.insert(0, ancestor_str)
            break


_ensure_mlops_core_importable()

from mlops_core.ingest import select_day, is_exhausted  # noqa: E402

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

    if is_exhausted(batch_index, None, days=DAYS):
        logger.info("No more days available (batch_index=%s)", batch_index)
        context['ti'].xcom_push(key='data_exhausted', value=True)
        engine.dispose()
        return

    day = select_day(batch_index, days=DAYS)
    logger.info(f"Fetching batch for day: {day}")

    try:
        response = requests.get(
            f"{DATA_API_URL}/data",
            params={"group_number": GROUP_NUMBER, "day": day},
            timeout=120
        )

        # Agotamiento señalado por la Data_API (HTTP 400 de fin de datos): se
        # registra el metadato con estado 'exhausted' y se finaliza sin error
        # ni inserción de registros (RF1.6, RF12.1).
        if is_exhausted(batch_index, response.status_code, days=DAYS):
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                detail = response.text
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

        # Persistencia de metadatos de ejecución en UTC.
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO batch_control (batch_number, day_used, records_count, fetched_at, status)
                VALUES (:bn, :day, :cnt, :ts, 'fetched')
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
