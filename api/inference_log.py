"""Módulo de registro de inferencias en ``inference_log`` (raw_db) — RF8.

Responsabilidades:
- Crear la tabla ``inference_log`` en ``raw_db`` si no existe (RF8.1, RF8.4).
- Construir eventos de inferencia vía ``mlops_core.logging_schema.build_inference_event``
  (RF8.2, RF8.3).
- Persistir cada solicitud de inferencia (exitosa o fallida) en la tabla.

La tabla se crea al importar el módulo o al invocar ``ensure_table()``. El
registro se realiza de forma asíncrona en un hilo separado para no bloquear la
respuesta de la API, pero con un fallback síncrono si el pool de hilos falla.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.types import JSON
from sqlalchemy.engine import Engine

from api.config import POSTGRES_RAW_CONN
from mlops_core.logging_schema import build_inference_event, inference_event_to_dict
from mlops_core.types import InferenceEvent, InferenceStatus

__all__ = [
    "ensure_table",
    "log_inference",
    "log_inference_async",
    "get_engine",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy engine (lazy singleton)
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Devuelve el engine de SQLAlchemy para raw_db (singleton lazy)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            POSTGRES_RAW_CONN,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


# ---------------------------------------------------------------------------
# Definición de la tabla inference_log
# ---------------------------------------------------------------------------

metadata = MetaData()

inference_log_table = Table(
    "inference_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_ts", DateTime(timezone=True), nullable=False),
    Column("input_data", JSON, nullable=True),
    Column("prediction", Float, nullable=True),
    Column("model_version", String(20), nullable=True),
    Column("status", String(10), nullable=False),
    Column("error", Text, nullable=True),
)


def ensure_table(engine: Optional[Engine] = None) -> None:
    """Crea la tabla ``inference_log`` en raw_db si no existe (RF8.1).

    Se invoca al iniciar la API y puede invocarse de forma idempotente.
    """
    eng = engine or get_engine()
    metadata.create_all(eng, tables=[inference_log_table], checkfirst=True)
    logger.info("Tabla inference_log verificada/creada en raw_db.")


# ---------------------------------------------------------------------------
# Registro de inferencias
# ---------------------------------------------------------------------------

# Thread pool para escritura asíncrona (no bloquea la respuesta HTTP)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference_log")


def log_inference(
    *,
    input_data: Optional[Dict[str, Any]] = None,
    prediction: Optional[float] = None,
    model_version: Optional[str] = None,
    status: str = "ok",
    error: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    engine: Optional[Engine] = None,
) -> None:
    """Persiste un evento de inferencia en ``inference_log`` (RF8.2, RF8.3).

    Construye el evento vía ``build_inference_event`` y lo inserta en la tabla.
    Registra tanto solicitudes exitosas como fallidas (RF8.3).

    Args:
        input_data: Features de entrada de la solicitud.
        prediction: Valor de predicción (None si hubo error).
        model_version: Versión del modelo utilizado.
        status: Estado de la solicitud ("ok" o "error").
        error: Mensaje de error (si aplica).
        timestamp: Marca de tiempo (UTC); si None se usa la actual.
        engine: Engine de SQLAlchemy (para testing); si None usa el singleton.
    """
    try:
        # Construir evento usando la lógica pura de mlops_core (RF8.2)
        event = build_inference_event(
            input_data=input_data,
            prediction=prediction,
            model_version=model_version,
            status=status,
            error=error,
            timestamp=timestamp,
        )

        # Persistir en la tabla
        _persist_event(event, engine=engine)

    except Exception as exc:
        # El logging de inferencias no debe interrumpir la respuesta de la API
        logger.error("Error al registrar evento de inferencia: %s", exc)


def log_inference_async(
    *,
    input_data: Optional[Dict[str, Any]] = None,
    prediction: Optional[float] = None,
    model_version: Optional[str] = None,
    status: str = "ok",
    error: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """Registra un evento de inferencia de forma asíncrona (no bloquea).

    Envía la escritura a un thread pool para no añadir latencia a la respuesta.
    Si el envío al pool falla, intenta la escritura síncrona como fallback.
    """
    try:
        _executor.submit(
            log_inference,
            input_data=input_data,
            prediction=prediction,
            model_version=model_version,
            status=status,
            error=error,
            timestamp=timestamp,
        )
    except Exception as exc:
        logger.warning(
            "No se pudo enviar al pool de logging; intentando síncrono: %s", exc
        )
        log_inference(
            input_data=input_data,
            prediction=prediction,
            model_version=model_version,
            status=status,
            error=error,
            timestamp=timestamp,
        )


def _persist_event(event: InferenceEvent, engine: Optional[Engine] = None) -> None:
    """Inserta un InferenceEvent en la tabla inference_log."""
    eng = engine or get_engine()

    insert_stmt = inference_log_table.insert().values(
        request_ts=event.timestamp,
        input_data=event.input_data,
        prediction=event.prediction,
        model_version=event.model_version,
        status=event.status.value,
        error=event.error,
    )

    with eng.connect() as conn:
        conn.execute(insert_stmt)
        conn.commit()
