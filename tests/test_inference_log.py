"""Tests unitarios para el módulo api.inference_log (RF8.1, RF8.2, RF8.3, RF8.4).

Verifica:
- Creación de la tabla inference_log (ensure_table).
- Registro de inferencias exitosas y fallidas (log_inference).
- Construcción de eventos vía build_inference_event.
- Persistencia incluyendo errores.

Usa SQLite en memoria como sustituto de PostgreSQL para pruebas unitarias.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect, select, text

from api.inference_log import (
    _persist_event,
    ensure_table,
    inference_log_table,
    log_inference,
    metadata,
)
from mlops_core.logging_schema import build_inference_event
from mlops_core.types import InferenceEvent, InferenceStatus


@pytest.fixture
def sqlite_engine():
    """Crea un engine SQLite en memoria para pruebas."""
    engine = create_engine("sqlite:///:memory:")
    # Crear la tabla en el engine de prueba
    metadata.create_all(engine, tables=[inference_log_table])
    return engine


class TestEnsureTable:
    """Verifica la creación idempotente de la tabla inference_log."""

    def test_creates_table(self, sqlite_engine):
        """La tabla inference_log se crea correctamente."""
        inspector = inspect(sqlite_engine)
        assert "inference_log" in inspector.get_table_names()

    def test_idempotent(self, sqlite_engine):
        """Llamar ensure_table múltiples veces no falla."""
        # La tabla ya existe por el fixture; llamar de nuevo no debe fallar
        ensure_table(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "inference_log" in inspector.get_table_names()

    def test_table_columns(self, sqlite_engine):
        """La tabla tiene las columnas esperadas según el diseño."""
        inspector = inspect(sqlite_engine)
        columns = {col["name"] for col in inspector.get_columns("inference_log")}
        expected = {"id", "request_ts", "input_data", "prediction", "model_version", "status", "error"}
        assert expected == columns


class TestLogInference:
    """Verifica el registro de inferencias exitosas y fallidas."""

    def test_log_successful_inference(self, sqlite_engine):
        """Una inferencia exitosa se persiste con status='ok' y prediction no nula."""
        input_data = {"bed": 3, "bath": 2, "city": "Austin", "state": "Texas"}
        log_inference(
            input_data=input_data,
            prediction=350000.0,
            model_version="5",
            status="ok",
            engine=sqlite_engine,
        )

        with sqlite_engine.connect() as conn:
            rows = conn.execute(select(inference_log_table)).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row.status == "ok"
        assert row.prediction == 350000.0
        assert row.model_version == "5"
        assert row.error is None
        assert row.request_ts is not None

    def test_log_failed_inference(self, sqlite_engine):
        """Una inferencia fallida se persiste con status='error' y error no vacío."""
        input_data = {"bed": 3, "bath": 2}
        log_inference(
            input_data=input_data,
            prediction=None,
            model_version=None,
            status="error",
            error="No hay modelo cargado",
            engine=sqlite_engine,
        )

        with sqlite_engine.connect() as conn:
            rows = conn.execute(select(inference_log_table)).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row.status == "error"
        assert row.prediction is None
        assert row.error == "No hay modelo cargado"

    def test_log_multiple_inferences(self, sqlite_engine):
        """Se pueden registrar múltiples inferencias consecutivas."""
        for i in range(5):
            log_inference(
                input_data={"index": i},
                prediction=float(i * 100000),
                model_version="3",
                status="ok",
                engine=sqlite_engine,
            )

        with sqlite_engine.connect() as conn:
            rows = conn.execute(select(inference_log_table)).fetchall()

        assert len(rows) == 5

    def test_log_inference_with_explicit_timestamp(self, sqlite_engine):
        """Se puede proporcionar un timestamp explícito."""
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        log_inference(
            input_data={"bed": 2},
            prediction=200000.0,
            model_version="1",
            status="ok",
            timestamp=ts,
            engine=sqlite_engine,
        )

        with sqlite_engine.connect() as conn:
            rows = conn.execute(select(inference_log_table)).fetchall()

        assert len(rows) == 1
        # El timestamp se persiste (SQLite almacena como string)
        assert rows[0].request_ts is not None

    def test_log_inference_does_not_raise_on_db_error(self):
        """Si la DB falla, log_inference no lanza excepción (no interrumpe la API)."""
        # Usar un engine inválido que fallará al conectar
        bad_engine = create_engine("sqlite:///nonexistent_dir/bad.db")
        # No debe lanzar excepción
        log_inference(
            input_data={"bed": 1},
            prediction=100000.0,
            model_version="1",
            status="ok",
            engine=bad_engine,
        )


class TestPersistEvent:
    """Verifica la persistencia directa de InferenceEvent."""

    def test_persist_ok_event(self, sqlite_engine):
        """Un evento con status OK se persiste correctamente."""
        event = build_inference_event(
            input_data={"house_size": 1800.0, "city": "Dallas"},
            prediction=425000.0,
            model_version="7",
            status="ok",
        )
        _persist_event(event, engine=sqlite_engine)

        with sqlite_engine.connect() as conn:
            rows = conn.execute(select(inference_log_table)).fetchall()

        assert len(rows) == 1
        assert rows[0].status == "ok"
        assert rows[0].prediction == 425000.0
        assert rows[0].model_version == "7"

    def test_persist_error_event(self, sqlite_engine):
        """Un evento con status error se persiste con el mensaje de error."""
        event = build_inference_event(
            input_data={"house_size": 1200.0},
            prediction=None,
            model_version="7",
            status="error",
            error="Feature mismatch",
        )
        _persist_event(event, engine=sqlite_engine)

        with sqlite_engine.connect() as conn:
            rows = conn.execute(select(inference_log_table)).fetchall()

        assert len(rows) == 1
        assert rows[0].status == "error"
        assert rows[0].error == "Feature mismatch"
        assert rows[0].prediction is None
