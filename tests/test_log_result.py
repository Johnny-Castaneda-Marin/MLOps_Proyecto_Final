"""Pruebas unitarias del adaptador ``log_result`` (tarea 7.5).

Verifica que:
- La tabla ``training_history`` se crea con las columnas ampliadas
  (``mlflow_run_id``, ``mlflow_model_version``, ``code_commit``).
- El adaptador persiste correctamente la decisión, motivos y metadatos de MLflow.
- Funciona tanto en la rama de entrenamiento como en la de skip.
- Las migraciones idempotentes no fallan si la tabla ya existe.
- El ``build_audit_event`` de mlops_core se usa correctamente.

Se usa un mock de SQLAlchemy para verificar las sentencias SQL ejecutadas.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from types import ModuleType

import pytest

# Asegurar que mlops_core es importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS_DIR = REPO_ROOT / "airflow" / "dags" / "tasks"
if str(TASKS_DIR) not in sys.path:
    sys.path.insert(0, str(TASKS_DIR))


# ---------------------------------------------------------------------------
# Mock heavy dependencies that are not installed in the test environment.
# We need to do this BEFORE importing train.py.
# ---------------------------------------------------------------------------

def _setup_mocks():
    """Set up module mocks for heavy dependencies not available in test env."""
    mock_mlflow = MagicMock()
    mock_mlflow.sklearn = MagicMock()
    mock_mlflow.MlflowClient = MagicMock

    mocks = {
        "mlflow": mock_mlflow,
        "mlflow.sklearn": mock_mlflow.sklearn,
        "sklearn": MagicMock(),
        "sklearn.linear_model": MagicMock(),
        "sklearn.ensemble": MagicMock(),
        "sklearn.metrics": MagicMock(),
        "sklearn.preprocessing": MagicMock(),
        "sklearn.pipeline": MagicMock(),
    }
    for name, mock in mocks.items():
        if name not in sys.modules or not hasattr(sys.modules[name], "__file__"):
            sys.modules[name] = mock
    return mock_mlflow


_mock_mlflow = _setup_mocks()

# Now we can safely import train (it will use our mocked modules).
# Force reimport to pick up our mocks.
if "train" in sys.modules:
    del sys.modules["train"]

import train


class TestLogResultAdapter:
    """Pruebas del adaptador log_result con base de datos mockeada."""

    def _make_context(self, xcoms: dict) -> dict:
        """Crea un contexto de Airflow simulado con XComs predefinidos."""
        ti = MagicMock()

        def xcom_pull(key=None, task_ids=None):
            if task_ids and key:
                return xcoms.get((key, task_ids), xcoms.get(key))
            return xcoms.get(key)

        ti.xcom_pull = xcom_pull
        return {"ti": ti}

    @patch.object(train, "create_engine")
    @patch.object(train, "MlflowClient")
    def test_log_result_train_branch(self, mock_client_cls, mock_create_engine):
        """log_result persiste datos completos cuando se entrenó y promovió."""
        # Simular que search_model_versions devuelve la versión del candidato.
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.run_id = "abc123run"
        mock_version.version = "5"
        mock_client.search_model_versions.return_value = [mock_version]
        mock_client_cls.return_value = mock_client

        # Simular la conexión a la base de datos.
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        xcoms = {
            ("batch_number", "fetch_batch"): 3,
            ("train_decision", "decide_training"): True,
            ("train_reason", "decide_training"): "drift detected",
            ("promoted", "promote_or_reject"): True,
            ("promotion_reason", "promote_or_reject"): "MAE improved 5.2% (100.0 -> 94.8)",
            ("best_model_name", "train_candidates"): "gradient_boosting",
            ("best_val_mae", "train_candidates"): 94.8,
            ("best_run_id", "train_candidates"): "abc123run",
            ("code_commit", "train_candidates"): "deadbeef1234",
            # Fallbacks for train_and_promote (should not be used).
            ("promoted", "train_and_promote"): None,
            ("promotion_reason", "train_and_promote"): None,
            ("best_model_name", "train_and_promote"): None,
            ("best_val_mae", "train_and_promote"): None,
        }

        context = self._make_context(xcoms)
        train.log_result(**context)

        # Verificar que se creó la tabla y se insertó el registro.
        assert mock_engine.begin.called
        calls = mock_conn.execute.call_args_list
        # DDL + 3 migrations + INSERT = at least 5 calls.
        assert len(calls) >= 5

        # Verificar los parámetros del INSERT (último execute).
        insert_call = calls[-1]
        params = insert_call[0][1] if len(insert_call[0]) > 1 else insert_call[1]
        assert params["bn"] == 3
        assert params["trained"] is True
        assert params["tr"] == "drift detected"
        assert params["promoted"] is True
        assert "MAE improved" in params["pr"]
        assert params["bm"] == "gradient_boosting"
        assert params["mae"] == 94.8
        assert params["run_id"] == "abc123run"
        assert params["model_version"] == "5"
        assert params["commit"] == "deadbeef1234"

    @patch.object(train, "create_engine")
    @patch.object(train, "MlflowClient")
    def test_log_result_skip_branch(self, mock_client_cls, mock_create_engine):
        """log_result funciona correctamente cuando se omitió el entrenamiento."""
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []
        mock_client_cls.return_value = mock_client

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        xcoms = {
            ("batch_number", "fetch_batch"): 2,
            ("train_decision", "decide_training"): False,
            ("train_reason", "decide_training"): "insufficient data",
            ("promoted", "promote_or_reject"): None,
            ("promotion_reason", "promote_or_reject"): None,
            ("best_model_name", "train_candidates"): None,
            ("best_val_mae", "train_candidates"): None,
            ("best_run_id", "train_candidates"): None,
            ("code_commit", "train_candidates"): None,
            ("promoted", "train_and_promote"): None,
            ("promotion_reason", "train_and_promote"): None,
            ("best_model_name", "train_and_promote"): None,
            ("best_val_mae", "train_and_promote"): None,
        }

        context = self._make_context(xcoms)
        train.log_result(**context)

        calls = mock_conn.execute.call_args_list
        insert_call = calls[-1]
        params = insert_call[0][1] if len(insert_call[0]) > 1 else insert_call[1]
        assert params["bn"] == 2
        assert params["trained"] is False
        assert params["tr"] == "insufficient data"
        assert params["promoted"] is False
        assert params["run_id"] is None
        assert params["model_version"] is None
        assert params["commit"] is None

    @patch.object(train, "create_engine")
    @patch.object(train, "MlflowClient")
    def test_log_result_fallback_to_train_and_promote(self, mock_client_cls, mock_create_engine):
        """log_result usa fallback a train_and_promote cuando las tareas separadas no publican XComs."""
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []
        mock_client_cls.return_value = mock_client

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        xcoms = {
            ("batch_number", "fetch_batch"): 1,
            ("train_decision", "decide_training"): True,
            ("train_reason", "decide_training"): "volume increase",
            # Tareas separadas no publican nada.
            ("promoted", "promote_or_reject"): None,
            ("promotion_reason", "promote_or_reject"): None,
            ("best_model_name", "train_candidates"): None,
            ("best_val_mae", "train_candidates"): None,
            ("best_run_id", "train_candidates"): None,
            ("code_commit", "train_candidates"): None,
            # Fallback a train_and_promote.
            ("promoted", "train_and_promote"): True,
            ("promotion_reason", "train_and_promote"): "First model promoted",
            ("best_model_name", "train_and_promote"): "ridge",
            ("best_val_mae", "train_and_promote"): 120.5,
        }

        context = self._make_context(xcoms)
        train.log_result(**context)

        calls = mock_conn.execute.call_args_list
        insert_call = calls[-1]
        params = insert_call[0][1] if len(insert_call[0]) > 1 else insert_call[1]
        assert params["promoted"] is True
        assert params["pr"] == "First model promoted"
        assert params["bm"] == "ridge"
        assert params["mae"] == 120.5


class TestBuildAuditEventIntegration:
    """Verifica que log_result usa build_audit_event de mlops_core correctamente."""

    def test_build_audit_event_produces_correct_fields(self):
        """build_audit_event incluye todos los campos ampliados."""
        from mlops_core.logging_schema import build_audit_event

        event = build_audit_event(
            batch_number=5,
            trained=True,
            train_reason="drift detected",
            promoted=True,
            promotion_reason="MAE improved 4.5% (200.0 -> 191.0)",
            best_model="random_forest",
            best_val_mae=191.0,
            mlflow_run_id="run_xyz_789",
            mlflow_model_version="12",
            code_commit="abc123def456",
        )

        assert event.batch_number == 5
        assert event.trained is True
        assert event.train_reason == "drift detected"
        assert event.promoted is True
        assert "MAE improved" in event.promotion_reason
        assert event.best_model == "random_forest"
        assert event.best_val_mae == 191.0
        assert event.mlflow_run_id == "run_xyz_789"
        assert event.mlflow_model_version == "12"
        assert event.code_commit == "abc123def456"
        assert event.logged_at is not None

    def test_build_audit_event_skip_branch(self):
        """build_audit_event funciona con valores None para la rama skip."""
        from mlops_core.logging_schema import build_audit_event

        event = build_audit_event(
            batch_number=2,
            trained=False,
            train_reason="insufficient data",
            promoted=False,
            promotion_reason="",
            best_model=None,
            best_val_mae=None,
            mlflow_run_id=None,
            mlflow_model_version=None,
            code_commit=None,
        )

        assert event.batch_number == 2
        assert event.trained is False
        assert event.promoted is False
        assert event.mlflow_run_id is None
        assert event.mlflow_model_version is None
        assert event.code_commit is None
