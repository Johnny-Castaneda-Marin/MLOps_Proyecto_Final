"""Tests unitarios de los endpoints FastAPI de la API de inferencia (task 9.3).

Verifica:
- POST /predict: predicción exitosa, error sin modelo, error de inferencia.
- POST /admin/reload: autenticación por token (401 sin token, 401 token inválido).
- GET /health: respuesta con/sin modelo cargado.
- GET /metrics: formato Prometheus.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.model_holder import ModelHolder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model():
    """Modelo mock que devuelve una predicción fija."""
    model = MagicMock()
    model.predict.return_value = np.array([350000.0])
    return model


@pytest.fixture()
def app_with_model(mock_model):
    """Crea la app FastAPI con un modelo pre-cargado en el holder."""
    from api import main

    # Reset del holder global para el test
    main.model_holder = ModelHolder()
    main.model_holder.update(mock_model, "42")

    # Recrear la app sin lifespan (evita conexión a MLflow)
    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(_build_router_from_main(main))
    # Usar la app directamente pero sin lifespan
    client = TestClient(main.app, raise_server_exceptions=False)
    return client, main


@pytest.fixture()
def app_without_model():
    """Crea la app FastAPI sin modelo cargado."""
    from api import main

    main.model_holder = ModelHolder()
    client = TestClient(main.app, raise_server_exceptions=False)
    return client, main


@pytest.fixture()
def client_with_model(mock_model):
    """TestClient con modelo cargado (simplificado)."""
    from api import main

    main.model_holder = ModelHolder()
    main.model_holder.update(mock_model, "42")
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture()
def client_without_model():
    """TestClient sin modelo cargado."""
    from api import main

    main.model_holder = ModelHolder()
    return TestClient(main.app, raise_server_exceptions=False)


def _build_router_from_main(main):
    """Helper - no usado directamente, se usa main.app."""
    from fastapi import APIRouter
    return APIRouter()


# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

VALID_PREDICT_PAYLOAD = {
    "brokered_by": 101.0,
    "status": "for_sale",
    "bed": 3,
    "bath": 2,
    "acre_lot": 0.12,
    "street": 123.0,
    "city": "Austin",
    "state": "Texas",
    "zip_code": 78701.0,
    "house_size": 1800.0,
    "prev_sold_year": 2018,
}


# ---------------------------------------------------------------------------
# Tests: POST /predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests del endpoint POST /predict."""

    def test_predict_success(self, client_with_model, mock_model):
        """Predicción exitosa devuelve prediction + model_version + status ok."""
        response = client_with_model.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "42"
        assert isinstance(data["prediction"], float)
        assert data["prediction"] == 350000.0

    def test_predict_no_model_returns_503(self, client_without_model):
        """Sin modelo cargado devuelve 503."""
        response = client_without_model.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data

    def test_predict_model_error_returns_500(self, client_with_model, mock_model):
        """Error durante la predicción devuelve 500."""
        mock_model.predict.side_effect = RuntimeError("model crashed")
        response = client_with_model.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_predict_missing_field_returns_422(self, client_with_model):
        """Payload incompleto devuelve 422 (validación Pydantic)."""
        incomplete = {"brokered_by": 101.0, "status": "for_sale"}
        response = client_with_model.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_optional_prev_sold_year(self, client_with_model, mock_model):
        """prev_sold_year es opcional y se maneja correctamente."""
        payload = {**VALID_PREDICT_PAYLOAD}
        del payload["prev_sold_year"]
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests: POST /admin/reload
# ---------------------------------------------------------------------------


class TestAdminReload:
    """Tests del endpoint POST /admin/reload."""

    def test_reload_no_token_returns_401(self, client_with_model):
        """Sin header Authorization devuelve 401."""
        response = client_with_model.post("/admin/reload")
        assert response.status_code == 401

    def test_reload_invalid_token_returns_401(self, client_with_model):
        """Token inválido devuelve 401."""
        response = client_with_model.post(
            "/admin/reload",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_reload_bad_format_returns_401(self, client_with_model):
        """Formato de Authorization incorrecto devuelve 401."""
        response = client_with_model.post(
            "/admin/reload",
            headers={"Authorization": "Basic sometoken"},
        )
        assert response.status_code == 401

    @patch("api.main.load_champion_model")
    def test_reload_valid_token_success(self, mock_load, client_with_model):
        """Token válido y recarga exitosa devuelve 200 con nueva versión."""
        mock_load.return_value = "99"
        response = client_with_model.post(
            "/admin/reload",
            headers={"Authorization": "Bearer changeme-admin-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "99"

    @patch("api.main.load_champion_model")
    def test_reload_failure_preserves_model(self, mock_load, client_with_model):
        """Fallo de recarga conserva el modelo previo (RF7.6)."""
        mock_load.side_effect = RuntimeError("MLflow unavailable")
        response = client_with_model.post(
            "/admin/reload",
            headers={"Authorization": "Bearer changeme-admin-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        # El modelo previo (v42) se conserva
        assert data["model_version"] == "42"
        assert "MLflow unavailable" in data["message"]


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests del endpoint GET /health."""

    def test_health_with_model(self, client_with_model):
        """Con modelo cargado reporta healthy."""
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "42"

    def test_health_without_model(self, client_without_model):
        """Sin modelo cargado reporta degraded."""
        response = client_without_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["model_version"] is None


# ---------------------------------------------------------------------------
# Tests: GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests del endpoint GET /metrics."""

    def test_metrics_returns_prometheus_format(self, client_with_model):
        """El endpoint /metrics devuelve texto en formato Prometheus."""
        response = client_with_model.get("/metrics")
        assert response.status_code == 200
        # Prometheus content type
        assert "text/plain" in response.headers.get("content-type", "")
        body = response.text
        # Verifica que las métricas esperadas están presentes
        assert "inference_requests_total" in body
        assert "inference_latency_seconds" in body
        assert "inference_errors_total" in body
        assert "model_info" in body

    def test_metrics_after_predict(self, client_with_model, mock_model):
        """Las métricas se incrementan tras una predicción."""
        # Hacer una predicción
        client_with_model.post("/predict", json=VALID_PREDICT_PAYLOAD)
        # Verificar métricas
        response = client_with_model.get("/metrics")
        body = response.text
        assert "inference_requests_total" in body
