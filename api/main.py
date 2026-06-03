"""FastAPI application de inferencia con recarga en caliente (RF7, RF8, RF10).

Endpoints:
- POST /predict: recibe features de propiedad, devuelve predicción + model_version.
- POST /admin/reload: fuerza recarga del modelo champion desde MLflow (auth por token).
- GET /health: liveness/readiness check.
- GET /metrics: métricas en formato Prometheus.

El modelo se sirve a través del ``ModelHolder`` thread-safe (task 9.1). La
recarga se realiza consultando MLflow por el alias ``champion``. Cada solicitud
de inferencia se registra como evento en ``inference_log`` (task 9.4).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from api.config import (
    ADMIN_TOKEN,
    APP_NAME,
    APP_VERSION,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    RELOAD_INTERVAL_SECONDS,
)
from api.inference_log import ensure_table, log_inference_async
from api.model_holder import ModelHolder, ModelNotInitializedError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (RF10.1)
# ---------------------------------------------------------------------------

registry = CollectorRegistry()

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests received",
    registry=registry,
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Latency of inference requests in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=registry,
)

INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total number of inference errors",
    registry=registry,
)

MODEL_INFO = Gauge(
    "model_info",
    "Information about the currently loaded model",
    labelnames=["version", "model_name"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Features de una propiedad inmobiliaria para predicción."""

    brokered_by: float = Field(..., description="ID del broker")
    status: str = Field(..., description="Estado de la propiedad (for_sale, etc.)")
    bed: float = Field(..., description="Número de habitaciones")
    bath: float = Field(..., description="Número de baños")
    acre_lot: float = Field(..., description="Tamaño del lote en acres")
    street: float = Field(..., description="Identificador de calle")
    city: str = Field(..., description="Ciudad")
    state: str = Field(..., description="Estado")
    zip_code: float = Field(..., description="Código postal")
    house_size: float = Field(..., description="Tamaño de la vivienda en sqft")
    prev_sold_year: Optional[float] = Field(
        None, description="Año de venta previa"
    )


class PredictResponse(BaseModel):
    """Respuesta de predicción de precio."""

    model_config = {"protected_namespaces": ()}

    prediction: float
    model_version: str
    status: str = "ok"


class ErrorResponse(BaseModel):
    """Respuesta de error."""

    detail: str
    status: str = "error"


class HealthResponse(BaseModel):
    """Respuesta del health check."""

    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ReloadResponse(BaseModel):
    """Respuesta del endpoint de recarga."""

    model_config = {"protected_namespaces": ()}

    status: str
    model_version: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def load_champion_model(holder: ModelHolder) -> str:
    """Carga el modelo champion desde MLflow y lo instala en el holder.

    Usa MLflow como única fuente de verdad (RF7.1, RF7.2): resuelve el alias
    ``champion`` para obtener la versión y carga el modelo correspondiente.
    No contiene rutas locales ni versiones fijas.

    Returns:
        La versión del modelo cargado.

    Raises:
        Exception: si MLflow no está disponible o el modelo no existe.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Resolver la versión apuntada por el alias champion
    model_version_info = client.get_model_version_by_alias(
        name=MLFLOW_MODEL_NAME, alias=MLFLOW_CHAMPION_ALIAS
    )
    version = str(model_version_info.version)

    # Cargar el modelo desde el registry usando el URI de modelos
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_CHAMPION_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Instalar atómicamente en el holder (RF7.5)
    holder.update(model, version)
    logger.info("Modelo champion v%s cargado exitosamente desde MLflow.", version)
    return version


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

# Global model holder instance
model_holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: crea tabla inference_log e intenta cargar el modelo champion."""
    # Crear tabla inference_log en raw_db si no existe (RF8.1)
    try:
        ensure_table()
        logger.info("Tabla inference_log verificada en raw_db.")
    except Exception as exc:
        logger.warning(
            "No se pudo crear/verificar la tabla inference_log: %s. "
            "El logging de inferencias podría fallar.",
            exc,
        )

    try:
        version = load_champion_model(model_holder)
        MODEL_INFO.labels(version=version, model_name=MLFLOW_MODEL_NAME).set(1)
        logger.info("API iniciada con modelo champion v%s.", version)
    except Exception as exc:
        # La API arranca sin modelo; /health reportará model_loaded=False
        logger.warning(
            "No se pudo cargar el modelo champion al iniciar: %s. "
            "La API arrancará sin modelo cargado.",
            exc,
        )

    # Background task: recarga periódica del modelo champion
    async def _periodic_reload():
        """Intenta recargar el modelo cada RELOAD_INTERVAL_SECONDS.

        Si no hay modelo cargado, reintenta cada 10s para recuperarse rápido.
        Si ya hay modelo, verifica si hay nueva versión cada RELOAD_INTERVAL_SECONDS.
        """
        while True:
            interval = 10 if not model_holder.is_initialized else RELOAD_INTERVAL_SECONDS
            await asyncio.sleep(interval)
            try:
                current_version = model_holder.version
                new_version = load_champion_model(model_holder)
                if new_version != current_version:
                    MODEL_INFO.labels(version=new_version, model_name=MLFLOW_MODEL_NAME).set(1)
                    logger.info("Modelo recargado: v%s -> v%s", current_version, new_version)
            except Exception as exc:
                logger.debug("Recarga periódica falló (se reintentará): %s", exc)

    reload_task = asyncio.create_task(_periodic_reload())

    yield

    # Cleanup al apagar
    reload_task.cancel()
    try:
        await reload_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="MLOps Real Estate Inference API",
    description="API de inferencia de precios de inmuebles con recarga en caliente.",
    version=APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse, responses={503: {"model": ErrorResponse}})
async def predict(request: PredictRequest) -> PredictResponse:
    """Recibe features de propiedad y devuelve predicción + model_version (RF7.1).

    Registra cada solicitud como evento de inferencia (RF8.2, RF8.3).
    """
    INFERENCE_REQUESTS_TOTAL.inc()
    start_time = time.time()

    # Capturar input_data para logging (RF8.2)
    input_data = request.model_dump()

    try:
        if not model_holder.is_initialized:
            raise ModelNotInitializedError(
                "No hay modelo cargado. Intente POST /admin/reload."
            )

        # Construir array de features en el orden esperado por el modelo
        features = _build_feature_array(request)

        # Predicción atómica con versión (RF7.7)
        prediction_raw, version = model_holder.predict_with_version(features)

        # Extraer valor escalar
        prediction_value = float(
            prediction_raw[0] if hasattr(prediction_raw, "__iter__") else prediction_raw
        )

        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.observe(latency)

        # Registrar inferencia exitosa en inference_log (RF8.1, RF8.2)
        log_inference_async(
            input_data=input_data,
            prediction=prediction_value,
            model_version=version,
            status="ok",
        )

        return PredictResponse(
            prediction=prediction_value,
            model_version=version,
            status="ok",
        )

    except ModelNotInitializedError as exc:
        INFERENCE_ERRORS_TOTAL.inc()
        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.observe(latency)

        # Registrar error en inference_log (RF8.3)
        log_inference_async(
            input_data=input_data,
            prediction=None,
            model_version=None,
            status="error",
            error=str(exc),
        )

        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        INFERENCE_ERRORS_TOTAL.inc()
        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.observe(latency)
        logger.exception("Error durante la inferencia: %s", exc)

        # Registrar error en inference_log (RF8.3)
        log_inference_async(
            input_data=input_data,
            prediction=None,
            model_version=model_holder.version,
            status="error",
            error=str(exc),
        )

        raise HTTPException(status_code=500, detail=f"Error de inferencia: {exc}")


@app.post("/admin/reload", response_model=ReloadResponse)
async def admin_reload(
    authorization: Optional[str] = Header(None),
) -> ReloadResponse:
    """Fuerza recarga del modelo champion desde MLflow (RF7.3, RF7.4).

    Requiere autenticación por token Bearer en el header Authorization.
    Si la recarga falla, conserva el modelo previo (RF7.6).
    """
    # Autenticación por token (RF7.4)
    _verify_admin_token(authorization)

    try:
        version = load_champion_model(model_holder)
        # Actualizar gauge de Prometheus
        MODEL_INFO.labels(version=version, model_name=MLFLOW_MODEL_NAME).set(1)
        return ReloadResponse(
            status="ok",
            model_version=version,
            message=f"Modelo champion v{version} recargado exitosamente.",
        )
    except Exception as exc:
        # RF7.6: conserva el modelo previo ante fallo
        logger.error("Fallo al recargar modelo: %s", exc)
        current_version = model_holder.version
        return ReloadResponse(
            status="error",
            model_version=current_version,
            message=f"Fallo al recargar: {exc}. Modelo previo conservado.",
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness check (RF13.2).

    Reporta si el modelo está cargado y su versión actual.
    """
    is_loaded = model_holder.is_initialized
    version = model_holder.version
    status = "healthy" if is_loaded else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=is_loaded,
        model_version=version,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Métricas en formato Prometheus (RF10.1).

    Expone: inference_requests_total, inference_latency_seconds,
    inference_errors_total, model_info{version, model_name}.
    """
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _build_feature_array(request: PredictRequest) -> np.ndarray:
    """Construye el array de features a partir del request.

    El orden de las columnas debe coincidir con el usado durante el
    entrenamiento. Se usa el mismo orden que en ``clean_properties``.
    """
    features = [
        request.brokered_by,
        _encode_status(request.status),
        request.bed,
        request.bath,
        request.acre_lot,
        request.street,
        _encode_city(request.city),
        _encode_state(request.state),
        request.zip_code,
        request.house_size,
        request.prev_sold_year if request.prev_sold_year is not None else 0.0,
    ]
    return np.array([features], dtype=np.float64)


def _encode_status(status: str) -> float:
    """Codifica el status como valor numérico (misma lógica que preprocess)."""
    status_map = {"for_sale": 0.0, "ready_to_build": 1.0, "sold": 2.0}
    return status_map.get(status.lower().strip(), -1.0)


def _encode_city(city: str) -> float:
    """Codifica la ciudad como hash numérico (categoría desconocida = -1)."""
    # En producción, el modelo pyfunc maneja la codificación internamente.
    # Este fallback numérico se usa solo si el modelo espera features numéricas.
    return float(hash(city.lower().strip()) % 10000)


def _encode_state(state: str) -> float:
    """Codifica el estado como hash numérico (categoría desconocida = -1)."""
    return float(hash(state.lower().strip()) % 1000)


def _verify_admin_token(authorization: Optional[str]) -> None:
    """Verifica el token Bearer del header Authorization (RF7.4).

    Raises:
        HTTPException 401: si el token es inválido o no se proporciona.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Se requiere autenticación. Proporcione el header Authorization: Bearer <token>.",
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Formato de autorización inválido. Use: Bearer <token>.",
        )

    token = parts[1].strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token de administración inválido.",
        )
