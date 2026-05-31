"""Construcción pura de payloads de auditoría, experimento e inferencia (RF4.6,
RF6.7, RF5.5, RF8.2, RF8.3).

Este módulo no realiza I/O: solo construye estructuras de datos y las serializa
hacia / desde JSON. Los adaptadores de Airflow (``log_result``,
``train_candidates``) y la API de inferencia (``/predict``) delegan en estas
funciones para producir registros uniformes y reproducibles.

Funciones de construcción:

- ``build_audit_event``: registro de la Tabla_Auditoria (``training_history``)
  con la decisión de entrenamiento y su motivo, la decisión de promoción y su
  motivo (incluyendo el cambio de desempeño frente a producción), el mejor
  modelo, su MAE de validación y los identificadores de reproducibilidad
  (run de MLflow, versión del modelo y commit de código).
- ``build_experiment_payload``: payload del experimento de MLflow con los
  ``batch_number`` usados, el motivo del entrenamiento, los parámetros del
  modelo y del preprocesamiento, las métricas y el commit de código.
- ``build_inference_event``: evento de inferencia con marca de tiempo en UTC,
  datos de entrada, predicción, versión del modelo, estado (``ok``/``error``)
  y mensaje de error (no vacío cuando el estado es ``error``; ``None`` cuando
  es ``ok``).

Helpers de serialización (round-trip JSON estable para auditoría e inferencia):

- ``audit_event_to_json`` / ``audit_event_from_json``
- ``experiment_payload_to_json`` / ``experiment_payload_from_json``
- ``inference_event_to_json`` / ``inference_event_from_json``
- variantes ``*_to_dict`` / ``*_from_dict`` para componer payloads mayores
- ``serialize`` / ``deserialize`` genéricos por tipo de evento
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Union

from mlops_core.types import (
    AuditEvent,
    ExperimentPayload,
    InferenceEvent,
    InferenceStatus,
)

__all__ = [
    "build_audit_event",
    "build_experiment_payload",
    "build_inference_event",
    "audit_event_to_dict",
    "audit_event_from_dict",
    "audit_event_to_json",
    "audit_event_from_json",
    "experiment_payload_to_dict",
    "experiment_payload_from_dict",
    "experiment_payload_to_json",
    "experiment_payload_from_json",
    "inference_event_to_dict",
    "inference_event_from_dict",
    "inference_event_to_json",
    "inference_event_from_json",
    "serialize",
    "deserialize",
]


# ---------------------------------------------------------------------------
# Utilidades internas de tiempo
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Marca de tiempo actual con zona horaria UTC explícita (RF8.2)."""
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    """Normaliza un ``datetime`` a UTC consciente de zona horaria.

    Un ``datetime`` naíf se interpreta como UTC; uno consciente se convierte a
    UTC. Esto garantiza que la marca de tiempo serializada sea inequívoca y
    round-trippeable.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dt_to_iso(value: Optional[datetime]) -> Optional[str]:
    """Serializa un ``datetime`` a ISO-8601 en UTC, o ``None``."""
    if value is None:
        return None
    return _ensure_utc(value).isoformat()


def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    """Deserializa una cadena ISO-8601 a ``datetime`` UTC, o ``None``.

    Acepta el sufijo ``Z`` además del offset ``+00:00`` producido por
    ``datetime.isoformat``.
    """
    if value is None:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return _ensure_utc(parsed)


# ---------------------------------------------------------------------------
# Evento de auditoría (Tabla_Auditoria / training_history) - RF4.6, RF6.7
# ---------------------------------------------------------------------------

def build_audit_event(
    *,
    batch_number: int,
    trained: bool,
    train_reason: str,
    promoted: bool,
    promotion_reason: str,
    best_model: Optional[str] = None,
    best_val_mae: Optional[float] = None,
    mlflow_run_id: Optional[str] = None,
    mlflow_model_version: Optional[str] = None,
    code_commit: Optional[str] = None,
    logged_at: Optional[datetime] = None,
) -> AuditEvent:
    """Construye el registro de auditoría de entrenamiento/promoción (RF4.6,
    RF6.7).

    El ``promotion_reason`` debe incluir el cambio de desempeño frente al
    modelo de producción; este módulo lo conserva tal cual lo recibe. Si no se
    proporciona ``logged_at`` se usa la marca de tiempo actual en UTC.
    """
    return AuditEvent(
        batch_number=int(batch_number),
        trained=bool(trained),
        train_reason=str(train_reason),
        promoted=bool(promoted),
        promotion_reason=str(promotion_reason),
        best_model=best_model,
        best_val_mae=None if best_val_mae is None else float(best_val_mae),
        mlflow_run_id=mlflow_run_id,
        mlflow_model_version=mlflow_model_version,
        code_commit=code_commit,
        logged_at=_ensure_utc(logged_at) if logged_at is not None else _now_utc(),
    )


def audit_event_to_dict(event: AuditEvent) -> Dict[str, Any]:
    """Convierte un ``AuditEvent`` en un dict JSON-serializable."""
    data = asdict(event)
    data["logged_at"] = _dt_to_iso(event.logged_at)
    return data


def audit_event_from_dict(data: Dict[str, Any]) -> AuditEvent:
    """Reconstruye un ``AuditEvent`` a partir de su representación en dict."""
    return AuditEvent(
        batch_number=int(data["batch_number"]),
        trained=bool(data["trained"]),
        train_reason=data["train_reason"],
        promoted=bool(data["promoted"]),
        promotion_reason=data["promotion_reason"],
        best_model=data.get("best_model"),
        best_val_mae=data.get("best_val_mae"),
        mlflow_run_id=data.get("mlflow_run_id"),
        mlflow_model_version=data.get("mlflow_model_version"),
        code_commit=data.get("code_commit"),
        logged_at=_iso_to_dt(data.get("logged_at")),
    )


def audit_event_to_json(event: AuditEvent) -> str:
    """Serializa un ``AuditEvent`` a una cadena JSON."""
    return json.dumps(audit_event_to_dict(event), sort_keys=True)


def audit_event_from_json(payload: str) -> AuditEvent:
    """Deserializa un ``AuditEvent`` desde una cadena JSON."""
    return audit_event_from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Payload de experimento de MLflow - RF5
# ---------------------------------------------------------------------------

def build_experiment_payload(
    *,
    batch_numbers: Optional[Iterable[int]] = None,
    train_reason: str = "",
    model_params: Optional[Dict[str, Any]] = None,
    preprocessing_params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    code_commit: Optional[str] = None,
) -> ExperimentPayload:
    """Construye el payload del experimento de MLflow (RF5.2-RF5.5).

    Reúne los lotes usados, el motivo del entrenamiento, los parámetros del
    modelo y del preprocesamiento, las métricas y el commit de código para
    reproducibilidad. Las colecciones se copian para evitar aliasing con las
    estructuras del llamador.
    """
    return ExperimentPayload(
        batch_numbers=[int(b) for b in (batch_numbers or [])],
        train_reason=str(train_reason),
        model_params=dict(model_params or {}),
        preprocessing_params=dict(preprocessing_params or {}),
        metrics={k: float(v) for k, v in (metrics or {}).items()},
        code_commit=code_commit,
    )


def experiment_payload_to_dict(payload: ExperimentPayload) -> Dict[str, Any]:
    """Convierte un ``ExperimentPayload`` en un dict JSON-serializable."""
    return asdict(payload)


def experiment_payload_from_dict(data: Dict[str, Any]) -> ExperimentPayload:
    """Reconstruye un ``ExperimentPayload`` desde su representación en dict."""
    return ExperimentPayload(
        batch_numbers=[int(b) for b in data.get("batch_numbers", [])],
        train_reason=data.get("train_reason", ""),
        model_params=dict(data.get("model_params", {})),
        preprocessing_params=dict(data.get("preprocessing_params", {})),
        metrics={k: float(v) for k, v in data.get("metrics", {}).items()},
        code_commit=data.get("code_commit"),
    )


def experiment_payload_to_json(payload: ExperimentPayload) -> str:
    """Serializa un ``ExperimentPayload`` a una cadena JSON."""
    return json.dumps(experiment_payload_to_dict(payload), sort_keys=True)


def experiment_payload_from_json(payload: str) -> ExperimentPayload:
    """Deserializa un ``ExperimentPayload`` desde una cadena JSON."""
    return experiment_payload_from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Evento de inferencia (inference_log) - RF8.2, RF8.3
# ---------------------------------------------------------------------------

def _coerce_status(status: Union[InferenceStatus, str, None]) -> Optional[InferenceStatus]:
    """Normaliza el estado a ``InferenceStatus`` (acepta enum o cadena)."""
    if status is None:
        return None
    if isinstance(status, InferenceStatus):
        return status
    return InferenceStatus(str(status))


def build_inference_event(
    *,
    input_data: Optional[Dict[str, Any]] = None,
    prediction: Optional[float] = None,
    model_version: Optional[str] = None,
    status: Union[InferenceStatus, str, None] = None,
    error: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> InferenceEvent:
    """Construye un evento de inferencia consistente (RF8.2, RF8.3).

    Garantiza las invariantes del dominio:

    - La marca de tiempo es siempre consciente de zona horaria en UTC; si no se
      proporciona se usa la actual.
    - El estado se deriva del ``error`` cuando no se especifica: ``error`` si se
      recibe un mensaje no vacío, ``ok`` en caso contrario.
    - Cuando el estado es ``error``, el campo ``error`` queda garantizado como
      no vacío (se usa un mensaje por defecto si falta).
    - Cuando el estado es ``ok``, el campo ``error`` se normaliza a ``None``.
    """
    ts = _ensure_utc(timestamp) if timestamp is not None else _now_utc()
    normalized_error = error if (error is not None and str(error).strip() != "") else None

    resolved = _coerce_status(status)
    if resolved is None:
        resolved = InferenceStatus.ERROR if normalized_error is not None else InferenceStatus.OK

    if resolved is InferenceStatus.ERROR:
        if normalized_error is None:
            normalized_error = "unknown error"
    else:  # OK: no se conserva mensaje de error
        normalized_error = None

    return InferenceEvent(
        timestamp=ts,
        input_data=dict(input_data or {}),
        prediction=None if prediction is None else float(prediction),
        model_version=model_version,
        status=resolved,
        error=normalized_error,
    )


def inference_event_to_dict(event: InferenceEvent) -> Dict[str, Any]:
    """Convierte un ``InferenceEvent`` en un dict JSON-serializable."""
    return {
        "timestamp": _dt_to_iso(event.timestamp),
        "input_data": dict(event.input_data),
        "prediction": event.prediction,
        "model_version": event.model_version,
        "status": event.status.value,
        "error": event.error,
    }


def inference_event_from_dict(data: Dict[str, Any]) -> InferenceEvent:
    """Reconstruye un ``InferenceEvent`` desde su representación en dict."""
    timestamp = _iso_to_dt(data.get("timestamp"))
    if timestamp is None:
        raise ValueError("inference event dict requires a 'timestamp'")
    return InferenceEvent(
        timestamp=timestamp,
        input_data=dict(data.get("input_data", {})),
        prediction=data.get("prediction"),
        model_version=data.get("model_version"),
        status=InferenceStatus(data.get("status", InferenceStatus.OK.value)),
        error=data.get("error"),
    )


def inference_event_to_json(event: InferenceEvent) -> str:
    """Serializa un ``InferenceEvent`` a una cadena JSON."""
    return json.dumps(inference_event_to_dict(event), sort_keys=True)


def inference_event_from_json(payload: str) -> InferenceEvent:
    """Deserializa un ``InferenceEvent`` desde una cadena JSON."""
    return inference_event_from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Helpers genéricos de serialización por tipo de evento
# ---------------------------------------------------------------------------

_TO_JSON = {
    AuditEvent: audit_event_to_json,
    ExperimentPayload: experiment_payload_to_json,
    InferenceEvent: inference_event_to_json,
}

_FROM_JSON = {
    AuditEvent: audit_event_from_json,
    ExperimentPayload: experiment_payload_from_json,
    InferenceEvent: inference_event_from_json,
}


def serialize(event: Union[AuditEvent, ExperimentPayload, InferenceEvent]) -> str:
    """Serializa cualquier evento soportado a JSON despachando por su tipo."""
    serializer = _TO_JSON.get(type(event))
    if serializer is None:
        raise TypeError(f"Unsupported event type: {type(event)!r}")
    return serializer(event)


def deserialize(
    payload: str,
    event_type: type,
) -> Union[AuditEvent, ExperimentPayload, InferenceEvent]:
    """Deserializa una cadena JSON al ``event_type`` indicado."""
    deserializer = _FROM_JSON.get(event_type)
    if deserializer is None:
        raise TypeError(f"Unsupported event type: {event_type!r}")
    return deserializer(payload)
