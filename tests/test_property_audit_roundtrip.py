"""Prueba basada en propiedades del round-trip del registro de auditoría (tarea 6.2).

Implementa la Property 14 de la sección "Correctness Properties" del diseño,
sobre la construcción y serialización del registro de la Tabla_Auditoria
(``training_history``) que realizan ``mlops_core.logging_schema``.

La propiedad afirma que, *para toda* decisión de entrenamiento/promoción, el
``AuditEvent`` construido por ``build_audit_event`` contiene la decisión de
entrenamiento, su motivo, la decisión de promoción y su motivo (incluyendo el
cambio de desempeño frente a producción), y que serializar a JSON y
deserializar de vuelta preserva todos esos campos sin pérdida.

Se ejercitan tanto los helpers específicos (``audit_event_to_json`` /
``audit_event_from_json``) como los genéricos (``serialize`` / ``deserialize``).
La marca de tiempo ``logged_at`` se compara como instante UTC tras el round-trip
(``build_audit_event`` normaliza cualquier ``datetime`` naíf o consciente a UTC).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.logging_schema import (
    audit_event_from_json,
    audit_event_to_json,
    build_audit_event,
    deserialize,
    serialize,
)
from mlops_core.types import AuditEvent


# Texto opcional para identificadores y campos descriptivos (acepta None).
_optional_text = st.one_of(st.none(), st.text(max_size=50))

# MAE de validación: float finito opcional (sin NaN/inf para round-trip JSON
# exacto) o None.
_optional_mae = st.one_of(
    st.none(),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
)

# Zonas horarias fijas para ejercitar la normalización a UTC (naíf + offsets).
_tz_strategy = st.sampled_from(
    [
        timezone.utc,
        timezone(timedelta(hours=5)),
        timezone(timedelta(hours=-8)),
        timezone(timedelta(hours=5, minutes=30)),
    ]
)

# ``logged_at`` opcional: ausente (usa _now_utc), naíf (interpretado como UTC) o
# consciente con un offset fijo (convertido a UTC).
_optional_logged_at = st.one_of(
    st.none(),
    st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2100, 1, 1),
        timezones=st.one_of(st.none(), _tz_strategy),
    ),
)


# Feature: mlops-real-estate-platform, Property 14: Round-trip del registro de auditoría
@settings(max_examples=100)
@given(
    batch_number=st.integers(min_value=0, max_value=1_000_000),
    trained=st.booleans(),
    train_reason=st.text(max_size=80),
    promoted=st.booleans(),
    promotion_reason=st.text(max_size=80),
    best_model=_optional_text,
    best_val_mae=_optional_mae,
    mlflow_run_id=_optional_text,
    mlflow_model_version=_optional_text,
    code_commit=_optional_text,
    logged_at=_optional_logged_at,
)
def test_audit_event_roundtrip_preserves_all_fields(
    batch_number: int,
    trained: bool,
    train_reason: str,
    promoted: bool,
    promotion_reason: str,
    best_model: Optional[str],
    best_val_mae: Optional[float],
    mlflow_run_id: Optional[str],
    mlflow_model_version: Optional[str],
    code_commit: Optional[str],
    logged_at: Optional[datetime],
) -> None:
    """El registro de auditoría conserva todos sus campos en el round-trip JSON.

    *Para toda* decisión de entrenamiento/promoción, ``build_audit_event``
    produce un ``AuditEvent`` que contiene la decisión de entrenamiento
    (``trained``) y su motivo (``train_reason``), y la decisión de promoción
    (``promoted``) y su motivo (``promotion_reason``, que incluye el cambio de
    desempeño frente a producción); y serializar + deserializar preserva todos
    esos campos (incluyendo ``logged_at`` como instante UTC).

    **Validates: Requirements 4.6, 6.7**
    """
    event = build_audit_event(
        batch_number=batch_number,
        trained=trained,
        train_reason=train_reason,
        promoted=promoted,
        promotion_reason=promotion_reason,
        best_model=best_model,
        best_val_mae=best_val_mae,
        mlflow_run_id=mlflow_run_id,
        mlflow_model_version=mlflow_model_version,
        code_commit=code_commit,
        logged_at=logged_at,
    )

    # 1) El evento construido CONTIENE la decisión y los motivos solicitados.
    assert event.trained is trained
    assert event.train_reason == train_reason
    assert event.promoted is promoted
    # El motivo de promoción (incluido el cambio de desempeño) se conserva tal cual.
    assert event.promotion_reason == promotion_reason
    # La marca de tiempo siempre queda consciente de zona horaria en UTC.
    assert event.logged_at is not None
    assert event.logged_at.tzinfo is not None
    assert event.logged_at.utcoffset() == timedelta(0)

    # 2) Round-trip vía helpers específicos del evento de auditoría.
    restored = audit_event_from_json(audit_event_to_json(event))
    assert isinstance(restored, AuditEvent)
    _assert_audit_events_equal(restored, event)

    # 3) Round-trip vía helpers genéricos de despacho por tipo.
    restored_generic = deserialize(serialize(event), AuditEvent)
    assert isinstance(restored_generic, AuditEvent)
    _assert_audit_events_equal(restored_generic, event)


def _assert_audit_events_equal(actual: AuditEvent, expected: AuditEvent) -> None:
    """Compara dos ``AuditEvent`` campo a campo.

    ``logged_at`` se compara como instante UTC: la igualdad de ``datetime``
    conscientes de zona horaria compara el instante, y ambos extremos del
    round-trip están normalizados a UTC.
    """
    assert actual.batch_number == expected.batch_number
    assert actual.trained == expected.trained
    assert actual.train_reason == expected.train_reason
    assert actual.promoted == expected.promoted
    assert actual.promotion_reason == expected.promotion_reason
    assert actual.best_model == expected.best_model
    assert actual.best_val_mae == expected.best_val_mae
    assert actual.mlflow_run_id == expected.mlflow_run_id
    assert actual.mlflow_model_version == expected.mlflow_model_version
    assert actual.code_commit == expected.code_commit
    assert actual.logged_at == expected.logged_at
    # La igualdad estructural completa del dataclass también debe cumplirse.
    assert actual == expected
