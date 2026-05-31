"""Prueba basada en propiedades para el evento de inferencia.

Implementa la Property 16 de la sección "Correctness Properties" del diseño,
sobre las funciones puras ``mlops_core.logging_schema.build_inference_event``,
``inference_event_to_json`` e ``inference_event_from_json`` (tarea 6.3).
Valida los requisitos RF8.2 y RF8.3.

La propiedad afirma que, para toda solicitud de inferencia (exitosa o fallida),
``build_inference_event`` produce un evento que incluye:

- una marca de tiempo consciente de zona horaria en UTC,
- los datos de entrada,
- la predicción,
- la versión del modelo y
- el estado (``ok``/``error``);

y que cuando la solicitud falla el estado es ``error`` con un campo de error no
vacío, mientras que cuando es ``ok`` el campo de error es ``None``. Finalmente,
serializar el evento a JSON y deserializarlo preserva todos los campos.

La resolución esperada de estado y error se calcula de forma independiente en
este test (espejo del contrato del dominio, sin invocar la implementación) y se
compara contra el evento producido.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.logging_schema import (
    build_inference_event,
    inference_event_from_json,
    inference_event_to_json,
)
from mlops_core.types import InferenceStatus

# Valores escalares JSON-serializables y estables bajo round-trip (sin NaN/inf,
# que romperían la igualdad o no son JSON estándar).
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)


@st.composite
def _aware_or_naive_datetime(draw: st.DrawFn) -> datetime:
    """Genera ``datetime`` naíf o consciente con offsets variados.

    ``build_inference_event`` normaliza ambos casos a UTC, de modo que el test
    ejercita la normalización de zona horaria además del round-trip.
    """
    base = draw(st.datetimes(min_value=datetime(1970, 1, 1), max_value=datetime(2100, 1, 1)))
    offset_hours = draw(st.one_of(st.none(), st.integers(min_value=-12, max_value=14)))
    if offset_hours is None:
        return base  # naíf -> interpretado como UTC
    return base.replace(tzinfo=timezone(timedelta(hours=offset_hours)))


def _resolve_expected(
    status: Union[InferenceStatus, str, None],
    error: Optional[str],
) -> tuple[InferenceStatus, Optional[str]]:
    """Calcula el (estado, error) esperados según el contrato del dominio."""
    normalized_error = error if (error is not None and str(error).strip() != "") else None

    if status is None:
        resolved = (
            InferenceStatus.ERROR if normalized_error is not None else InferenceStatus.OK
        )
    elif isinstance(status, InferenceStatus):
        resolved = status
    else:
        resolved = InferenceStatus(str(status))

    if resolved is InferenceStatus.ERROR:
        if normalized_error is None:
            normalized_error = "unknown error"
    else:
        normalized_error = None
    return resolved, normalized_error


# Feature: mlops-real-estate-platform, Property 16: Cobertura y round-trip del evento de inferencia
@settings(max_examples=100)
@given(
    input_data=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=_json_scalars,
        max_size=6,
    ),
    prediction=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    model_version=st.one_of(st.none(), st.text(max_size=20)),
    error=st.one_of(
        st.none(),
        st.just(""),
        st.just("   "),
        st.text(max_size=40),
    ),
    status=st.one_of(
        st.none(),
        st.sampled_from(list(InferenceStatus)),
        st.sampled_from(["ok", "error"]),
    ),
    timestamp=st.one_of(st.none(), _aware_or_naive_datetime()),
)
def test_property_cobertura_y_round_trip_del_evento_de_inferencia(
    input_data: Dict[str, Any],
    prediction: Optional[float],
    model_version: Optional[str],
    error: Optional[str],
    status: Union[InferenceStatus, str, None],
    timestamp: Optional[datetime],
) -> None:
    """Para toda solicitud de inferencia, ``build_inference_event`` cubre todos
    los campos requeridos, deriva correctamente estado/error y es estable bajo
    serialización/deserialización JSON.
    """
    event = build_inference_event(
        input_data=input_data,
        prediction=prediction,
        model_version=model_version,
        status=status,
        error=error,
        timestamp=timestamp,
    )

    expected_status, expected_error = _resolve_expected(status, error)

    # Cobertura: la marca de tiempo siempre está presente y es UTC consciente.
    assert event.timestamp is not None
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() == timedelta(0)

    # Cobertura: datos de entrada, predicción y versión del modelo presentes.
    assert event.input_data == input_data
    expected_prediction = None if prediction is None else float(prediction)
    assert event.prediction == expected_prediction
    assert event.model_version == model_version

    # Estado y error derivados según el contrato del dominio (RF8.2, RF8.3).
    assert event.status == expected_status
    if event.status is InferenceStatus.ERROR:
        assert event.error is not None and event.error.strip() != ""
    else:
        assert event.error is None
    assert event.error == expected_error

    # Round-trip JSON: serializar y deserializar preserva todos los campos.
    restored = inference_event_from_json(inference_event_to_json(event))
    assert restored.timestamp == event.timestamp
    assert restored.input_data == event.input_data
    assert restored.prediction == event.prediction
    assert restored.model_version == event.model_version
    assert restored.status == event.status
    assert restored.error == event.error
