"""Prueba basada en propiedades para la validación de calidad del lote.

Implementa la Property 7 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.validation.validate_quality`` (tarea 3.3).
Valida los requisitos RF3.2 y RF3.8.

La propiedad afirma que ``validate_quality`` marca el lote como **no apto**
(``valid=False``) si y solo si existe al menos un problema grave:

- alguna columna con **más del 50%** de nulos (estrictamente ``> 0.5``), o
- uno o más precios menores o iguales a cero (``price <= 0``), o
- **más del 30%** de precios nulos (estrictamente ``> 0.3``).

La condición de "problema grave" se calcula de forma independiente en este test
(sin invocar la implementación) y se compara contra ``not result.valid``.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.validation import EXPECTED_SCHEMA, validate_quality

# Columnas posibles del lote (incluye "price", cuya calidad se evalúa aparte).
_COLUMNS = list(EXPECTED_SCHEMA.keys())

# Umbrales por defecto de ``validate_quality`` (estrictamente mayores).
_NULL_COL_THRESHOLD = 0.5
_NULL_PRICE_THRESHOLD = 0.3

# Valores finitos acotados para no introducir inf/nan implícitos.
_finite_floats = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9
)

# Valor genérico para columnas distintas de "price": null (None/NaN), número o texto.
_generic_value = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.integers(min_value=-1000, max_value=1000),
    _finite_floats,
    st.text(min_size=0, max_size=5),
)

# Valor para "price": null (None/NaN) o número (puede ser <= 0 para inyectar inválidos).
_price_value = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.integers(min_value=-1000, max_value=1000),
    _finite_floats,
)


@st.composite
def _record(draw: st.DrawFn) -> dict:
    """Genera un registro con inyección controlada de nulos y precios inválidos.

    Cada columna puede estar ausente (cuenta como nulo), ser nula (``None``/NaN),
    numérica o textual; ``price`` puede además ser <= 0 para inyectar precios
    inválidos.
    """
    record: dict = {}
    for column in _COLUMNS:
        if not draw(st.booleans()):
            continue  # columna ausente -> nulo para ese registro
        if column == "price":
            record[column] = draw(_price_value)
        else:
            record[column] = draw(_generic_value)
    return record


def _is_null(value: Any) -> bool:
    """Réplica independiente de la noción de nulo: ``None`` o ``NaN``."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_float(value: Any) -> Optional[float]:
    """Réplica independiente de la conversión a ``float`` usada para precios."""
    if _is_null(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None
    return None


def _expected_not_fit(records: List[Mapping[str, Any]]) -> bool:
    """Calcula de forma independiente si el lote tiene un problema grave.

    Devuelve ``True`` cuando el lote debe marcarse como **no apto**.
    """
    total = len(records)
    if total == 0:
        return True  # lote vacío: no apto (RF3.9)

    columns = set()
    for record in records:
        columns.update(record.keys())

    # Alguna columna con más del 50% de nulos (estrictamente > 0.5).
    high_null = False
    for column in columns:
        null_count = sum(
            1 for record in records if column not in record or _is_null(record[column])
        )
        if null_count / total > _NULL_COL_THRESHOLD:
            high_null = True
            break

    # Precios inválidos (<= 0) y proporción de precios nulos (> 0.3).
    invalid_prices = 0
    null_price_count = 0
    for record in records:
        if "price" not in record or _is_null(record["price"]):
            null_price_count += 1
            continue
        price = _to_float(record["price"])
        if price is not None and price <= 0:
            invalid_prices += 1
    null_price_ratio = null_price_count / total

    return (
        high_null
        or invalid_prices > 0
        or null_price_ratio > _NULL_PRICE_THRESHOLD
    )


# Feature: mlops-real-estate-platform, Property 7: Validación de calidad y aptitud del lote
@settings(max_examples=100)
@given(records=st.lists(_record(), min_size=0, max_size=12))
def test_property_validacion_calidad_y_aptitud_del_lote(
    records: List[dict],
) -> None:
    """Para todo conjunto de registros, ``validate_quality`` marca el lote como
    no apto (``valid=False``) si y solo si existe al menos un problema grave:
    una columna con > 50% de nulos, uno o más precios <= 0, o > 30% de precios
    nulos.
    """
    result = validate_quality(records)
    expected_not_fit = _expected_not_fit(records)

    assert (not result.valid) == expected_not_fit
