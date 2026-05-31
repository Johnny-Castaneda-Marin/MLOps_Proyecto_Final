"""Lógica pura de validación de datos y control de cambios (RF3).

Este módulo concentra las reglas de validación del pipeline como funciones puras
(sin I/O), de modo que puedan ejercitarse de forma determinista con pruebas
basadas en propiedades y reutilizarse desde los adaptadores de Airflow.

Responsabilidades:
- ``validate_schema``: columnas faltantes/adicionales y cambios de tipo lógico
  respecto al esquema esperado (RF3.1, RF3.9).
- ``validate_quality``: proporción de nulos por columna, filas duplicadas,
  precios inválidos (<= 0) y proporción de precios nulos (RF3.2, RF3.8, RF3.9).
- ``detect_drift``: drift numérico por umbral de cambio relativo de la media y
  detección de nuevas categorías generalizada a todas las variables categóricas
  (RF3.3, RF3.4); maneja el caso ``no_history``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from mlops_core.types import (
    DriftResult,
    NumericStats,
    QualityResult,
    SchemaResult,
)

# Esquema esperado: nombre de columna -> tipo lógico ("number" / "string").
EXPECTED_SCHEMA: Dict[str, str] = {
    "brokered_by": "number",
    "status": "string",
    "price": "number",
    "bed": "number",
    "bath": "number",
    "acre_lot": "number",
    "street": "number",
    "city": "string",
    "state": "string",
    "zip_code": "number",
    "house_size": "number",
    "prev_sold_date": "string",
}

# Umbral por defecto de drift numérico (RF3.3): cambio relativo absoluto > 10%.
DRIFT_THRESHOLD = 0.1


# --------------------------------------------------------------------------- #
# Helpers internos (puros)
# --------------------------------------------------------------------------- #
def _is_null(value: Any) -> bool:
    """``True`` si el valor representa un nulo (``None`` o ``NaN``)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_number(value: Any) -> bool:
    """``True`` si el valor es numérico real (``int``/``float``, excluyendo ``bool``)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _to_float(value: Any) -> Optional[float]:
    """Convierte el valor a ``float`` cuando es posible; en caso contrario ``None``."""
    if _is_null(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None
    return None


def _infer_column_type(values: Sequence[Any]) -> Optional[str]:
    """Infiere el tipo lógico de una columna a partir de sus valores no nulos.

    Devuelve ``"number"`` si todos los valores no nulos son numéricos,
    ``"string"`` si hay al menos un valor no numérico, o ``None`` cuando no hay
    valores no nulos (no es posible inferir el tipo, por lo que no se marca como
    cambio de tipo).
    """
    non_null = [v for v in values if not _is_null(v)]
    if not non_null:
        return None
    for v in non_null:
        if not _is_number(v):
            return "string"
    return "number"


def _received_columns(records: Sequence[Mapping[str, Any]]) -> Set[str]:
    """Conjunto de columnas presentes en al menos un registro del lote."""
    columns: Set[str] = set()
    for record in records:
        columns.update(record.keys())
    return columns


def _canonical(record: Mapping[str, Any]) -> str:
    """Representación canónica de un registro para comparar duplicados."""
    return json.dumps(record, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# Validación de esquema (RF3.1, RF3.9)
# --------------------------------------------------------------------------- #
def validate_schema(
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, str] = EXPECTED_SCHEMA,
) -> SchemaResult:
    """Valida el esquema del lote contra ``expected``.

    Detecta columnas faltantes, columnas adicionales y cambios de tipo lógico
    (inferido por columna). El esquema se marca como inválido cuando existe al
    menos una discrepancia. Un lote vacío (sin registros) se marca como inválido
    sin interrumpir la ejecución (RF3.9).
    """
    if not records:
        return SchemaResult(valid=False)

    expected_cols = set(expected.keys())
    received_cols = _received_columns(records)

    missing = sorted(expected_cols - received_cols)
    extra = sorted(received_cols - expected_cols)

    type_mismatches: List[str] = []
    for column in sorted(expected_cols & received_cols):
        values = [record[column] for record in records if column in record]
        inferred = _infer_column_type(values)
        expected_type = expected[column]
        if inferred is not None and inferred != expected_type:
            type_mismatches.append(
                f"{column}: expected {expected_type}, got {inferred}"
            )

    valid = not missing and not extra and not type_mismatches
    return SchemaResult(
        valid=valid,
        missing_columns=missing,
        extra_columns=extra,
        type_mismatches=type_mismatches,
    )


# --------------------------------------------------------------------------- #
# Validación de calidad (RF3.2, RF3.8, RF3.9)
# --------------------------------------------------------------------------- #
def validate_quality(
    records: Sequence[Mapping[str, Any]],
    null_col_threshold: float = 0.5,
    null_price_threshold: float = 0.3,
) -> QualityResult:
    """Evalúa la calidad del lote.

    Calcula, por columna, la proporción de nulos (columna problemática si supera
    ``null_col_threshold``), el conteo de filas duplicadas, el conteo de precios
    inválidos (``price <= 0``) y la proporción de precios nulos (marca el lote si
    supera ``null_price_threshold``).

    El lote se marca como **no apto** (``valid=False``) ante cualquier problema
    grave: una o más columnas con más del 50% de nulos, uno o más precios menores
    o iguales a cero, o más del 30% de precios nulos. Un lote vacío se marca como
    no apto sin interrumpir la ejecución (RF3.9).
    """
    if not records:
        return QualityResult(valid=False, issues=["empty batch"])

    total = len(records)
    columns = _received_columns(records)

    # Proporción de nulos por columna (una columna ausente en un registro cuenta
    # como nulo en ese registro).
    high_null_columns: List[str] = []
    for column in sorted(columns):
        null_count = sum(
            1 for record in records if column not in record or _is_null(record[column])
        )
        if null_count / total > null_col_threshold:
            high_null_columns.append(column)

    # Filas duplicadas (réplicas exactas de una fila previa).
    unique_rows = {_canonical(record) for record in records}
    duplicate_rows = total - len(unique_rows)

    # Precios inválidos (<= 0) y proporción de precios nulos.
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

    issues: List[str] = []
    if high_null_columns:
        issues.append(f"High null rate in: {high_null_columns}")
    if invalid_prices > 0:
        issues.append(f"{invalid_prices} records with invalid price")
    if null_price_ratio > null_price_threshold:
        issues.append(
            f"More than {null_price_threshold:.0%} null prices "
            f"({null_price_ratio:.0%})"
        )
    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate rows")

    # Problemas graves que marcan el lote como no apto (RF3.8). Los duplicados se
    # reportan como issue pero no invalidan el lote por sí solos.
    grave_problem = (
        bool(high_null_columns)
        or invalid_prices > 0
        or null_price_ratio > null_price_threshold
    )

    return QualityResult(
        valid=not grave_problem,
        high_null_columns=high_null_columns,
        duplicate_rows=duplicate_rows,
        invalid_prices=invalid_prices,
        null_price_ratio=null_price_ratio,
        issues=issues,
    )


# --------------------------------------------------------------------------- #
# Detección de drift y nuevas categorías (RF3.3, RF3.4)
# --------------------------------------------------------------------------- #
def detect_drift(
    new_stats: NumericStats,
    hist_stats: NumericStats,
    new_categories: Mapping[str, Set[Any]],
    hist_categories: Mapping[str, Set[Any]],
    drift_threshold: float = DRIFT_THRESHOLD,
    new_cat_min_count: int = 5,
) -> DriftResult:
    """Detecta drift numérico y nuevas categorías comparando lote vs histórico.

    - **Drift numérico**: se reporta cuando el cambio relativo absoluto de la
      media de alguna variable numérica supera ``drift_threshold``
      (``|new_mean - hist_mean| / |hist_mean| > drift_threshold``). Las variables
      con media histórica igual a cero se omiten (no es posible el cambio
      relativo).
    - **Nuevas categorías**: para cada variable categórica del lote, el conjunto
      reportado es exactamente la diferencia ``batch - historia``. Esto se
      generaliza a todas las variables categóricas (``status``, ``city``,
      ``state``, etc.).

    Caso ``no_history``: cuando no existe histórico (ni medias numéricas ni
    categorías históricas), se devuelve ``drift_detected=False`` con
    ``details="no_history"`` (RF3.3).

    ``drift_detected`` depende **únicamente** del drift numérico; las nuevas
    categorías se reportan por separado en ``new_categories`` (el conteo se
    consume luego en la decisión de entrenamiento, RF4.2). ``new_cat_min_count``
    se usa para anotar en ``details`` cuándo el número total de categorías nuevas
    es significativo, sin alterar ``drift_detected``.
    """
    if not hist_stats.means and not hist_categories:
        return DriftResult(drift_detected=False, details="no_history")

    # Drift numérico por umbral de cambio relativo de la media.
    drifted_columns: List[str] = []
    for column, new_mean in new_stats.means.items():
        if column not in hist_stats.means:
            continue
        hist_mean = hist_stats.means[column]
        if hist_mean == 0:
            continue
        pct_change = abs(new_mean - hist_mean) / abs(hist_mean)
        if pct_change > drift_threshold:
            drifted_columns.append(column)
    drifted_columns.sort()

    # Nuevas categorías por variable: diferencia exacta batch - historia.
    new_categories_out: Dict[str, List[Any]] = {}
    total_new_categories = 0
    for variable, batch_values in new_categories.items():
        historic = hist_categories.get(variable, set())
        unseen = set(batch_values) - set(historic)
        new_categories_out[variable] = sorted(unseen, key=_category_sort_key)
        total_new_categories += len(unseen)

    drift_detected = bool(drifted_columns)

    detail_parts: List[str] = []
    if drifted_columns:
        detail_parts.append(f"numeric_drift: {drifted_columns}")
    if total_new_categories > 0:
        flag = (
            "significant" if total_new_categories >= new_cat_min_count else "minor"
        )
        detail_parts.append(f"new_categories({flag}): {total_new_categories}")
    details = "; ".join(detail_parts) if detail_parts else "no_drift"

    return DriftResult(
        drift_detected=drift_detected,
        drifted_columns=drifted_columns,
        new_categories=new_categories_out,
        details=details,
    )


def _category_sort_key(value: Any) -> str:
    """Clave de ordenamiento estable para categorías de tipo heterogéneo."""
    return str(value)
