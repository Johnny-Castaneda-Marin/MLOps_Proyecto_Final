"""Prueba basada en propiedades del particionado determinista (tarea 4.3).

Implementa la Property 17 de la sección "Correctness Properties" del diseño,
sobre :func:`mlops_core.features.split_dataset`.

La propiedad asegura la **reproducibilidad** del particionado train/val/test:
con la misma semilla (``random_state=42``), particionar dos veces el mismo
dataset produce particiones idénticas. Esto cubre tanto la asignación de la
columna ``split`` como el orden y contenido de las filas resultantes (el
barajado interno también es determinista), e incluye la columna de trazabilidad
``batch_number`` (RF2.5/RF11.3).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.features import DEFAULT_RANDOM_STATE, split_dataset

# Estrategia para un valor de columna numérica/textual realista, con nulos.
_numbers = st.one_of(
    st.none(),
    st.integers(min_value=-10_000, max_value=10_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
)
_strings = st.one_of(st.none(), st.text(max_size=12))


@st.composite
def property_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """Genera un ``DataFrame`` pequeño con ``batch_number`` y columnas variadas.

    El número de filas se mantiene reducido (0..30) porque el particionado sobre
    frames pequeños es muy rápido y permite ejercitar todos los tamaños de borde
    (vacío, una fila, etc.).
    """

    n_rows = draw(st.integers(min_value=0, max_value=30))
    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        rows.append(
            {
                "batch_number": draw(st.integers(min_value=0, max_value=50)),
                "price": draw(_numbers),
                "city": draw(_strings),
                "house_size": draw(_numbers),
                "row_id": i,  # identificador estable para comparar contenido
            }
        )
    columns = ["batch_number", "price", "city", "house_size", "row_id"]
    return pd.DataFrame(rows, columns=columns)


# Feature: mlops-real-estate-platform, Property 17: Particionado determinista (reproducibilidad)
@settings(max_examples=100, deadline=None)
@given(df=property_dataframe())
def test_split_dataset_is_deterministic(df: pd.DataFrame) -> None:
    """Particionar dos veces con la misma semilla produce particiones idénticas.

    *Para todo* dataset, ``split_dataset`` con ``random_state=42`` es
    reproducible: ambas invocaciones devuelven la misma columna ``split`` y el
    mismo orden/contenido de filas (incluida ``batch_number``).

    **Validates: Requirements 11.3**
    """
    first = split_dataset(df, random_state=DEFAULT_RANDOM_STATE)
    second = split_dataset(df, random_state=DEFAULT_RANDOM_STATE)

    # La columna de partición es idéntica entre ejecuciones.
    assert list(first["split"]) == list(second["split"])

    # El orden y contenido completo de las filas (tras el barajado determinista)
    # también coinciden exactamente.
    pd.testing.assert_frame_equal(first, second)
