"""Prueba basada en propiedades para la selección de día por índice acumulado.

Implementa la Property 1 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.ingest.select_day`` y la lista ``DAYS``
(tarea 2.2). Valida el requisito RF1.1.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.ingest import DAYS, select_day


# Feature: mlops-real-estate-platform, Property 1: Selección de día por índice acumulado
@settings(max_examples=100)
@given(
    i=st.one_of(
        # Cubre explícitamente el rango de días válidos (índice dentro de DAYS),
        st.integers(min_value=0, max_value=len(DAYS) - 1),
        # y el rango de agotamiento (índice >= len(DAYS)), con un espacio amplio
        # para que Hypothesis ejecute 100 ejemplos sin agotar el dominio.
        st.integers(min_value=len(DAYS), max_value=1_000_000),
    )
)
def test_property_select_day_por_indice_acumulado(i: int) -> None:
    """Para todo índice de lote acumulado ``i``:

    - si ``i < len(DAYS)`` entonces ``select_day(i) == DAYS[i]``;
    - si ``i >= len(DAYS)`` entonces ``select_day(i) is None`` (sin excepción).
    """
    result = select_day(i)

    if i < len(DAYS):
        assert result == DAYS[i]
    else:
        assert result is None
