"""Prueba basada en propiedades de la condición de agotamiento (tarea 2.3).

Implementa la Property 2 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.ingest.is_exhausted``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.ingest import DAYS, is_exhausted


# Feature: mlops-real-estate-platform, Property 2: Condición de agotamiento
@settings(max_examples=100)
@given(
    batch_index=st.integers(min_value=-5, max_value=len(DAYS) + 5),
    http_status=st.one_of(
        st.none(),
        st.sampled_from([200, 204, 301, 400, 401, 404, 422, 500, 503]),
        st.integers(min_value=100, max_value=599),
    ),
)
def test_is_exhausted_iff_index_out_of_range_or_http_400(
    batch_index: int, http_status: int | None
) -> None:
    """``is_exhausted`` es True si y solo si el índice agota los días o HTTP 400.

    *Para todo* par ``(batch_index, http_status)``, ``is_exhausted`` devuelve
    ``True`` si y solo si ``batch_index >= len(DAYS)`` o ``http_status == 400``,
    y ``False`` en cualquier otro caso.

    **Validates: Requirements 1.6**
    """
    expected = batch_index >= len(DAYS) or http_status == 400
    assert is_exhausted(batch_index, http_status) is expected
