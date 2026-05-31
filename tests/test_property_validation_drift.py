"""Prueba basada en propiedades para la detección de drift numérico por umbral.

Implementa la Property 8 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.validation.detect_drift`` y el tipo
``mlops_core.types.NumericStats`` (tarea 3.4). Valida el requisito RF3.3.

Solo se ejercita el drift numérico: se pasan diccionarios de categorías vacíos
(``{}``) y un histórico no vacío (``hist_stats.means``), de modo que no se tomen
ni la rama ``no_history`` ni la lógica de nuevas categorías.
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.types import NumericStats
from mlops_core.validation import DRIFT_THRESHOLD, detect_drift


@st.composite
def mean_pairs(draw):
    """Genera medias de variables numéricas (lote nuevo vs histórico).

    Ambos diccionarios comparten exactamente las mismas claves. Las medias
    históricas se restringen a magnitud ``>= 1e-3`` (distintas de cero) para que
    el cambio relativo esté siempre definido y la implementación no omita la
    columna por baseline cero. Las magnitudes se acotan a ``[-1e6, 1e6]`` para
    evitar desbordamientos de punto flotante.
    """
    pairs = draw(
        st.dictionaries(
            keys=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
            values=st.tuples(
                st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ),
                st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ).filter(lambda x: abs(x) >= 1e-3),
            ),
            min_size=1,
            max_size=6,
        )
    )
    new_means = {key: value[0] for key, value in pairs.items()}
    hist_means = {key: value[1] for key, value in pairs.items()}
    return new_means, hist_means


# Feature: mlops-real-estate-platform, Property 8: Detección de drift numérico por umbral
@settings(max_examples=100)
@given(means=mean_pairs())
def test_property_deteccion_drift_numerico_por_umbral(means) -> None:
    """Para todo par de medias (lote nuevo vs histórico) sobre las mismas claves
    y con medias históricas no nulas, ``detect_drift`` reporta drift si y solo si
    el cambio relativo absoluto de la media de alguna variable numérica supera
    ``DRIFT_THRESHOLD``.
    """
    new_means, hist_means = means

    result = detect_drift(
        NumericStats(means=new_means),
        NumericStats(means=hist_means),
        {},  # sin categorías nuevas: solo se ejercita el drift numérico
        {},  # sin categorías históricas
        drift_threshold=DRIFT_THRESHOLD,
    )

    # Expectativa independiente, usando la MISMA aritmética y la misma comparación
    # estricta ``>`` que la implementación para alinear el punto flotante.
    expected_drift = any(
        abs(new_means[column] - hist_means[column]) / abs(hist_means[column])
        > DRIFT_THRESHOLD
        for column in new_means
    )

    assert result.drift_detected == expected_drift
