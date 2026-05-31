"""Prueba basada en propiedades para la detección de nuevas categorías.

Implementa la Property 9 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.validation.detect_drift`` (tarea 3.5).
Valida el requisito RF3.4.

Para toda variable categórica, el conjunto de categorías reportadas como nuevas
es exactamente la diferencia entre las categorías del lote y las categorías del
histórico, evaluado para todas las variables categóricas.
"""

from __future__ import annotations

from typing import Dict, Set

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.types import NumericStats
from mlops_core.validation import detect_drift

# Variables categóricas del dominio (RF3.4): status, city, state.
CATEGORICAL_VARIABLES = ["status", "city", "state"]

# Estrategia de categorías: valores discretos sencillos y ordenables como str.
_category = st.one_of(
    st.text(min_size=1, max_size=6),
    st.integers(min_value=-50, max_value=50),
)

# Conjunto de categorías por variable (puede ser vacío).
_category_set = st.sets(_category, max_size=8)

# Diccionario de categorías para todas las variables categóricas del dominio.
_categories_by_variable = st.fixed_dictionaries(
    {variable: _category_set for variable in CATEGORICAL_VARIABLES}
)


# Feature: mlops-real-estate-platform, Property 9: Detección de nuevas categorías
@settings(max_examples=100)
@given(
    batch_categories=_categories_by_variable,
    hist_categories=_categories_by_variable,
)
def test_property_deteccion_de_nuevas_categorias(
    batch_categories: Dict[str, Set[object]],
    hist_categories: Dict[str, Set[object]],
) -> None:
    """Para cada variable categórica, ``detect_drift`` reporta como nuevas
    exactamente las categorías que están en el lote pero no en el histórico.

    Se provee histórico no vacío (``hist_stats.means`` con una media) para que
    no se tome la rama ``no_history`` y se ejercite realmente la diferencia de
    categorías por variable.
    """
    # Histórico numérico no vacío => no se toma la rama ``no_history``.
    new_stats = NumericStats(means={"price": 100.0})
    hist_stats = NumericStats(means={"price": 100.0})

    result = detect_drift(
        new_stats=new_stats,
        hist_stats=hist_stats,
        new_categories=batch_categories,
        hist_categories=hist_categories,
    )

    for variable in CATEGORICAL_VARIABLES:
        expected = set(batch_categories[variable]) - set(hist_categories[variable])
        assert set(result.new_categories[variable]) == expected
