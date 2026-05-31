"""Prueba basada en propiedades para el manejo robusto de categorías desconocidas.

Implementa la Property 10 de la sección "Correctness Properties" del diseño,
sobre la lógica pura de ``mlops_core.features`` (tarea 4.2). Valida el
requisito RF3.5.

Para todo vocabulario de entrenamiento y toda entrada que contenga categorías no
vistas (o valores nulos), la codificación nunca lanza excepción y mapea cada
categoría desconocida a un valor reservado válido (el código de "otros").
"""

from __future__ import annotations

from typing import Any, List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.features import _is_missing, fit_categorical_encoder

# Categorías del dominio: valores discretos, hasheables y ordenables como str.
_category = st.one_of(
    st.text(min_size=1, max_size=6),
    st.integers(min_value=-50, max_value=50),
)

# Valores nulos que deben mapear al código reservado de "otros".
_null = st.sampled_from([None, float("nan")])


@st.composite
def _vocabulary_and_inputs(draw) -> Tuple[List[Any], List[Any]]:
    """Genera un vocabulario de entrenamiento y entradas que mezclan
    categorías conocidas, desconocidas y nulas.

    El vocabulario puede ser vacío (caso límite). Las entradas conocidas se
    muestrean del propio vocabulario; las desconocidas se generan libremente
    (pueden coincidir con el vocabulario, y la aserción lo trata de forma
    genérica); los nulos cubren ``None`` y ``NaN``.
    """

    vocabulary = draw(st.lists(_category, max_size=10))

    choices = [_category, _null]  # desconocidas + nulos siempre presentes
    if vocabulary:
        choices.append(st.sampled_from(vocabulary))  # conocidas

    inputs = draw(st.lists(st.one_of(*choices), max_size=20))
    return vocabulary, inputs


# Feature: mlops-real-estate-platform, Property 10: Manejo robusto de categorías desconocidas
@settings(max_examples=100)
@given(data=_vocabulary_and_inputs())
def test_property_manejo_robusto_de_categorias_desconocidas(
    data: Tuple[List[Any], List[Any]],
) -> None:
    """Codificar entradas con categorías no vistas nunca lanza, produce códigos
    enteros en ``[0, others_code]``, mapea desconocidos/nulos a ``others_code`` y
    conserva el código asignado a cada categoría conocida."""

    vocabulary, inputs = data

    encoder = fit_categorical_encoder(vocabulary, "city")

    # Nunca lanza excepción al codificar (RF3.5).
    codes = encoder.encode(inputs)

    fitted = set(encoder.categories)
    others_code = encoder.others_code

    # El código reservado es un entero válido y distinto de los conocidos.
    assert others_code == len(encoder.categories)
    assert len(codes) == len(inputs)

    for value, code in zip(inputs, codes):
        # Todo código es un entero válido dentro del rango reservado.
        assert isinstance(code, int)
        assert 0 <= code <= others_code

        if _is_missing(value) or value not in fitted:
            # Desconocidos y nulos se mapean al código de "otros".
            assert code == others_code
        else:
            # Las categorías conocidas conservan su código asignado en mapping.
            assert code == encoder.mapping[value]
            assert 0 <= code < others_code

    # Codificar valor a valor coincide con codificar el iterable completo.
    assert [encoder.encode_value(v) for v in inputs] == codes
