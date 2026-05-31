"""Prueba basada en propiedades para la decisión automática de entrenamiento.

Implementa la Property 11 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.decision.decide_training`` (tarea 5.2).
Valida los requisitos RF3.6, RF3.7, RF4.2 y RF4.4.

La propiedad afirma que, para todo ``DecisionInputs``, ``decide_training``:

- devuelve SKIP cuando ``total_records < MIN_RECORDS_TO_TRAIN`` o
  ``quality_valid`` es falso;
- en caso contrario, devuelve TRAIN si y solo si hay drift, o la proporción de
  volumen nuevo (``inserted_records / total_records``) es mayor o igual a
  ``MIN_VOLUME_INCREASE_PCT``, o ``new_categories_count`` alcanza el mínimo;
- y SKIP cuando ninguna señal técnica se cumple.

La decisión esperada se calcula de forma independiente en este test (sin
invocar la implementación) y se compara contra ``result.decision`` y
``result.should_train``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.decision import decide_training
from mlops_core.types import DecisionInputs, DecisionType

# Umbrales explícitos (espejo de los valores por defecto de ``decide_training``).
_MIN_RECORDS = 1000
_MIN_VOLUME_PCT = 0.05
_MIN_NEW_CATEGORIES = 5


@st.composite
def _decision_inputs(draw: st.DrawFn) -> DecisionInputs:
    """Genera ``DecisionInputs`` cubriendo el espacio de entrada de forma inteligente.

    Sesga ``total_records`` alrededor del umbral de suficiencia (1000) y
    ``new_categories_count`` alrededor de su mínimo (5), y genera
    ``inserted_records`` en ``[0, total_records]`` para barrer la proporción de
    volumen nuevo a ambos lados de ``MIN_VOLUME_INCREASE_PCT`` (0.05).
    """
    # ``total_records`` sesgado para straddlear el umbral de suficiencia.
    total_records = draw(
        st.one_of(
            st.integers(min_value=0, max_value=5000),
            st.sampled_from([0, 999, 1000, 1001, 2000]),
        )
    )
    # ``inserted_records`` en [0, total] barre proporciones nuevas en [0, 1].
    inserted_records = draw(st.integers(min_value=0, max_value=max(total_records, 0)))
    drift_detected = draw(st.booleans())
    quality_valid = draw(st.booleans())
    new_categories_count = draw(
        st.one_of(
            st.integers(min_value=0, max_value=20),
            st.sampled_from([0, 4, 5, 6]),
        )
    )
    return DecisionInputs(
        total_records=total_records,
        inserted_records=inserted_records,
        drift_detected=drift_detected,
        quality_valid=quality_valid,
        new_categories_count=new_categories_count,
    )


def _expected_decision(inp: DecisionInputs) -> DecisionType:
    """Calcula la decisión esperada de forma independiente a la implementación."""
    if inp.total_records < _MIN_RECORDS:
        return DecisionType.SKIP
    if not inp.quality_valid:
        return DecisionType.SKIP

    volume_pct = (
        inp.inserted_records / inp.total_records if inp.total_records > 0 else 0.0
    )
    technical_signal = (
        inp.drift_detected
        or volume_pct >= _MIN_VOLUME_PCT
        or inp.new_categories_count >= _MIN_NEW_CATEGORIES
    )
    return DecisionType.TRAIN if technical_signal else DecisionType.SKIP


# Feature: mlops-real-estate-platform, Property 11: Decisión de entrenamiento por reglas técnicas
@settings(max_examples=100)
@given(inp=_decision_inputs())
def test_property_decision_de_entrenamiento_por_reglas_tecnicas(
    inp: DecisionInputs,
) -> None:
    """Para todo ``DecisionInputs``, ``decide_training`` devuelve SKIP cuando los
    datos son insuficientes o la calidad no es apta; en caso contrario devuelve
    TRAIN si y solo si hay drift, volumen nuevo suficiente o categorías nuevas
    suficientes; y SKIP cuando ninguna señal técnica se cumple.
    """
    result = decide_training(
        inp,
        min_records=_MIN_RECORDS,
        min_volume_pct=_MIN_VOLUME_PCT,
        min_new_categories=_MIN_NEW_CATEGORIES,
    )
    expected = _expected_decision(inp)

    assert result.decision == expected
    assert result.should_train == (expected is DecisionType.TRAIN)
