"""Prueba basada en propiedades para la regla de promoción del modelo (RF6).

Implementa la Property 13 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.promotion.should_promote`` (tarea 7.2).
Valida los requisitos RF6.1, RF6.2, RF6.3, RF6.4 y RF6.5.

La propiedad afirma que, para todo par de métricas candidato/champion,
``should_promote`` devuelve ``True`` si y solo si:

- no existe champion previo (``champion is None``); o bien
- el MAE del candidato mejora al menos ``mae_improvement_pct`` respecto al
  champion **y** el RMSE del candidato no empeora más de
  ``rmse_max_worsening_pct``.

En cualquier otro caso devuelve ``False``.

El valor esperado se calcula de forma independiente en este test usando la
misma forma de umbral que la implementación (sin división, robusta), y se
compara contra ``result.promote``.
"""

from __future__ import annotations

from typing import Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.promotion import should_promote
from mlops_core.types import Metrics

# Umbrales explícitos (espejo de los valores por defecto de ``should_promote``).
_MAE_IMPROVEMENT_PCT = 0.03
_RMSE_MAX_WORSENING_PCT = 0.01


def _metrics() -> st.SearchStrategy[Metrics]:
    """Genera ``Metrics`` con MAE/RMSE positivos y finitos.

    Se usa ``min_value`` estrictamente positivo para mantener bien definidas las
    razones de mejora/empeoramiento, y un rango acotado para evitar problemas de
    precisión en punto flotante.
    """
    positive = st.floats(
        min_value=1e-3,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    )
    return st.builds(Metrics, mae=positive, rmse=positive)


def _champion() -> st.SearchStrategy[Optional[Metrics]]:
    """Genera un champion: a veces ``None`` (primer modelo), a veces ``Metrics``."""
    return st.one_of(st.none(), _metrics())


def _expected_promote(
    candidate: Metrics,
    champion: Optional[Metrics],
    mae_improvement_pct: float,
    rmse_max_worsening_pct: float,
) -> bool:
    """Calcula la promoción esperada de forma independiente a la implementación."""
    if champion is None:
        return True
    mae_improved = candidate.mae <= champion.mae * (1.0 - mae_improvement_pct)
    rmse_guard_ok = candidate.rmse <= champion.rmse * (1.0 + rmse_max_worsening_pct)
    return mae_improved and rmse_guard_ok


# Feature: mlops-real-estate-platform, Property 13: Regla de promoción (MAE con guarda de RMSE)
@settings(max_examples=100)
@given(candidate=_metrics(), champion=_champion())
def test_property_regla_de_promocion_mae_con_guarda_de_rmse(
    candidate: Metrics,
    champion: Optional[Metrics],
) -> None:
    """Para todo par candidato/champion, ``should_promote`` promueve si y solo si
    no hay champion previo, o el MAE mejora lo suficiente y el RMSE no empeora
    más de lo tolerado; en cualquier otro caso no promueve.
    """
    result = should_promote(
        candidate,
        champion,
        mae_improvement_pct=_MAE_IMPROVEMENT_PCT,
        rmse_max_worsening_pct=_RMSE_MAX_WORSENING_PCT,
    )
    expected = _expected_promote(
        candidate,
        champion,
        _MAE_IMPROVEMENT_PCT,
        _RMSE_MAX_WORSENING_PCT,
    )

    assert result.promote == expected
