"""Pruebas unitarias de la regla de promoción ``should_promote`` (tarea 7.1).

Cubren los casos de aceptación RF6.2, RF6.3, RF6.4 y RF6.5 con ejemplos y
casos borde concretos. La prueba de propiedad correspondiente (Property 13) se
implementa por separado en la tarea 7.2.
"""

from __future__ import annotations

import math

from mlops_core.promotion import should_promote
from mlops_core.types import Metrics, PromotionDecision


def test_promueve_primer_modelo_sin_champion() -> None:
    """RF6.4: sin champion previo se promueve el primer candidato."""
    candidate = Metrics(mae=100.0, rmse=200.0)
    decision = should_promote(candidate, champion=None)

    assert isinstance(decision, PromotionDecision)
    assert decision.promote is True
    assert decision.mae_change_pct is None
    assert decision.rmse_change_pct is None
    assert "primer modelo" in decision.reason.lower()


def test_promueve_cuando_mae_mejora_y_rmse_no_empeora() -> None:
    """RF6.2/6.3: MAE mejora >= 3% y RMSE no empeora más de 1%."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=95.0, rmse=200.0)  # MAE -5%, RMSE 0%

    decision = should_promote(candidate, champion)

    assert decision.promote is True
    assert math.isclose(decision.mae_change_pct, 0.05)
    assert math.isclose(decision.rmse_change_pct, 0.0)


def test_promueve_en_el_umbral_exacto_de_mae() -> None:
    """RF6.2: una mejora de MAE exactamente igual al umbral promueve (>=)."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=97.0, rmse=200.0)  # MAE -3% exacto

    decision = should_promote(candidate, champion)

    assert decision.promote is True
    assert math.isclose(decision.mae_change_pct, 0.03)


def test_conserva_champion_cuando_mae_no_mejora_lo_suficiente() -> None:
    """RF6.5: mejora de MAE por debajo del umbral conserva champion."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=98.0, rmse=200.0)  # MAE -2% < 3%

    decision = should_promote(candidate, champion)

    assert decision.promote is False
    assert math.isclose(decision.mae_change_pct, 0.02)


def test_conserva_champion_cuando_mae_empeora() -> None:
    """RF6.5: si el MAE empeora (candidato mayor) no se promueve."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=110.0, rmse=200.0)  # MAE +10% peor

    decision = should_promote(candidate, champion)

    assert decision.promote is False
    assert decision.mae_change_pct < 0  # mejora negativa = empeoró


def test_guarda_rmse_bloquea_promocion_pese_a_mae_mejor() -> None:
    """RF6.3: aunque el MAE mejore, un RMSE que empeora > 1% bloquea."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=90.0, rmse=210.0)  # MAE -10% pero RMSE +5%

    decision = should_promote(candidate, champion)

    assert decision.promote is False
    assert math.isclose(decision.mae_change_pct, 0.10)
    assert math.isclose(decision.rmse_change_pct, 0.05)
    assert "rmse" in decision.reason.lower()


def test_guarda_rmse_permite_empeoramiento_en_el_umbral() -> None:
    """RF6.3: un empeoramiento de RMSE exactamente en el umbral se tolera."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=95.0, rmse=202.0)  # MAE -5%, RMSE +1% exacto

    decision = should_promote(candidate, champion)

    assert decision.promote is True
    assert math.isclose(decision.rmse_change_pct, 0.01)


def test_rmse_que_mejora_no_bloquea() -> None:
    """Un RMSE que mejora (candidato menor) nunca activa la guarda."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=95.0, rmse=180.0)  # MAE -5%, RMSE -10%

    decision = should_promote(candidate, champion)

    assert decision.promote is True
    assert decision.rmse_change_pct < 0


def test_umbrales_personalizados() -> None:
    """Los umbrales son configurables vía parámetros."""
    champion = Metrics(mae=100.0, rmse=200.0)
    candidate = Metrics(mae=96.0, rmse=200.0)  # MAE -4%

    # Con umbral del 5% no alcanza para promover.
    estricta = should_promote(candidate, champion, mae_improvement_pct=0.05)
    assert estricta.promote is False

    # Con umbral del 2% sí promueve.
    laxa = should_promote(candidate, champion, mae_improvement_pct=0.02)
    assert laxa.promote is True
