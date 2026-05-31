"""Lógica pura de promoción del modelo candidato (RF6).

Responsabilidad (tarea 7.1):
- ``should_promote(candidate, champion)``: regla de promoción basada en MAE con
  guarda de RMSE; promueve el primer modelo cuando no hay champion previo.

Convención de signos (importante):
- ``mae_change_pct`` representa la **mejora** del MAE frente al champion. Es
  positivo cuando el MAE del candidato es **menor** que el del champion
  (mejor): ``(champion.mae - candidate.mae) / champion.mae``.
- ``rmse_change_pct`` representa el **empeoramiento** del RMSE frente al
  champion. Es positivo cuando el RMSE del candidato es **mayor** que el del
  champion (peor): ``(candidate.rmse - champion.rmse) / champion.rmse``.

Con esta convención los umbrales se leen de forma natural: se promueve cuando
``mae_change_pct >= mae_improvement_pct`` (mejora suficiente) **y**
``rmse_change_pct <= rmse_max_worsening_pct`` (la guarda de RMSE no se viola).
"""

from __future__ import annotations

from typing import Optional

from mlops_core.types import Metrics, PromotionDecision


def _improvement_pct(baseline: float, value: float) -> float:
    """Mejora relativa de ``value`` respecto a ``baseline`` (positivo = mejor).

    Para métricas de error (MAE), "mejorar" significa disminuir; por eso un
    ``value`` menor que ``baseline`` produce un porcentaje positivo. Si la línea
    base es ``0`` no hay un porcentaje finito definido: se devuelve ``0.0``
    cuando ambos valores son ``0`` (sin cambio) y ``-inf`` cuando empeora, de
    modo que la comparación contra cualquier umbral positivo sea consistente.
    """

    if baseline == 0:
        return 0.0 if value == 0 else float("-inf")
    return (baseline - value) / baseline


def _worsening_pct(baseline: float, value: float) -> float:
    """Empeoramiento relativo de ``value`` respecto a ``baseline`` (positivo = peor).

    Para métricas de error (RMSE), "empeorar" significa aumentar; por eso un
    ``value`` mayor que ``baseline`` produce un porcentaje positivo. Si la línea
    base es ``0`` se devuelve ``0.0`` cuando no hay cambio y ``+inf`` cuando
    empeora, de modo que la guarda contra cualquier umbral finito se active.
    """

    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - baseline) / baseline


def should_promote(
    candidate: Metrics,
    champion: Optional[Metrics],
    mae_improvement_pct: float = 0.03,
    rmse_max_worsening_pct: float = 0.01,
) -> PromotionDecision:
    """Decide si el modelo candidato debe promoverse a producción (RF6).

    Reglas (RF6.2, RF6.3, RF6.4):
    - Si no existe champion previo, se promueve el primer candidato.
    - En caso contrario, se promueve si y solo si el MAE del candidato mejora al
      menos ``mae_improvement_pct`` respecto al champion **y** el RMSE no empeora
      más de ``rmse_max_worsening_pct``.
    - En cualquier otro caso se conserva el champion (``promote=False``), lo que
      implica que el alias ``champion`` no se reasigna (RF6.5).

    Args:
        candidate: Métricas del modelo recién entrenado.
        champion: Métricas del modelo de producción vigente, o ``None`` si no
            existe un champion previo.
        mae_improvement_pct: Mejora mínima de MAE exigida (fracción, p. ej.
            ``0.03`` = 3%).
        rmse_max_worsening_pct: Empeoramiento máximo de RMSE tolerado (fracción,
            p. ej. ``0.01`` = 1%).

    Returns:
        ``PromotionDecision`` con la decisión, el motivo y los cambios de
        desempeño (``mae_change_pct`` como mejora, ``rmse_change_pct`` como
        empeoramiento).
    """

    # RF6.4: sin champion previo, se promueve el primer candidato.
    if champion is None:
        return PromotionDecision(
            promote=True,
            reason="Promovido: primer modelo (no existe champion previo).",
            mae_change_pct=None,
            rmse_change_pct=None,
        )

    mae_change_pct = _improvement_pct(champion.mae, candidate.mae)
    rmse_change_pct = _worsening_pct(champion.rmse, candidate.rmse)

    # Comparaciones equivalentes a las basadas en porcentaje pero sin división,
    # robustas cuando la línea base del champion es 0.
    mae_improved = candidate.mae <= champion.mae * (1.0 - mae_improvement_pct)
    rmse_guard_ok = candidate.rmse <= champion.rmse * (1.0 + rmse_max_worsening_pct)

    if mae_improved and rmse_guard_ok:
        reason = (
            f"Promovido: MAE mejora {mae_change_pct:.2%} "
            f"(>= umbral {mae_improvement_pct:.2%}) y RMSE cambia "
            f"{rmse_change_pct:.2%} (<= umbral {rmse_max_worsening_pct:.2%})."
        )
        return PromotionDecision(
            promote=True,
            reason=reason,
            mae_change_pct=mae_change_pct,
            rmse_change_pct=rmse_change_pct,
        )

    # Conservar champion: detallar el o los motivos del rechazo.
    motivos = []
    if not mae_improved:
        motivos.append(
            f"MAE mejora {mae_change_pct:.2%} (< umbral {mae_improvement_pct:.2%})"
        )
    if not rmse_guard_ok:
        motivos.append(
            f"RMSE empeora {rmse_change_pct:.2%} (> umbral {rmse_max_worsening_pct:.2%})"
        )
    reason = "Conservado champion: " + "; ".join(motivos) + "."

    return PromotionDecision(
        promote=False,
        reason=reason,
        mae_change_pct=mae_change_pct,
        rmse_change_pct=rmse_change_pct,
    )
