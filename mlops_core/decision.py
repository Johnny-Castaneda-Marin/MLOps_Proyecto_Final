"""Lógica pura de decisión automática de entrenamiento (RF4).

Responsabilidad:
- ``decide_training(inp)``: devuelve TRAIN/SKIP con motivo a partir de reglas
  técnicas (drift, volumen, suficiencia, nuevas categorías), excluyendo la
  periodicidad y el conteo bruto de lotes.
"""

from __future__ import annotations

from .types import Decision, DecisionInputs, DecisionType


def decide_training(
    inp: DecisionInputs,
    min_records: int = 1000,
    min_volume_pct: float = 0.05,
    min_new_categories: int = 5,
) -> Decision:
    """Decide entre entrenar (TRAIN) u omitir (SKIP) según reglas técnicas.

    No usa periodicidad ni el conteo bruto de lotes como criterio (RF4.3).

    Reglas (en orden de evaluación):
      - SKIP si ``total_records < min_records`` (datos insuficientes, RF3.7/RF4.4).
      - SKIP si ``quality_valid`` es ``False`` (lote no apto, RF3.8/RF4.4).
      - TRAIN si hay drift, o la proporción de volumen nuevo
        (``inserted_records / total_records``) alcanza ``min_volume_pct``, o el
        número de categorías nuevas alcanza ``min_new_categories`` (RF3.6/RF4.2).
      - SKIP en caso contrario (ninguna señal técnica se cumple).

    Args:
        inp: Señales técnicas para la decisión.
        min_records: Umbral de suficiencia de datos (``MIN_RECORDS_TO_TRAIN``).
        min_volume_pct: Proporción mínima de volumen nuevo
            (``MIN_VOLUME_INCREASE_PCT``) para gatillar reentrenamiento.
        min_new_categories: Número mínimo de categorías nuevas para gatillar
            reentrenamiento.

    Returns:
        Una ``Decision`` con el tipo (TRAIN/SKIP) y un motivo legible.
    """
    # Regla 1: suficiencia de datos.
    if inp.total_records < min_records:
        return Decision(
            decision=DecisionType.SKIP,
            reason=(
                f"Datos insuficientes: total_records={inp.total_records} "
                f"< min_records={min_records}."
            ),
        )

    # Regla 2: calidad del lote.
    if not inp.quality_valid:
        return Decision(
            decision=DecisionType.SKIP,
            reason="Calidad del lote no apta para entrenamiento (quality_valid=False).",
        )

    # Regla 3: señales técnicas que gatillan reentrenamiento.
    if inp.drift_detected:
        return Decision(
            decision=DecisionType.TRAIN,
            reason="Drift detectado en las variables numéricas del lote.",
        )

    # ``total_records >= min_records`` aquí; si ``min_records`` fuese 0 podría ser
    # 0, así que se evita la división por cero.
    volume_pct = (
        inp.inserted_records / inp.total_records if inp.total_records > 0 else 0.0
    )
    if volume_pct >= min_volume_pct:
        return Decision(
            decision=DecisionType.TRAIN,
            reason=(
                f"Incremento de volumen suficiente: "
                f"proporción nueva={volume_pct:.4f} >= min_volume_pct={min_volume_pct}."
            ),
        )

    if inp.new_categories_count >= min_new_categories:
        return Decision(
            decision=DecisionType.TRAIN,
            reason=(
                f"Nuevas categorías suficientes: "
                f"new_categories_count={inp.new_categories_count} "
                f">= min_new_categories={min_new_categories}."
            ),
        )

    # Regla 4: ninguna señal técnica se cumple.
    return Decision(
        decision=DecisionType.SKIP,
        reason=(
            "Sin señales técnicas para reentrenar: sin drift, "
            f"proporción de volumen nuevo={volume_pct:.4f} < {min_volume_pct}, "
            f"categorías nuevas={inp.new_categories_count} < {min_new_categories}."
        ),
    )
