"""Lógica pura de decisión automática de entrenamiento (RF4).

Responsabilidad:
- ``decide_training(inp)``: devuelve TRAIN/SKIP con motivo a partir de reglas
  técnicas (drift, volumen, suficiencia, nuevas categorías, columnas nuevas),
  excluyendo la periodicidad y el conteo bruto de lotes.
"""

from __future__ import annotations

from .types import Decision, DecisionInputs, DecisionType


def decide_training(
    inp: DecisionInputs,
    min_records: int = 1000,
    min_volume_pct: float = 0.10,
    min_new_categories: int = 5,
    min_new_columns: int = 2,
) -> Decision:
    """Decide entre entrenar (TRAIN) u omitir (SKIP) según reglas técnicas.

    No usa periodicidad ni el conteo bruto de lotes como criterio (RF4.3).

    Reglas (en orden de evaluación):
      - SKIP si ``total_records < min_records`` (datos insuficientes).
      - SKIP si ``quality_valid`` es ``False`` (lote no apto).
      - TRAIN si hay drift en variables numéricas.
      - TRAIN si la proporción de volumen nuevo >= 10%.
      - TRAIN si hay >= 5 categorías nuevas en al menos una columna.
      - TRAIN si hay >= 2 columnas nuevas en el esquema.
      - SKIP en caso contrario.

    Args:
        inp: Señales técnicas para la decisión.
        min_records: Umbral de suficiencia de datos.
        min_volume_pct: Proporción mínima de registros nuevos (10%).
        min_new_categories: Categorías nuevas mínimas para reentrenar (5).
        min_new_columns: Columnas nuevas mínimas en el esquema para reentrenar (2).

    Returns:
        Una ``Decision`` con el tipo (TRAIN/SKIP) y un motivo legible.
    """
    # Regla 1: señales técnicas que gatillan reentrenamiento.
    if inp.drift_detected:
        return Decision(
            decision=DecisionType.TRAIN,
            reason="Drift detectado en las variables numéricas del lote.",
        )

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

    if inp.new_columns_count >= min_new_columns:
        return Decision(
            decision=DecisionType.TRAIN,
            reason=(
                f"Columnas nuevas en el esquema: "
                f"new_columns_count={inp.new_columns_count} "
                f">= min_new_columns={min_new_columns}."
            ),
        )

    # Ninguna señal técnica se cumple → SKIP.
    return Decision(
        decision=DecisionType.SKIP,
        reason=(
            "Sin señales técnicas para reentrenar: sin drift, "
            f"proporción de volumen nuevo={volume_pct:.4f} < {min_volume_pct}, "
            f"categorías nuevas={inp.new_categories_count} < {min_new_categories}, "
            f"columnas nuevas={inp.new_columns_count} < {min_new_columns}."
        ),
    )
