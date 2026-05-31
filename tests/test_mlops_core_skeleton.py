"""Pruebas de humo del esqueleto de ``mlops_core`` (tarea 1.1).

Verifican que el paquete y sus módulos importan limpiamente y que los tipos
compartidos están disponibles y se comportan como se espera. Las reglas de
negocio se prueban en tareas posteriores.
"""

from __future__ import annotations

import importlib

import pytest

from mlops_core import (
    Decision,
    DecisionInputs,
    DecisionType,
    DriftResult,
    Metrics,
    NumericStats,
    PromotionDecision,
    QualityResult,
    SchemaResult,
)


@pytest.mark.parametrize(
    "module_name",
    [
        "mlops_core",
        "mlops_core.types",
        "mlops_core.ingest",
        "mlops_core.validation",
        "mlops_core.decision",
        "mlops_core.promotion",
        "mlops_core.features",
        "mlops_core.logging_schema",
    ],
)
def test_modules_import_cleanly(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_decision_inputs_construction() -> None:
    inputs = DecisionInputs(
        total_records=1500,
        inserted_records=200,
        drift_detected=True,
        quality_valid=True,
        new_categories_count=3,
    )
    assert inputs.total_records == 1500
    assert inputs.inserted_records == 200
    assert inputs.drift_detected is True


def test_decision_should_train_property() -> None:
    train = Decision(decision=DecisionType.TRAIN, reason="drift detected")
    skip = Decision(decision=DecisionType.SKIP, reason="not enough data")
    assert train.should_train is True
    assert skip.should_train is False


def test_schema_and_quality_result_defaults() -> None:
    schema = SchemaResult(valid=True)
    quality = QualityResult(valid=False, issues=["bad price"])
    assert schema.missing_columns == []
    assert schema.extra_columns == []
    assert schema.type_mismatches == []
    assert quality.valid is False
    assert quality.issues == ["bad price"]


def test_drift_and_numeric_stats() -> None:
    stats = NumericStats(means={"price": 100.0, "house_size": 1500.0})
    drift = DriftResult(drift_detected=True, drifted_columns=["price"])
    assert stats.means["price"] == 100.0
    assert drift.drift_detected is True
    assert drift.new_categories == {}


def test_metrics_and_promotion_decision() -> None:
    candidate = Metrics(mae=10.0, rmse=20.0)
    decision = PromotionDecision(promote=True, reason="first model")
    assert candidate.mae == 10.0
    assert candidate.rmse == 20.0
    assert decision.promote is True
    assert decision.mae_change_pct is None
