"""``mlops_core``: capa de lógica pura y testeable de la plataforma MLOps.

Este paquete extrae las reglas de negocio del sistema (ingesta, validación,
decisión de entrenamiento, promoción, ingeniería de variables y esquemas de
logging) a funciones puras sin dependencias de infraestructura, de modo que
puedan ejercitarse con pruebas basadas en propiedades y reutilizarse tanto en
las tareas de Airflow como en la API de inferencia.

Los módulos (``ingest``, ``validation``, ``decision``, ``promotion``,
``features``, ``logging_schema``) se irán implementando en tareas posteriores.
Aquí se exponen únicamente los tipos compartidos.
"""

from __future__ import annotations

from mlops_core.types import (
    AuditEvent,
    Decision,
    DecisionInputs,
    DecisionType,
    DriftResult,
    ExperimentPayload,
    InferenceEvent,
    InferenceStatus,
    Metrics,
    NumericStats,
    PromotionDecision,
    QualityResult,
    SchemaResult,
)

__all__ = [
    "AuditEvent",
    "Decision",
    "DecisionInputs",
    "DecisionType",
    "DriftResult",
    "ExperimentPayload",
    "InferenceEvent",
    "InferenceStatus",
    "Metrics",
    "NumericStats",
    "PromotionDecision",
    "QualityResult",
    "SchemaResult",
]
