"""Tipos y dataclasses compartidos de la capa de lógica pura ``mlops_core``.

Este módulo concentra los tipos de resultado intercambiados entre los módulos
puros (``ingest``, ``validation``, ``decision``, ``promotion``, ``features``,
``logging_schema``) y los adaptadores de Airflow / FastAPI que los consumen.

No contiene dependencias de infraestructura (PostgreSQL, MLflow, HTTP), de modo
que estos tipos pueden importarse y ejercitarse de forma determinista en
pruebas basadas en propiedades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DecisionType(str, Enum):
    """Resultado posible de la decisión automática de entrenamiento."""

    TRAIN = "TRAIN"
    SKIP = "SKIP"


@dataclass(frozen=True)
class DecisionInputs:
    """Entradas técnicas para la decisión de entrenamiento (RF4).

    Excluye deliberadamente la periodicidad y el conteo bruto de lotes: la
    decisión depende únicamente de señales técnicas (RF4.3).
    """

    total_records: int
    inserted_records: int
    drift_detected: bool
    quality_valid: bool
    new_categories_count: int = 0


@dataclass(frozen=True)
class Decision:
    """Resultado de ``decide_training``: entrenar u omitir, con su motivo."""

    decision: DecisionType
    reason: str

    @property
    def should_train(self) -> bool:
        return self.decision is DecisionType.TRAIN


@dataclass(frozen=True)
class SchemaResult:
    """Resultado de la validación de esquema (RF3.1)."""

    valid: bool
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    type_mismatches: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualityResult:
    """Resultado de la validación de calidad del lote (RF3.2, RF3.8)."""

    valid: bool
    high_null_columns: List[str] = field(default_factory=list)
    duplicate_rows: int = 0
    invalid_prices: int = 0
    null_price_ratio: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class NumericStats:
    """Medias de las variables numéricas usadas para detección de drift (RF3.3).

    Las claves son nombres de columna y los valores la media observada.
    """

    means: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftResult:
    """Resultado de la detección de drift y nuevas categorías (RF3.3, RF3.4)."""

    drift_detected: bool
    drifted_columns: List[str] = field(default_factory=list)
    new_categories: Dict[str, List[str]] = field(default_factory=dict)
    details: str = ""


@dataclass(frozen=True)
class Metrics:
    """Métricas de regresión usadas en la comparación de modelos (RF6)."""

    mae: float
    rmse: float


@dataclass(frozen=True)
class PromotionDecision:
    """Resultado de la regla de promoción (RF6.2, RF6.3, RF6.4)."""

    promote: bool
    reason: str
    mae_change_pct: Optional[float] = None
    rmse_change_pct: Optional[float] = None


class InferenceStatus(str, Enum):
    """Estado de una solicitud de inferencia atendida por la API (RF8.2)."""

    OK = "ok"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Registro de la Tabla_Auditoria (``training_history``) por lote.

    Concentra la decisión de entrenamiento y su motivo, la decisión de
    promoción y su motivo (incluyendo el cambio de desempeño frente a
    producción), el mejor modelo y su MAE de validación, y los identificadores
    de reproducibilidad de MLflow / commit de código (RF4.6, RF6.7, RF5.5,
    RF9.4, RF11.1, RF11.2).
    """

    batch_number: int
    trained: bool
    train_reason: str
    promoted: bool
    promotion_reason: str
    best_model: Optional[str] = None
    best_val_mae: Optional[float] = None
    mlflow_run_id: Optional[str] = None
    mlflow_model_version: Optional[str] = None
    code_commit: Optional[str] = None
    logged_at: Optional[datetime] = None


@dataclass
class ExperimentPayload:
    """Payload del experimento de MLflow para un entrenamiento (RF5).

    Reúne los ``batch_number`` utilizados, el motivo del entrenamiento, los
    parámetros del modelo y del preprocesamiento, las métricas registradas y el
    commit de código asociado para reproducibilidad (RF5.2-RF5.5, RF11.2).
    """

    batch_numbers: List[int] = field(default_factory=list)
    train_reason: str = ""
    model_params: Dict[str, Any] = field(default_factory=dict)
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    code_commit: Optional[str] = None


@dataclass
class InferenceEvent:
    """Evento de inferencia registrado en ``inference_log`` (RF8.2, RF8.3).

    Incluye la marca de tiempo en UTC, los datos de entrada, la predicción, la
    versión del modelo servido, el estado (``ok``/``error``) y el mensaje de
    error. Cuando el estado es ``error`` el campo ``error`` es no vacío; cuando
    es ``ok`` el campo ``error`` es ``None``.
    """

    timestamp: datetime
    input_data: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[float] = None
    model_version: Optional[str] = None
    status: InferenceStatus = InferenceStatus.OK
    error: Optional[str] = None
