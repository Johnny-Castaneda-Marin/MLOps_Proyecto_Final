import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from config import (
    POSTGRES_RAW_CONN,
    MIN_RECORDS_TO_TRAIN,
    MIN_VOLUME_INCREASE_PCT,
    DRIFT_THRESHOLD,
)


# --------------------------------------------------------------------------- #
# Shim de importación de ``mlops_core``.
#
# El DAG inserta ``tasks/`` en ``sys.path`` (no la raíz del repositorio), por lo
# que el paquete de lógica pura ``mlops_core`` no es importable por defecto desde
# este adaptador. Buscamos hacia arriba el directorio que contiene ``mlops_core``
# y lo añadimos a ``sys.path``. Esto resuelve tanto en el repo (raíz del proyecto)
# como en el contenedor de Airflow. No interfiere con ``from config import ...``,
# pues ``tasks/`` permanece en ``sys.path``.
# --------------------------------------------------------------------------- #
def _ensure_mlops_core_on_path() -> None:
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(candidate, "mlops_core")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    for env_root in (os.getenv("MLOPS_CORE_ROOT"), "/opt/airflow", "/opt/airflow/repo"):
        if env_root and os.path.isdir(os.path.join(env_root, "mlops_core")):
            if env_root not in sys.path:
                sys.path.insert(0, env_root)
            return


_ensure_mlops_core_on_path()

from mlops_core.types import NumericStats
from mlops_core.decision import decide_training as core_decide_training
from mlops_core.types import DecisionInputs
from mlops_core.validation import (
    EXPECTED_SCHEMA,
    detect_drift as core_detect_drift,
    validate_quality as core_validate_quality,
    validate_schema as core_validate_schema,
)

logger = logging.getLogger(__name__)

# Variables consideradas para drift numérico y nuevas categorías (RF3.3, RF3.4).
NUMERIC_COLS = ["price", "house_size", "bed", "bath"]
CATEGORICAL_COLS = ["status", "city", "state"]


def validate_schema(**context):
    """Adaptador: delega la validación de esquema en ``mlops_core.validation``.

    Lee el lote desde XCom, delega en ``core_validate_schema`` (que detecta
    columnas faltantes/adicionales y cambios de tipo, RF3.1) y empuja
    ``schema_valid`` / ``schema_issues``. Un lote vacío se marca como no válido
    sin interrumpir la ejecución (RF3.9).
    """
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    result = core_validate_schema(records or [], EXPECTED_SCHEMA)

    if result.valid:
        logger.info("Schema validation passed")
        ti.xcom_push(key='schema_valid', value=True)
        ti.xcom_push(key='schema_issues', value="none")
        return

    issues = []
    if result.missing_columns:
        issues.append(f"Missing columns: {result.missing_columns}")
    if result.extra_columns:
        issues.append(f"Extra columns: {result.extra_columns}")
    if result.type_mismatches:
        issues.append(f"Type mismatches: {result.type_mismatches}")
    issues_str = str(issues) if issues else "empty batch"
    logger.warning(f"Schema issues: {issues_str}")
    ti.xcom_push(key='schema_valid', value=False)
    ti.xcom_push(key='schema_issues', value=issues_str)


def validate_quality(**context):
    """Adaptador: delega la validación de calidad en ``mlops_core.validation``.

    Lee el lote desde XCom, delega en ``core_validate_quality`` (nulos por
    columna, duplicados, precios inválidos y proporción de precios nulos, RF3.2 /
    RF3.8) y empuja ``quality_valid`` / ``quality_issues`` / ``total_records``.
    Un lote vacío se marca como no apto sin interrumpir la ejecución (RF3.9).
    """
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch') or []
    result = core_validate_quality(records)
    total = len(records)

    logger.info(f"Quality check: {total} records, {len(result.issues)} issues")
    ti.xcom_push(key='quality_valid', value=result.valid)
    ti.xcom_push(key='quality_issues', value=str(result.issues) if result.issues else "none")
    ti.xcom_push(key='total_records', value=total)


def detect_drift(**context):
    """Adaptador: delega la detección de drift en ``mlops_core.validation``.

    Construye las estadísticas numéricas del lote nuevo (medias de ``price``,
    ``house_size``, ``bed``, ``bath``) y del histórico acumulado, y los conjuntos
    de categorías (``status``, ``city``, ``state``) del lote y del histórico
    (excluyendo el lote actual). Delega en ``core_detect_drift`` (RF3.3, RF3.4) y
    empuja ``drift_detected``, ``drift_details`` y el conteo de nuevas categorías
    en ``new_categories`` para la decisión de entrenamiento (RF4.2). Maneja el
    caso ``no_history`` sin interrumpir la ejecución.
    """
    ti = context['ti']
    records = ti.xcom_pull(key='records', task_ids='fetch_batch')
    if not records:
        ti.xcom_push(key='drift_detected', value=False)
        ti.xcom_push(key='drift_details', value="no_records")
        ti.xcom_push(key='new_categories', value=0)
        return

    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')
    df_new = pd.DataFrame(records)

    # Estadísticas del lote nuevo.
    new_means = {}
    for col in NUMERIC_COLS:
        if col in df_new.columns:
            mean = pd.to_numeric(df_new[col], errors='coerce').dropna().mean()
            if pd.notna(mean):
                new_means[col] = float(mean)
    new_categories = {
        col: set(df_new[col].dropna().unique()) if col in df_new.columns else set()
        for col in CATEGORICAL_COLS
    }

    # Estadísticas del histórico acumulado (excluye el lote actual por número).
    hist_means = {}
    hist_categories = {}
    engine = create_engine(POSTGRES_RAW_CONN)
    try:
        with engine.connect() as conn:
            avg_row = conn.execute(
                text(
                    "SELECT AVG(price), AVG(house_size), AVG(bed), AVG(bath) "
                    "FROM raw_properties WHERE batch_number IS DISTINCT FROM :bn"
                ),
                {"bn": batch_number},
            ).fetchone()
            if avg_row is not None:
                for col, value in zip(NUMERIC_COLS, avg_row):
                    if value is not None:
                        hist_means[col] = float(value)
            for col in CATEGORICAL_COLS:
                rows = conn.execute(
                    text(
                        f"SELECT DISTINCT {col} FROM raw_properties "
                        f"WHERE batch_number IS DISTINCT FROM :bn AND {col} IS NOT NULL"
                    ),
                    {"bn": batch_number},
                ).fetchall()
                hist_categories[col] = {row[0] for row in rows}
    except Exception as exc:
        logger.warning(f"Could not read history for drift detection: {exc}")
        hist_means = {}
        hist_categories = {}
    finally:
        engine.dispose()

    history_exists = bool(hist_means) or any(hist_categories.values())
    if not history_exists:
        logger.info("No historical data, skipping drift detection")
        ti.xcom_push(key='drift_detected', value=False)
        ti.xcom_push(key='drift_details', value="no_history")
        ti.xcom_push(key='new_categories', value=0)
        return

    result = core_detect_drift(
        new_stats=NumericStats(means=new_means),
        hist_stats=NumericStats(means=hist_means),
        new_categories=new_categories,
        hist_categories=hist_categories,
        drift_threshold=DRIFT_THRESHOLD,
    )

    new_categories_count = sum(len(values) for values in result.new_categories.values())
    logger.info(
        f"Drift detected: {result.drift_detected}, details: {result.details}, "
        f"new categories: {new_categories_count}"
    )
    ti.xcom_push(key='drift_detected', value=result.drift_detected)
    ti.xcom_push(key='drift_details', value=result.details)
    ti.xcom_push(key='new_categories', value=new_categories_count)
    ti.xcom_push(key='new_categories_info', value=str(result.new_categories))

def decide_training(**context):
    """Adaptador (BranchPythonOperator): delega la decisión en ``mlops_core.decision``.

    Construye ``DecisionInputs`` a partir de los XComs de tareas previas y delega
    la decisión técnica en ``core_decide_training`` (RF4.1, RF4.2, RF4.3):

      - ``total_records``: conteo acumulado de ``raw_properties`` (vía SQL COUNT).
      - ``inserted_records``: ``inserted_count`` empujado por ``store_raw``.
      - ``drift_detected``: resultado de ``detect_drift``.
      - ``quality_valid``: resultado de ``validate_quality``.
      - ``new_categories_count``: nuevas categorías contadas en ``detect_drift``.

    Bifurca a ``preprocess_data`` (entrenar) o ``skip_training`` (omitir) según la
    decisión, y propaga fielmente el motivo en ``train_reason`` además de
    ``train_decision`` (bool) para que las tareas aguas abajo —``train_and_promote``,
    ``skip_training`` y ``log_result``— registren la decisión y su motivo en la
    Tabla_Auditoria (RF4.4, RF4.5, RF4.6).
    """
    ti = context['ti']
    inserted = ti.xcom_pull(key='inserted_count', task_ids='store_raw') or 0
    drift = ti.xcom_pull(key='drift_detected', task_ids='detect_drift') or False
    quality_valid = ti.xcom_pull(key='quality_valid', task_ids='validate_quality')
    new_categories_count = ti.xcom_pull(key='new_categories', task_ids='detect_drift') or 0

    # Conteo acumulado de registros en bruto (mantiene el comportamiento actual).
    engine = create_engine(POSTGRES_RAW_CONN)
    try:
        with engine.connect() as conn:
            try:
                total = conn.execute(text("SELECT COUNT(*) FROM raw_properties")).fetchone()[0]
            except Exception:
                total = 0
    finally:
        engine.dispose()

    inputs = DecisionInputs(
        total_records=int(total or 0),
        inserted_records=int(inserted or 0),
        drift_detected=bool(drift),
        quality_valid=bool(quality_valid),
        new_categories_count=int(new_categories_count or 0),
    )
    decision = core_decide_training(
        inputs,
        min_records=MIN_RECORDS_TO_TRAIN,
        min_volume_pct=MIN_VOLUME_INCREASE_PCT,
    )

    # Propaga el motivo fielmente (incluye, cuando aplica, la adaptación de
    # esquema registrada como motivo, RF4.5).
    ti.xcom_push(key='train_decision', value=decision.should_train)
    ti.xcom_push(key='train_reason', value=decision.reason)

    if decision.should_train:
        logger.info(f"Decision: TRAIN — {decision.reason}")
        return 'preprocess_data'

    logger.info(f"Decision: SKIP — {decision.reason}")
    return 'skip_training'
