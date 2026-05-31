import os
import sys
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sqlalchemy import create_engine, text
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
from config import POSTGRES_CLEAN_CONN, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, PROMOTION_MAE_IMPROVEMENT_PCT

logger = logging.getLogger(__name__)


# --- Import shim para `mlops_core` -------------------------------------------
# El DAG agrega `tasks/` a sys.path (para `from config import ...`), pero NO la
# raíz del repositorio donde vive el paquete `mlops_core`. Este shim añade la
# raíz al final de sys.path (para no ensombrecer librerías instaladas como
# `mlflow`) solo si `mlops_core` no es importable todavía. Nunca rompe la
# importación de `config` ni falla la carga del módulo si el paquete no existe.
def _ensure_mlops_core_importable():
    try:
        import mlops_core  # noqa: F401
        return
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # tasks -> dags -> airflow -> <repo root>
        os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir)),
        # Puntos de montaje habituales en contenedores de Airflow.
        "/opt/airflow",
        "/opt/airflow/repo",
        "/opt/airflow/dags/repo",
    ]
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "mlops_core")) and path not in sys.path:
            sys.path.append(path)


_ensure_mlops_core_importable()

try:
    from mlops_core.features import fit_preprocessor
    from mlops_core.logging_schema import build_experiment_payload
    from mlops_core.promotion import should_promote
    from mlops_core.types import Metrics
    _MLOPS_CORE_AVAILABLE = True
except Exception:  # pragma: no cover - mlops_core no disponible en el entorno
    _MLOPS_CORE_AVAILABLE = False
    fit_preprocessor = None  # type: ignore
    build_experiment_payload = None  # type: ignore
    should_promote = None  # type: ignore
    Metrics = None  # type: ignore

def train_and_promote(**context):
    ti = context['ti']
    train_reason = ti.xcom_pull(key='train_reason', task_ids='decide_training')
    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()

    engine = create_engine(POSTGRES_CLEAN_CONN)
    df_train = pd.read_sql("SELECT * FROM clean_properties WHERE split='train'", engine)
    df_val   = pd.read_sql("SELECT * FROM clean_properties WHERE split='val'", engine)
    df_test  = pd.read_sql("SELECT * FROM clean_properties WHERE split='test'", engine)
    engine.dispose()

    feature_cols = ["brokered_by", "status", "bed", "bath", "acre_lot",
                    "street", "city", "state", "zip_code", "house_size", "prev_sold_year"]
    target = "price"

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target]
    X_val   = df_val[feature_cols].fillna(0)
    y_val   = df_val[target]
    X_test  = df_test[feature_cols].fillna(0)
    y_test  = df_test[target]

    candidates = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    best_run_id = None
    best_val_mae = float("inf")
    best_model_name = None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"batch_{batch_number}_{name}") as run:
            model.fit(X_train, y_train)
            val_preds  = model.predict(X_val)
            test_preds = model.predict(X_test)
            val_mae  = mean_absolute_error(y_val, val_preds)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            val_r2   = r2_score(y_val, val_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            test_rmse= np.sqrt(mean_squared_error(y_test, test_preds))
            test_r2  = r2_score(y_test, test_preds)
            mlflow.log_params({"model": name, "batch_number": batch_number, "train_reason": train_reason})
            mlflow.log_metrics({
                "val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2,
                "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2,
                "train_size": len(X_train), "val_size": len(X_val)
            })
            mlflow.sklearn.log_model(model, "model", registered_model_name=MLFLOW_MODEL_NAME)
            logger.info(f"{name}: val_mae={val_mae:.2f}, val_rmse={val_rmse:.2f}, val_r2={val_r2:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run.info.run_id
                best_model_name = name

    logger.info(f"Best candidate: {best_model_name} with val_mae={best_val_mae:.2f}")

    promoted = False
    promotion_reason = ""
    try:
        champion = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
        champ_run = client.get_run(champion.run_id)
        champ_mae = champ_run.data.metrics.get("val_mae", float("inf"))
        improvement = (champ_mae - best_val_mae) / champ_mae
        if improvement >= PROMOTION_MAE_IMPROVEMENT_PCT:
            promoted = True
            promotion_reason = f"MAE improved {improvement:.2%} over champion ({champ_mae:.2f} -> {best_val_mae:.2f})"
        else:
            promotion_reason = f"No improvement: candidate MAE {best_val_mae:.2f} vs champion {champ_mae:.2f}"
    except Exception:
        promoted = True
        promotion_reason = "No existing champion, promoting first model"

    if promoted:
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        best_version = None
        for v in versions:
            if v.run_id == best_run_id:
                best_version = v.version
                break
        if best_version:
            client.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", best_version)
            logger.info(f"Promoted version {best_version} as champion: {promotion_reason}")

    ti.xcom_push(key='promoted', value=promoted)
    ti.xcom_push(key='promotion_reason', value=promotion_reason)
    ti.xcom_push(key='best_val_mae', value=best_val_mae)
    ti.xcom_push(key='best_model_name', value=best_model_name)

# --- Tarea 7.3: train_candidates (entrena + registra en MLflow) --------------

#: Variables predictoras usadas por los candidatos (coinciden con clean_properties).
FEATURE_COLS = [
    "brokered_by", "status", "bed", "bath", "acre_lot",
    "street", "city", "state", "zip_code", "house_size", "prev_sold_year",
]
TARGET_COL = "price"


def _get_code_commit():
    """Lee el commit de código desde variables de entorno (RF5.5, RF11.2).

    Devuelve el primer valor no vacío entre varias variables habituales de CI/CD,
    o ``None`` si ninguna está definida.
    """
    for var in ("GIT_COMMIT", "MLFLOW_GIT_COMMIT", "SOURCE_COMMIT", "GITHUB_SHA"):
        value = os.getenv(var)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _model_params(name, model):
    """Construye un dict plano y serializable de hiperparámetros del modelo (RF5.4)."""
    params = {"model": name}
    try:
        raw = model.get_params(deep=True)
        for key, value in raw.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                params[key] = value
    except Exception:  # pragma: no cover - get_params siempre disponible en sklearn
        pass
    return params


def _capture_preprocessing_params(df_train):
    """Captura parámetros de preprocesamiento serializables para MLflow (RF5.4).

    Delega en ``mlops_core.features.fit_preprocessor`` cuando está disponible;
    nunca interrumpe el entrenamiento si la captura falla.
    """
    if not _MLOPS_CORE_AVAILABLE or fit_preprocessor is None:
        return {}
    try:
        params = fit_preprocessor(df_train)
        return params.to_dict()
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning(f"No se pudieron capturar parámetros de preprocesamiento: {exc}")
        return {}


def _log_performance_artifacts(model, name, X_eval, y_true, y_pred, feature_cols):
    """Registra artefactos de desempeño en MLflow (RF5.6).

    Genera, con backend no interactivo (Agg): gráfico real vs predicho,
    distribución de errores e importancia de variables. Cada artefacto se
    protege con try/except para que el entrenamiento nunca falle por graficar.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend no interactivo (sin display)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib ausente
        logger.warning(f"matplotlib no disponible, se omiten artefactos: {exc}")
        return

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true

    # 1) Real vs predicho
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, s=8, alpha=0.4)
        lo = float(min(y_true.min(), y_pred.min())) if len(y_true) else 0.0
        hi = float(max(y_true.max(), y_pred.max())) if len(y_true) else 1.0
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Precio real")
        ax.set_ylabel("Precio predicho")
        ax.set_title(f"Real vs Predicho - {name}")
        mlflow.log_figure(fig, "plots/real_vs_predicted.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning(f"No se pudo registrar real_vs_predicted para {name}: {exc}")

    # 2) Distribución de errores
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(errors, bins=50, color="steelblue", alpha=0.8)
        ax.set_xlabel("Error (predicho - real)")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de errores - {name}")
        mlflow.log_figure(fig, "plots/error_distribution.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning(f"No se pudo registrar error_distribution para {name}: {exc}")

    # 3) Importancia de variables (feature_importances_ o |coef_| de Ridge)
    try:
        importances = None
        estimator = model
        if isinstance(model, Pipeline):
            estimator = model.named_steps.get("model", model)
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            importances = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()

        if importances is not None and len(importances) == len(feature_cols):
            order = np.argsort(importances)[::-1]
            sorted_feats = [feature_cols[i] for i in order]
            sorted_vals = importances[order]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(sorted_feats)), sorted_vals[::-1], color="darkorange")
            ax.set_yticks(range(len(sorted_feats)))
            ax.set_yticklabels(sorted_feats[::-1])
            ax.set_xlabel("Importancia")
            ax.set_title(f"Importancia de variables - {name}")
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/feature_importance.png")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning(f"No se pudo registrar feature_importance para {name}: {exc}")


def train_candidates(**context):
    """Entrena candidatos y los registra en MLflow (RF5.1-5.7, RF11.2, RF11.3).

    Entrena Ridge, RandomForest y GradientBoosting sobre CLEAN_DATA; para cada
    candidato registra en MLflow: parámetros del modelo y del preprocesamiento
    (RF5.4), métricas val/test (MAE/RMSE/R2) (RF5.3), el commit de código
    (RF5.5), artefactos de desempeño (RF5.6) y el modelo serializado con
    ``registered_model_name`` (RF5.7). Selecciona el mejor candidato por
    ``val_mae`` y empuja los XComs que necesitan las tareas de comparación,
    promoción y auditoría.
    """
    ti = context['ti']
    train_reason = ti.xcom_pull(key='train_reason', task_ids='decide_training')
    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')
    code_commit = _get_code_commit()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    engine = create_engine(POSTGRES_CLEAN_CONN)
    df_train = pd.read_sql("SELECT * FROM clean_properties WHERE split='train'", engine)
    df_val   = pd.read_sql("SELECT * FROM clean_properties WHERE split='val'", engine)
    df_test  = pd.read_sql("SELECT * FROM clean_properties WHERE split='test'", engine)
    engine.dispose()

    if df_train.empty or df_val.empty:
        logger.warning("Sin datos suficientes de train/val en clean_properties; se omite el entrenamiento.")
        ti.xcom_push(key='best_run_id', value=None)
        ti.xcom_push(key='best_val_mae', value=None)
        ti.xcom_push(key='best_model_name', value=None)
        ti.xcom_push(key='best_test_mae', value=None)
        ti.xcom_push(key='best_val_rmse', value=None)
        ti.xcom_push(key='code_commit', value=code_commit)
        return

    X_train, y_train = df_train[FEATURE_COLS].fillna(0), df_train[TARGET_COL]
    X_val,   y_val   = df_val[FEATURE_COLS].fillna(0),   df_val[TARGET_COL]
    X_test,  y_test  = df_test[FEATURE_COLS].fillna(0),  df_test[TARGET_COL]

    # Lotes de origen presentes en los datos limpios (trazabilidad RF5.2/RF11.1).
    try:
        batch_numbers = sorted(
            int(b) for b in pd.concat([df_train, df_val, df_test])["batch_number"].dropna().unique()
        )
    except Exception:
        batch_numbers = [int(batch_number)] if batch_number is not None else []

    preprocessing_params = _capture_preprocessing_params(df_train)

    candidates = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    best_run_id = None
    best_val_mae = float("inf")
    best_val_rmse = None
    best_test_mae = None
    best_model_name = None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"batch_{batch_number}_{name}") as run:
            model.fit(X_train, y_train)
            val_preds  = model.predict(X_val)
            test_preds = model.predict(X_test) if len(X_test) else np.array([])

            val_mae  = float(mean_absolute_error(y_val, val_preds))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            val_r2   = float(r2_score(y_val, val_preds))
            if len(X_test):
                test_mae  = float(mean_absolute_error(y_test, test_preds))
                test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
                test_r2   = float(r2_score(y_test, test_preds))
            else:
                test_mae = test_rmse = test_r2 = float("nan")

            metrics = {
                "val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2,
                "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2,
                "train_size": float(len(X_train)), "val_size": float(len(X_val)),
            }
            model_params = _model_params(name, model)
            model_params["batch_number"] = batch_number
            model_params["train_reason"] = train_reason

            # Payload estructurado del experimento (RF5.2-5.5) vía mlops_core.
            if _MLOPS_CORE_AVAILABLE and build_experiment_payload is not None:
                try:
                    payload = build_experiment_payload(
                        batch_numbers=batch_numbers,
                        train_reason=train_reason or "",
                        model_params=model_params,
                        preprocessing_params=preprocessing_params,
                        metrics={k: v for k, v in metrics.items() if v == v},  # excluye NaN
                        code_commit=code_commit,
                    )
                    model_params = dict(payload.model_params)
                    preprocessing_log = dict(payload.preprocessing_params)
                    metrics_log = dict(payload.metrics)
                    payload_commit = payload.code_commit
                except Exception as exc:  # pragma: no cover - defensivo
                    logger.warning(f"No se pudo construir el payload de experimento: {exc}")
                    preprocessing_log = preprocessing_params
                    metrics_log = {k: v for k, v in metrics.items() if v == v}
                    payload_commit = code_commit
            else:
                preprocessing_log = preprocessing_params
                metrics_log = {k: v for k, v in metrics.items() if v == v}
                payload_commit = code_commit

            # Parámetros del modelo y del preprocesamiento (RF5.4).
            mlflow.log_params(model_params)
            if preprocessing_log:
                mlflow.log_params({f"prep_{k}": v for k, v in preprocessing_log.items()})

            # Lotes usados y motivo (RF5.2).
            mlflow.set_tag("batch_numbers", ",".join(str(b) for b in batch_numbers))
            if train_reason:
                mlflow.set_tag("train_reason", str(train_reason))

            # Commit de código para reproducibilidad (RF5.5, RF11.2).
            if payload_commit:
                mlflow.set_tag("code_commit", payload_commit)
                mlflow.set_tag("mlflow.source.git.commit", payload_commit)
                mlflow.log_param("code_commit", payload_commit)

            # Métricas val/test (RF5.3).
            mlflow.log_metrics(metrics_log)

            # Artefactos de desempeño (RF5.6).
            _log_performance_artifacts(model, name, X_val, y_val, val_preds, FEATURE_COLS)

            # Modelo serializado + registro en el Model Registry (RF5.7).
            mlflow.sklearn.log_model(model, "model", registered_model_name=MLFLOW_MODEL_NAME)

            logger.info(
                f"{name}: val_mae={val_mae:.2f}, val_rmse={val_rmse:.2f}, val_r2={val_r2:.4f}"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_test_mae = test_mae
                best_run_id = run.info.run_id
                best_model_name = name

    logger.info(f"Mejor candidato: {best_model_name} con val_mae={best_val_mae:.2f}")

    # XComs para compare_models / promote_or_reject / log_result (tareas 7.4 y 7.5).
    ti.xcom_push(key='best_run_id', value=best_run_id)
    ti.xcom_push(key='best_val_mae', value=best_val_mae)
    ti.xcom_push(key='best_model_name', value=best_model_name)
    ti.xcom_push(key='best_val_rmse', value=best_val_rmse)
    ti.xcom_push(key='best_test_mae', value=best_test_mae)
    ti.xcom_push(key='code_commit', value=code_commit)
    ti.xcom_push(key='batch_numbers', value=batch_numbers)


# --- Tarea 7.4: compare_models + promote_or_reject ---------------------------

#: Empeoramiento máximo tolerado de RMSE frente al champion (RF6.3). Espejo del
#: valor por defecto de ``mlops_core.promotion.should_promote``.
PROMOTION_RMSE_MAX_WORSENING_PCT = 0.01


def _resolve_candidate_version(client, run_id):
    """Resuelve la versión del modelo registrado asociada a ``run_id`` (RF6.6).

    Recorre ``search_model_versions`` para ``MLFLOW_MODEL_NAME`` y devuelve la
    ``version`` cuyo ``run_id`` coincide con el del mejor candidato, o ``None``
    si no se encuentra.
    """
    if not run_id:
        return None
    try:
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning(f"No se pudieron listar versiones del modelo: {exc}")
        return None
    for v in versions:
        if v.run_id == run_id:
            return v.version
    return None


def compare_models(**context):
    """Compara el candidato contra el ``champion`` vigente en MLflow (RF6.1).

    Recupera las métricas del candidato (``best_run_id`` publicado por
    ``train_candidates``) y las del ``champion`` actual (alias ``champion`` del
    Model Registry). Empuja en XComs las métricas de ambos (``champion`` en
    ``None`` cuando no existe modelo de producción previo) y la versión del
    champion, para que ``promote_or_reject`` aplique la regla de promoción.
    """
    ti = context['ti']
    best_run_id = ti.xcom_pull(key='best_run_id', task_ids='train_candidates')
    best_val_mae = ti.xcom_pull(key='best_val_mae', task_ids='train_candidates')
    best_val_rmse = ti.xcom_pull(key='best_val_rmse', task_ids='train_candidates')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # --- Métricas del candidato -------------------------------------------------
    # Se prefieren los XComs (ya calculados por train_candidates); si faltan, se
    # recuperan desde el run de MLflow por ``best_run_id``.
    candidate_mae = best_val_mae
    candidate_rmse = best_val_rmse
    if best_run_id and (candidate_mae is None or candidate_rmse is None):
        try:
            run = client.get_run(best_run_id)
            metrics = run.data.metrics
            if candidate_mae is None:
                candidate_mae = metrics.get("val_mae")
            if candidate_rmse is None:
                candidate_rmse = metrics.get("val_rmse")
        except Exception as exc:  # pragma: no cover - defensivo
            logger.warning(f"No se pudieron leer métricas del candidato {best_run_id}: {exc}")

    # --- Métricas del champion vigente ------------------------------------------
    champion_mae = None
    champion_rmse = None
    champion_version = None
    try:
        champion = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
        champion_version = champion.version
        champ_run = client.get_run(champion.run_id)
        champ_metrics = champ_run.data.metrics
        champion_mae = champ_metrics.get("val_mae")
        champion_rmse = champ_metrics.get("val_rmse")
        logger.info(
            f"Champion vigente: versión {champion_version} "
            f"(val_mae={champion_mae}, val_rmse={champion_rmse})"
        )
    except Exception:
        logger.info("No existe champion previo; se promoverá el primer candidato.")

    logger.info(
        f"Comparación candidato (mae={candidate_mae}, rmse={candidate_rmse}) "
        f"vs champion (mae={champion_mae}, rmse={champion_rmse})"
    )

    # XComs para promote_or_reject.
    ti.xcom_push(key='candidate_mae', value=candidate_mae)
    ti.xcom_push(key='candidate_rmse', value=candidate_rmse)
    ti.xcom_push(key='champion_mae', value=champion_mae)
    ti.xcom_push(key='champion_rmse', value=champion_rmse)
    ti.xcom_push(key='champion_version', value=champion_version)


def promote_or_reject(**context):
    """Aplica la regla de promoción y reasigna el alias ``champion`` (RF6.2-6.6).

    Usa ``mlops_core.promotion.should_promote`` con las métricas publicadas por
    ``compare_models``. Reasigna el alias ``champion`` en el Model Registry
    **solo** cuando la promoción se aprueba, y **nunca** en caso contrario
    (RF6.5/RF6.6). Publica ``promoted`` (bool) y ``promotion_reason`` (incluye el
    cambio de desempeño frente a producción) para la auditoría (tarea 7.5).
    """
    ti = context['ti']
    best_run_id = ti.xcom_pull(key='best_run_id', task_ids='train_candidates')
    candidate_mae = ti.xcom_pull(key='candidate_mae', task_ids='compare_models')
    candidate_rmse = ti.xcom_pull(key='candidate_rmse', task_ids='compare_models')
    champion_mae = ti.xcom_pull(key='champion_mae', task_ids='compare_models')
    champion_rmse = ti.xcom_pull(key='champion_rmse', task_ids='compare_models')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Sin candidato válido no hay nada que promover (p. ej. entrenamiento omitido).
    if candidate_mae is None or candidate_rmse is None:
        promoted = False
        promotion_reason = "Sin candidato válido para comparar; no se promueve."
        logger.warning(promotion_reason)
        ti.xcom_push(key='promoted', value=promoted)
        ti.xcom_push(key='promotion_reason', value=promotion_reason)
        return

    if not _MLOPS_CORE_AVAILABLE or should_promote is None or Metrics is None:
        raise RuntimeError(
            "mlops_core no está disponible; no se puede evaluar la promoción."
        )

    candidate = Metrics(mae=float(candidate_mae), rmse=float(candidate_rmse))
    champion = None
    if champion_mae is not None and champion_rmse is not None:
        champion = Metrics(mae=float(champion_mae), rmse=float(champion_rmse))

    decision = should_promote(
        candidate,
        champion,
        mae_improvement_pct=PROMOTION_MAE_IMPROVEMENT_PCT,
        rmse_max_worsening_pct=PROMOTION_RMSE_MAX_WORSENING_PCT,
    )
    promoted = bool(decision.promote)
    promotion_reason = decision.reason

    # RF6.5/RF6.6: reasignar el alias `champion` SOLO si la promoción se aprueba.
    if promoted:
        best_version = _resolve_candidate_version(client, best_run_id)
        if best_version is not None:
            client.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", best_version)
            logger.info(
                f"Promovida versión {best_version} como champion: {promotion_reason}"
            )
        else:
            # No se pudo resolver la versión: no se reasigna el alias y se refleja
            # en el motivo para la auditoría.
            promoted = False
            promotion_reason = (
                f"{promotion_reason} (no se encontró la versión del candidato "
                f"para run_id={best_run_id}; alias champion sin cambios)"
            )
            logger.warning(promotion_reason)
    else:
        logger.info(f"No se promueve; se conserva el champion: {promotion_reason}")

    ti.xcom_push(key='promoted', value=promoted)
    ti.xcom_push(key='promotion_reason', value=promotion_reason)


def skip_training(**context):
    ti = context['ti']
    reason = ti.xcom_pull(key='train_reason', task_ids='decide_training')
    logger.info(f"Training skipped: {reason}")


# --- Tarea 7.5: log_result (adaptador de auditoría) --------------------------

#: Conexión a la base de auditoría (mlops_db). Se deriva de POSTGRES_CLEAN_CONN
#: reemplazando el nombre de la base.
_POSTGRES_MLOPS_CONN = POSTGRES_CLEAN_CONN.replace("/clean_db", "/mlops_db")

#: DDL para crear/ampliar la tabla training_history con las columnas nuevas
#: (mlflow_run_id, mlflow_model_version, code_commit) requeridas por RF9.4,
#: RF11.1, RF11.2.
_TRAINING_HISTORY_DDL = """\
CREATE TABLE IF NOT EXISTS training_history (
    id SERIAL PRIMARY KEY,
    batch_number INTEGER,
    trained BOOLEAN,
    train_reason TEXT,
    promoted BOOLEAN,
    promotion_reason TEXT,
    best_model VARCHAR(50),
    best_val_mae FLOAT,
    mlflow_run_id VARCHAR(64),
    mlflow_model_version VARCHAR(20),
    code_commit VARCHAR(64),
    logged_at TIMESTAMP DEFAULT NOW()
);
"""

#: Sentencias ALTER TABLE para añadir las columnas nuevas si la tabla ya existe
#: sin ellas (migración idempotente).
_TRAINING_HISTORY_MIGRATIONS = [
    "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS mlflow_run_id VARCHAR(64);",
    "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS mlflow_model_version VARCHAR(20);",
    "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS code_commit VARCHAR(64);",
]


def _resolve_model_version_for_run(run_id):
    """Resuelve la versión del modelo registrado asociada a un run_id.

    Consulta el Model Registry de MLflow para encontrar la versión cuyo
    ``run_id`` coincide con el del mejor candidato. Devuelve la versión como
    cadena o ``None`` si no se encuentra.
    """
    if not run_id:
        return None
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        for v in versions:
            if v.run_id == run_id:
                return str(v.version)
    except Exception as exc:
        logger.warning(f"No se pudo resolver la versión del modelo para run_id={run_id}: {exc}")
    return None


def log_result(**context):
    """Registra la decisión de entrenamiento/promoción en training_history (RF4.6, RF6.7, RF9.4, RF11.1, RF11.2).

    Este adaptador:
    1. Crea/amplía la tabla ``training_history`` en ``mlops_db`` con las columnas
       ``mlflow_run_id``, ``mlflow_model_version`` y ``code_commit``.
    2. Construye un ``AuditEvent`` vía ``mlops_core.logging_schema.build_audit_event``
       con la decisión, motivos y cambio de desempeño frente a producción.
    3. Persiste el evento en la tabla.

    Funciona tanto cuando se entrenó (rama train) como cuando se omitió
    (rama skip_training), gracias al ``trigger_rule="none_failed_min_one_success"``
    del DAG.
    """
    ti = context['ti']

    # --- Recopilar datos de las tareas previas --------------------------------
    batch_number = ti.xcom_pull(key='batch_number', task_ids='fetch_batch')
    train_decision = ti.xcom_pull(key='train_decision', task_ids='decide_training')
    train_reason = ti.xcom_pull(key='train_reason', task_ids='decide_training') or ""

    # Intentar leer de las tareas separadas (7.3/7.4); fallback a train_and_promote
    # para compatibilidad con el DAG anterior.
    promoted = ti.xcom_pull(key='promoted', task_ids='promote_or_reject')
    if promoted is None:
        promoted = ti.xcom_pull(key='promoted', task_ids='train_and_promote')
    promoted = bool(promoted) if promoted is not None else False

    promotion_reason = ti.xcom_pull(key='promotion_reason', task_ids='promote_or_reject')
    if promotion_reason is None:
        promotion_reason = ti.xcom_pull(key='promotion_reason', task_ids='train_and_promote')
    promotion_reason = promotion_reason or ""

    best_model_name = ti.xcom_pull(key='best_model_name', task_ids='train_candidates')
    if best_model_name is None:
        best_model_name = ti.xcom_pull(key='best_model_name', task_ids='train_and_promote')
    best_model_name = best_model_name or ""

    best_val_mae = ti.xcom_pull(key='best_val_mae', task_ids='train_candidates')
    if best_val_mae is None:
        best_val_mae = ti.xcom_pull(key='best_val_mae', task_ids='train_and_promote')

    best_run_id = ti.xcom_pull(key='best_run_id', task_ids='train_candidates')
    code_commit = ti.xcom_pull(key='code_commit', task_ids='train_candidates')

    # Determinar si hubo entrenamiento.
    trained = bool(train_decision) if train_decision is not None else (best_run_id is not None)

    # Resolver la versión del modelo en MLflow.
    mlflow_model_version = _resolve_model_version_for_run(best_run_id)

    # --- Construir el AuditEvent vía mlops_core (si disponible) ---------------
    if _MLOPS_CORE_AVAILABLE:
        try:
            from mlops_core.logging_schema import build_audit_event
            event = build_audit_event(
                batch_number=int(batch_number) if batch_number is not None else -1,
                trained=trained,
                train_reason=train_reason,
                promoted=promoted,
                promotion_reason=promotion_reason,
                best_model=best_model_name or None,
                best_val_mae=float(best_val_mae) if best_val_mae is not None else None,
                mlflow_run_id=best_run_id,
                mlflow_model_version=mlflow_model_version,
                code_commit=code_commit,
            )
            # Usar los valores normalizados del evento.
            batch_number = event.batch_number
            trained = event.trained
            train_reason = event.train_reason
            promoted = event.promoted
            promotion_reason = event.promotion_reason
            best_model_name = event.best_model
            best_val_mae = event.best_val_mae
            best_run_id = event.mlflow_run_id
            mlflow_model_version = event.mlflow_model_version
            code_commit = event.code_commit
        except Exception as exc:
            logger.warning(f"No se pudo construir AuditEvent vía mlops_core: {exc}")

    # --- Persistir en training_history ----------------------------------------
    engine = create_engine(_POSTGRES_MLOPS_CONN)
    try:
        with engine.begin() as conn:
            # Crear tabla si no existe.
            conn.execute(text(_TRAINING_HISTORY_DDL))
            # Migrar columnas nuevas si la tabla ya existía sin ellas.
            for migration in _TRAINING_HISTORY_MIGRATIONS:
                conn.execute(text(migration))
            # Insertar el registro de auditoría.
            conn.execute(text("""
                INSERT INTO training_history
                (batch_number, trained, train_reason, promoted, promotion_reason,
                 best_model, best_val_mae, mlflow_run_id, mlflow_model_version, code_commit)
                VALUES (:bn, :trained, :tr, :promoted, :pr, :bm, :mae, :run_id, :model_version, :commit)
            """), {
                "bn": batch_number,
                "trained": trained,
                "tr": train_reason,
                "promoted": promoted,
                "pr": promotion_reason,
                "bm": best_model_name,
                "mae": best_val_mae,
                "run_id": best_run_id,
                "model_version": mlflow_model_version,
                "commit": code_commit,
            })
    finally:
        engine.dispose()

    logger.info(
        f"log_result: batch={batch_number}, trained={trained}, promoted={promoted}, "
        f"model={best_model_name}, mae={best_val_mae}, "
        f"run_id={best_run_id}, version={mlflow_model_version}, commit={code_commit}"
    )
