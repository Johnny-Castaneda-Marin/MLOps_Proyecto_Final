"""Configuración de la API de inferencia (RF7, RF8, RF10).

Todas las credenciales y URIs se leen de variables de entorno, inyectadas
mediante Kubernetes Secrets (RF14.1, RF14.2, RF14.3). Los valores por defecto
facilitan el desarrollo local.
"""

from __future__ import annotations

import os

# --- MLflow ---
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "http://mlflow-service:5000"
)
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "real_estate_champion")
MLFLOW_CHAMPION_ALIAS: str = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")

# --- Base de datos para inference_log (raw_db) ---
POSTGRES_RAW_CONN: str = os.getenv(
    "POSTGRES_RAW_CONN",
    "postgresql://mlops_user:mlops1234@postgres-service:5432/raw_db",
)

# --- Autenticación del endpoint /admin/reload (RF7.4) ---
ADMIN_TOKEN: str = os.getenv("API_ADMIN_TOKEN", "changeme-admin-token")

# --- Poller de recarga periódica ---
RELOAD_INTERVAL_SECONDS: int = int(os.getenv("RELOAD_INTERVAL_SECONDS", "60"))

# --- Nombre de la aplicación ---
APP_NAME: str = "mlops-real-estate-api"
APP_VERSION: str = "0.1.0"
