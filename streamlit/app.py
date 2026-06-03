"""Streamlit UI para la plataforma MLOps de predicción de precios de inmuebles (RF9).

Secciones:
- Inferencia: formulario de propiedad → POST /predict → predicción + model_version.
- Historial: lectura de training_history y visualización por lote.

La URL de la API se configura mediante la variable de entorno API_URL
(por defecto: http://api-service:8000).

La conexión a la base de datos para el historial se configura mediante
POSTGRES_CLEAN_CONN (por defecto: postgresql://mlops_user:mlops1234@postgres-service:5432/clean_db).
La tabla training_history reside en mlops_db (derivada de POSTGRES_CLEAN_CONN).
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

API_URL: str = os.getenv("API_URL", "http://api-service:8000")

POSTGRES_CLEAN_CONN: str = os.getenv(
    "POSTGRES_CLEAN_CONN",
    "postgresql://mlops_user:mlops1234@postgres-service:5432/clean_db",
)

# La tabla training_history reside en mlops_db (misma instancia PostgreSQL).
_POSTGRES_MLOPS_CONN: str = POSTGRES_CLEAN_CONN.replace("/clean_db", "/mlops_db")

# ---------------------------------------------------------------------------
# Página principal
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MLOps Real Estate - Predicción de Precios",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Predicción de Precios de Inmuebles")
st.markdown("Plataforma MLOps — Pontificia Universidad Javeriana")

# ---------------------------------------------------------------------------
# Sección de Inferencia (RF9.1, RF9.2)
# ---------------------------------------------------------------------------

st.header("Inferencia")
st.markdown(
    "Ingrese las características de la propiedad para obtener una predicción de precio."
)

with st.form("inference_form"):
    col1, col2 = st.columns(2)

    with col1:
        brokered_by = st.number_input(
            "Brokered By (ID del broker)",
            min_value=0.0,
            value=101.0,
            step=1.0,
            help="Identificador numérico del broker.",
        )
        status = st.selectbox(
            "Status",
            options=["for_sale", "ready_to_build", "sold"],
            index=0,
            help="Estado actual de la propiedad.",
        )
        bed = st.number_input(
            "Bedrooms",
            min_value=0.0,
            value=3.0,
            step=1.0,
            help="Número de habitaciones.",
        )
        bath = st.number_input(
            "Bathrooms",
            min_value=0.0,
            value=2.0,
            step=0.5,
            help="Número de baños.",
        )
        acre_lot = st.number_input(
            "Acre Lot",
            min_value=0.0,
            value=0.12,
            step=0.01,
            format="%.4f",
            help="Tamaño del lote en acres.",
        )
        street = st.number_input(
            "Street (ID)",
            min_value=0.0,
            value=123.0,
            step=1.0,
            help="Identificador numérico de la calle.",
        )

    with col2:
        city = st.text_input(
            "City",
            value="Austin",
            help="Ciudad donde se ubica la propiedad.",
        )
        state = st.text_input(
            "State",
            value="Texas",
            help="Estado donde se ubica la propiedad.",
        )
        zip_code = st.number_input(
            "Zip Code",
            min_value=0.0,
            value=78701.0,
            step=1.0,
            help="Código postal.",
        )
        house_size = st.number_input(
            "House Size (sqft)",
            min_value=0.0,
            value=1800.0,
            step=10.0,
            help="Tamaño de la vivienda en pies cuadrados.",
        )
        prev_sold_year = st.number_input(
            "Previous Sold Year",
            min_value=1900.0,
            max_value=2100.0,
            value=2018.0,
            step=1.0,
            help="Año en que se vendió previamente la propiedad.",
        )

    submitted = st.form_submit_button("🔮 Predecir Precio", use_container_width=True)

# ---------------------------------------------------------------------------
# Envío del formulario y visualización de resultado (RF9.2)
# ---------------------------------------------------------------------------

if submitted:
    payload = {
        "brokered_by": brokered_by,
        "status": status,
        "bed": bed,
        "bath": bath,
        "acre_lot": acre_lot,
        "street": street,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "house_size": house_size,
        "prev_sold_year": prev_sold_year,
    }

    with st.spinner("Consultando la API de inferencia..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                model_version = result.get("model_version")

                st.success("Predicción obtenida exitosamente")

                col_pred, col_ver = st.columns(2)
                with col_pred:
                    st.metric(
                        label="💰 Precio Predicho",
                        value=f"${prediction:,.2f}" if prediction is not None else "N/A",
                    )
                with col_ver:
                    st.metric(
                        label="🤖 Versión del Modelo",
                        value=f"v{model_version}" if model_version else "N/A",
                    )

            elif response.status_code == 503:
                st.error(
                    "El servicio de inferencia no tiene un modelo cargado. "
                    "Contacte al administrador."
                )
            else:
                error_detail = response.json().get("detail", response.text)
                st.error(f"Error del servidor (HTTP {response.status_code}): {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"No se pudo conectar a la API de inferencia en {API_URL}. "
                "Verifique que el servicio esté activo."
            )
        except requests.exceptions.Timeout:
            st.error("La solicitud a la API excedió el tiempo de espera (30s).")
        except Exception as exc:
            st.error(f"Error inesperado: {exc}")

# ---------------------------------------------------------------------------
# Sección de Historial de Entrenamiento y Despliegue (RF9.3, RF9.4, RF4.7)
# ---------------------------------------------------------------------------

st.divider()
st.header("📋 Historial de Entrenamiento y Despliegue")
st.markdown(
    "Registro por lote de las decisiones de entrenamiento y promoción del modelo."
)


@st.cache_data(ttl=60)
def _load_training_history() -> pd.DataFrame:
    """Lee la tabla training_history desde mlops_db.

    Retorna un DataFrame vacío si la tabla no existe o la conexión falla.
    """
    try:
        engine = create_engine(_POSTGRES_MLOPS_CONN)
        query = text(
            "SELECT batch_number, trained, train_reason, promoted, "
            "promotion_reason, best_model, best_val_mae, "
            "mlflow_run_id, mlflow_model_version, code_commit, logged_at "
            "FROM training_history ORDER BY logged_at DESC"
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


df_history = _load_training_history()

if df_history.empty:
    st.info(
        "No hay registros de entrenamiento disponibles. "
        "El historial se poblará a medida que el pipeline procese lotes."
    )
else:
    # Resumen rápido
    total_batches = len(df_history)
    trained_count = int(df_history["trained"].sum())
    promoted_count = int(df_history["promoted"].fillna(False).sum())

    col_total, col_trained, col_promoted = st.columns(3)
    with col_total:
        st.metric("Total Lotes", total_batches)
    with col_trained:
        st.metric("Entrenamientos", trained_count)
    with col_promoted:
        st.metric("Promociones", promoted_count)

    st.subheader("Detalle por Lote")

    total = len(df_history)
    for idx, (_, row) in enumerate(df_history.iterrows()):
        batch_num = total - idx
        trained = row["trained"]
        train_reason = row["train_reason"] or "—"
        promoted = row["promoted"]
        promotion_reason = row["promotion_reason"] or "—"
        best_model = row["best_model"] or "—"
        best_val_mae = row["best_val_mae"]
        mlflow_run_id = row["mlflow_run_id"] or "—"
        mlflow_model_version = row["mlflow_model_version"] or "—"
        code_commit = row["code_commit"] or "—"
        logged_at = row["logged_at"]

        # Indicador visual de estado
        if trained and promoted:
            status_icon = "✅"
            status_label = "Entrenado y Promovido"
        elif trained and not promoted:
            status_icon = "⚠️"
            status_label = "Entrenado — Rechazado"
        else:
            status_icon = "⏭️"
            status_label = "Sin entrenamiento"

        with st.expander(
            f"{status_icon} Lote {batch_num} — {status_label}",
            expanded=False,
        ):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Decisión de Entrenamiento**")
                st.write(f"- Entrenó: {'Sí' if trained else 'No'}")
                st.write(f"- Motivo: {train_reason}")
                if trained:
                    st.write(f"- Mejor modelo: {best_model}")
                    mae_display = (
                        f"${best_val_mae:,.2f}" if best_val_mae is not None else "—"
                    )
                    st.write(f"- Mejor MAE (validación): {mae_display}")

            with col_b:
                st.markdown("**Decisión de Promoción**")
                if trained:
                    st.write(f"- Promovido: {'Sí' if promoted else 'No'}")
                    st.write(f"- Motivo: {promotion_reason}")
                else:
                    st.write("- N/A (no hubo entrenamiento)")

                st.markdown("**MLflow & Trazabilidad**")
                st.write(f"- Run ID: `{mlflow_run_id}`")
                st.write(f"- Model Version: `{mlflow_model_version}`")
                st.write(f"- Commit: `{code_commit}`")

            if logged_at is not None:
                st.caption(f"Registrado: {logged_at}")

