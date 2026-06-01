# Imagen de Airflow para la plataforma MLOps.
#
# IMPORTANTE: esta imagen se construye con el CONTEXTO en la RAÍZ del repo
# (docker build -f airflow/Dockerfile .), porque necesita empaquetar el módulo
# de lógica pura `mlops_core`, que vive en la raíz y es compartido con la API.
# El workflow de CI usa `context: .` para este Dockerfile.

FROM apache/airflow:3.0.1-python3.10

# --- Dependencias del sistema (compilación de psycopg2, etc.) ----------------
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# --- Dependencias Python de los DAGs -----------------------------------------
# Se copian primero para aprovechar la caché de capas de Docker.
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Paquete compartido mlops_core -------------------------------------------
# Se instala como paquete real para que las tareas del DAG puedan hacer
# `import mlops_core` directamente, sin manipular sys.path en tiempo de
# ejecución. Esto resuelve el fallo de las tareas que importan mlops_core
# (store_raw, validate, train, etc.).
COPY pyproject.toml /opt/mlops/pyproject.toml
COPY mlops_core/ /opt/mlops/mlops_core/
RUN pip install --no-cache-dir /opt/mlops
