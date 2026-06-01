# Plataforma MLOps — Predicción de Precios de Inmuebles

Plataforma MLOps de extremo a extremo para la predicción de precios de bienes raíces, desplegada sobre Kubernetes con despliegue declarativo (GitOps) vía Argo CD. El sistema cubre el ciclo completo: ingesta incremental de datos, validación, decisión automática de entrenamiento, registro de experimentos en MLflow, promoción de modelos por desempeño, servicio de inferencia con recarga en caliente, interfaz de usuario, observabilidad y pruebas de carga.

## Tabla de Contenidos

1. [Arquitectura](#1-arquitectura)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Componentes y Namespaces](#3-componentes-y-namespaces)
4. [Capa de Lógica Pura `mlops_core`](#4-capa-de-lógica-pura-mlops_core)
5. [Pipeline de Airflow](#5-pipeline-de-airflow)
6. [Bases de Datos y Backends](#6-bases-de-datos-y-backends)
7. [API de Inferencia (FastAPI)](#7-api-de-inferencia-fastapi)
8. [Interfaz Streamlit](#8-interfaz-streamlit)
9. [Observabilidad: Prometheus + Grafana](#9-observabilidad-prometheus--grafana)
10. [Pruebas de Carga con Locust](#10-pruebas-de-carga-con-locust)
11. [CI/CD y GitOps](#11-cicd-y-gitops)
12. [Pruebas Basadas en Propiedades](#12-pruebas-basadas-en-propiedades)
13. [Despliegue](#13-despliegue)
14. [Evidencias](#14-evidencias)
15. [Colaboradores](#15-colaboradores)

---

## 1. Arquitectura

El sistema se despliega en Kubernetes y se sincroniza de forma declarativa con Argo CD. Las imágenes se construyen y publican en DockerHub mediante GitHub Actions. Cada componente vive en su propio namespace para aislamiento.

```mermaid
flowchart LR
    Dev[Developer] --> GH[GitHub Repo]
    GH --> GHA[GitHub Actions]
    GHA --> DH[DockerHub]
    GH --> ACD[Argo CD]
    ACD --> K8S[(Kubernetes)]
    DH --> K8S

    subgraph K8S[Kubernetes]
        DAPI[data-api]
        AF[Airflow]
        PG[(PostgreSQL)]
        MF[MLflow]
        MINIO[(MinIO S3)]
        API[FastAPI inference]
        ST[Streamlit UI]
        LOC[Locust]
        PROM[Prometheus]
        GRAF[Grafana]
    end

    DAPI --> AF
    AF --> PG
    AF --> MF
    MF --> PG
    MF --> MINIO
    API --> MF
    API --> PG
    ST --> API
    LOC --> API
    PROM --> API
    GRAF --> PROM
```

La decisión de diseño central es la extracción de toda la lógica de negocio a una capa pura y testeable (`mlops_core`), de modo que las tareas de Airflow y la API se convierten en **adaptadores delgados** que solo manejan I/O y delegan las reglas de decisión.

<!-- Imagen: Diagrama de arquitectura / vista general del despliegue -->
<!-- ![Arquitectura](images/arquitectura.png) -->

---

## 2. Estructura del Proyecto

```
MLOps_Proyecto_Final/
├── airflow/                          # Orquestación del pipeline
│   ├── dags/
│   │   ├── mlops_pipeline.py         # DAG principal
│   │   └── tasks/                    # Adaptadores delgados (I/O)
│   │       ├── fetch_batch.py
│   │       ├── store_raw.py
│   │       ├── validate.py
│   │       ├── preprocess.py
│   │       └── train.py
│   ├── Dockerfile
│   └── requirements.txt
├── mlops_core/                       # Lógica pura testeable (sin I/O)
│   ├── ingest.py                     # Selección de día, deduplicación, hash
│   ├── validation.py                 # Esquema, calidad, drift, nuevas categorías
│   ├── decision.py                   # Regla de decisión de entrenamiento
│   ├── promotion.py                  # Regla de promoción (MAE + guarda RMSE)
│   ├── features.py                   # Codificación + particionado determinista
│   ├── logging_schema.py             # Payloads de auditoría / experimento / inferencia
│   └── types.py                      # Dataclasses compartidas
├── api/                              # API de inferencia (FastAPI)
│   ├── main.py                       # Endpoints
│   ├── config.py                     # Configuración por variables de entorno
│   ├── model_holder.py               # Holder thread-safe + recarga en caliente
│   ├── inference_log.py              # Registro de inferencias en raw_db
│   └── Dockerfile
├── streamlit/                        # UI de inferencia + historial
│   ├── app.py
│   └── Dockerfile
├── locust/                           # Pruebas de carga
│   ├── locustfile.py
│   └── Dockerfile
├── prometheus/                       # Configuración de scrape
│   └── prometheus.yml
├── grafana/                          # Dashboards y provisioning
│   ├── dashboards/
│   └── provisioning/
├── mlflow/                           # Imagen de MLflow
│   └── Dockerfile
├── k8s/                              # Manifiestos de Kubernetes (GitOps)
│   ├── namespace/                    # Definición de namespaces
│   ├── postgres/  minio/  mlflow/    # Infraestructura
│   ├── data-api/  api/  streamlit/   # Aplicaciones
│   ├── locust/  prometheus/  grafana/# Carga y observabilidad
│   └── argocd/                       # Application de Argo CD
├── tests/                            # PBT + unitarias + integración
├── .github/workflows/                # CI/CD (build/tag/push a DockerHub)
└── README.md
```

La infraestructura (`k8s/`) está separada del código de cada componente, y la lógica de negocio (`mlops_core/`) está separada de los adaptadores de I/O (`airflow/dags/tasks/`, `api/`).

---

## 3. Componentes y Namespaces

Cada aplicación se despliega en un namespace dedicado. La comunicación usa FQDNs de Kubernetes (`service.namespace.svc.cluster.local`).
 
| Componente | Namespace | Puerto | Propósito |
|---|---|---|---|
| PostgreSQL | `mlops-data` | 5432 | Bases `raw_db`, `clean_db`, `mlops_db`, `mlflow_db`, `airflow_db` |
| Airflow | `mlops-data` | 8080 | Orquestador del pipeline |
| MinIO | `mlops-storage` | 9000 / 9001 | Almacén de artefactos S3 (`s3://mlflow`) |
| MLflow | `mlops-mlflow` | 5000 | Tracking + Model Registry |
| Data API | `mlops-data-api` | 80 | Fuente externa de lotes diarios |
| Inference API | `mlops-api` | 8000 | Servicio de predicción + recarga en caliente |
| Streamlit | `mlops-ui` | 8501 | Interfaz de inferencia e historial |
| Locust | `mlops-loadtest` | 8089 | Pruebas de carga |
| Prometheus | `mlops-monitoring` | 9090 | Recolección de métricas |
| Grafana | `mlops-monitoring` | 3000 | Dashboards de carga/latencia |
| Argo CD | `argocd` | 443 | Sincronización declarativa (GitOps) |

<!-- Imagen: kubectl get pods -A mostrando los pods por namespace -->
<!-- ![Pods por namespace](images/pods_namespaces.png) -->

---

## 4. Capa Lógica `mlops_core`

Toda la lógica de negocio vive en `mlops_core`, sin dependencias de infraestructura (PostgreSQL, MLflow, HTTP). Esto la hace determinista y testeable con pruebas basadas en propiedades.

| Módulo | Responsabilidad |
|---|---|
| `ingest` | `select_day()`, `is_exhausted()`, `row_hash()` (MD5 estable), `deduplicate()` |
| `validation` | Validación de esquema (incluye tipos), calidad, drift numérico y nuevas categorías |
| `decision` | `decide_training()` — regla técnica sin periodicidad ni conteo bruto de lotes |
| `promotion` | `should_promote()` — MAE con guarda de RMSE |
| `features` | Codificación de categóricas con manejo de desconocidos, particionado determinista |
| `logging_schema` | Construcción y serialización de eventos de auditoría, experimento e inferencia |

Las tareas de Airflow y la API importan estas funciones y solo se encargan de leer/escribir datos.

---

## 5. Pipeline de Airflow

El DAG `mlops_pipeline` orquesta el flujo completo desde la ingesta hasta la promoción.

```mermaid
flowchart TD
    start([start]) --> fetch[fetch_batch]
    fetch --> store[store_raw]
    store --> vs[validate_schema]
    vs --> vq[validate_quality]
    vq --> dd[detect_drift]
    dd --> decide{decide_training}
    decide -->|TRAIN| pre[preprocess_data]
    pre --> train[train_candidates]
    train --> compare[compare_models]
    compare --> promote[promote_or_reject]
    promote --> log[log_result]
    decide -->|SKIP| skip[skip_training]
    skip --> log
    log --> done([end])
```

### Tareas principales

- **`fetch_batch`**: ingesta incremental. Lee el conteo de `batch_control`, delega la selección de día y la detección de agotamiento a `mlops_core.ingest`, llama a la Data API con `timeout=120` y persiste metadatos.
- **`store_raw`**: persistencia cruda con deduplicación por `row_hash`, conservando los campos originales sin modificación.
- **`validate_schema` / `validate_quality` / `detect_drift`**: validaciones delegadas a `mlops_core.validation`.
- **`decide_training`** (`BranchPythonOperator`): decide TRAIN o SKIP según reglas técnicas (drift, volumen nuevo, categorías nuevas), nunca por periodicidad.
- **`train_candidates`**: entrena Ridge, RandomForest y GradientBoosting (`random_state=42`), registra en MLflow parámetros, métricas, commit de código y artefactos.
- **`compare_models` / `promote_or_reject`**: compara el candidato contra el `champion` y reasigna el alias solo si mejora el MAE sin empeorar el RMSE.
- **`log_result`**: registra la decisión, motivos e IDs de MLflow en `training_history`.

<!-- Imagen: Graph View del DAG mlops_pipeline en Airflow -->
<!-- ![DAG Graph View](images/dag_graph.png) -->

<!-- Imagen: Ejecución exitosa del DAG (todos los tasks en verde) -->
<!-- ![DAG exitoso](images/dag_exitoso.png) -->

---

## 6. Bases de Datos y Backends

El sistema usa PostgreSQL con bases dedicadas por dominio, creadas automáticamente en el primer arranque mediante un script de inicialización.

| Base | Contenido |
|---|---|
| `raw_db` | `batch_control`, `raw_properties`, `inference_log` |
| `clean_db` | `clean_properties` (datos procesados con `split`) |
| `mlops_db` | `training_history` (auditoría de entrenamiento/promoción) |
| `mlflow_db` | Backend de metadatos de MLflow |
| `airflow_db` | Metadatos internos de Airflow |

### Backends de MLflow

- **Metadatos**: PostgreSQL `mlflow_db` (no SQLite).
- **Artefactos**: bucket `s3://mlflow` en MinIO.
- **Modelo registrado**: `real_estate_champion`, con alias de producción `champion`.

<!-- Imagen: Bases de datos creadas en PostgreSQL -->
<!-- ![Bases de datos](images/databases.png) -->

<!-- Imagen: MLflow UI con experimentos y modelo registrado -->
<!-- ![MLflow](images/mlflow_ui.png) -->

---

## 7. API de Inferencia (FastAPI)

La API sirve el modelo `champion` con recarga en caliente, usando MLflow como única fuente de verdad (sin rutas locales ni versiones fijas en el código).

### Endpoints

| Método | Ruta | Descripción | Auth |
|---|---|---|---|
| `POST` | `/predict` | Recibe features de propiedad, devuelve predicción + versión del modelo | No |
| `POST` | `/admin/reload` | Fuerza recarga del modelo `champion` desde MLflow | Sí (token Bearer) |
| `GET` | `/health` | Liveness/readiness check | No |
| `GET` | `/metrics` | Métricas en formato Prometheus | No |

### Recarga en caliente

El `ModelHolder` mantiene el modelo y su versión en memoria, protegido con lock. Un poller periódico consulta el alias `champion` en MLflow y, si cambió, dispara una recarga atómica. Si la carga falla, se conserva el modelo previo. El endpoint `/admin/reload` permite recarga bajo demanda protegida por token.

### Contrato `/predict`

```json
// request
{ "brokered_by": 101.0, "status": "for_sale", "bed": 3, "bath": 2,
  "acre_lot": 0.12, "street": 123.0, "city": "Austin", "state": "Texas",
  "zip_code": 78701.0, "house_size": 1800.0, "prev_sold_year": 2018 }

// response
{ "prediction": 412500.0, "model_version": "7", "status": "ok" }
```

Cada solicitud (exitosa o fallida) se registra en la tabla `inference_log` de `raw_db`.

<!-- Imagen: Swagger UI de la API en /docs -->
<!-- ![Swagger UI](images/api_docs.png) -->

<!-- Imagen: Respuesta exitosa de POST /predict -->
<!-- ![Predicción](images/api_predict.png) -->

<!-- Imagen: Respuesta de GET /health y GET /metrics -->
<!-- ![Health y metrics](images/api_health_metrics.png) -->

---

## 8. Interfaz Streamlit

La UI tiene dos secciones:

- **Inferencia**: formulario con las características de la propiedad que envía un `POST /predict` y muestra la predicción y la versión del modelo.
- **Historial de Entrenamiento y Despliegue**: lee `training_history` y muestra por lote la decisión de entrenamiento, el motivo, si fue promovido o rechazado, el cambio de desempeño y los identificadores de MLflow.

<!-- Imagen: Sección de inferencia de Streamlit con una predicción -->
<!-- ![Streamlit Inferencia](images/streamlit_inferencia.png) -->

<!-- Imagen: Sección de historial de entrenamiento -->
<!-- ![Streamlit Historial](images/streamlit_historial.png) -->

---

## 9. Observabilidad: Prometheus + Grafana

La API expone métricas en `/metrics` con `prometheus_client`:

| Métrica | Tipo | Descripción |
|---|---|---|
| `inference_requests_total` | Counter | Total de solicitudes de inferencia |
| `inference_latency_seconds` | Histogram | Latencia de las solicitudes |
| `inference_errors_total` | Counter | Total de errores de inferencia |
| `model_info{version, model_name}` | Gauge | Versión del modelo en servicio |

Prometheus hace scrape de la API y Grafana presenta un dashboard con throughput, latencia p50/p95/p99, tasa de error y versión del modelo.

<!-- Imagen: Target de la API UP en Prometheus -->
<!-- ![Prometheus targets](images/prometheus_targets.png) -->

<!-- Imagen: Dashboard de Grafana con throughput y latencias -->
<!-- ![Grafana dashboard](images/grafana_dashboard.png) -->

---

## 10. Pruebas de Carga con Locust

Locust ejecuta escenarios de carga contra `/predict` con payloads aleatorios realistas. Define dos perfiles de usuario:

- **`PredictUser`**: carga sostenida (espera 1–3 s entre peticiones).
- **`AggressiveUser`**: carga pico (espera 0.1–0.5 s entre peticiones).

Locust se despliega en **modo web UI**: arranca de forma persistente exponiendo su interfaz en el puerto 8089, desde donde se define el número de usuarios y el spawn rate y se lanza la prueba contra `/predict`. El efecto de la carga se observa en el dashboard de Grafana.

```bash
# Acceder a la UI de Locust
kubectl port-forward -n mlops-loadtest svc/locust-service 8089:8089
# Abrir http://localhost:8089
```

<!-- Imagen: Estadísticas de Locust (RPS, latencias) -->
<!-- ![Locust stats](images/locust_stats.png) -->

<!-- Imagen: Latencia en Grafana durante la prueba de carga -->
<!-- ![Grafana bajo carga](images/grafana_carga.png) -->

### Solución de problemas: pod de Locust en `CrashLoopBackOff`

**Síntoma:** el pod de Locust queda en `CrashLoopBackOff` / `Degraded`, con eventos repetidos de:

- `Unhealthy`: `Readiness probe failed: Get "http://<pod-ip>:8089/": dial tcp ...:8089: connect: connection refused`
- `BackOff`: `Back-off restarting failed container locust`

**Causa:** el Deployment se había configurado con argumentos de modo batch (`--headless`, `--users`, `--spawn-rate`, `--run-time=5m`) que son incompatibles con un servicio web persistente con probes en el puerto 8089:

1. `--headless` hace que Locust **no levante la interfaz web** en el puerto 8089, por lo que el readiness/liveness probe a `:8089/` recibe `connection refused`.
2. `--run-time=5m` hace que el proceso **termine** tras 5 minutos; como es un `Deployment`, Kubernetes reinicia el contenedor y entra en `CrashLoopBackOff`.

**Solución:** ejecutar Locust en modo web UI (sin `--headless` ni `--run-time`), dejando solo el `--host` de destino, de modo que el proceso permanezca vivo y los probes del puerto 8089 pasen. La carga se controla desde la UI.

```yaml
# k8s/locust/deployment.yaml (extracto)
args:
  - "--host=http://inference-api-service.mlops-api.svc.cluster.local:8000"
# (se eliminaron --headless, --users, --spawn-rate y --run-time)
readinessProbe:
  httpGet:
    path: /
    port: 8089
  initialDelaySeconds: 15
  periodSeconds: 10
livenessProbe:
  httpGet:
    path: /
    port: 8089
  initialDelaySeconds: 30
  periodSeconds: 15
```

```bash
# Aplicar el fix y reiniciar
kubectl apply -f k8s/locust/deployment.yaml
kubectl rollout restart deployment/locust -n mlops-loadtest
```

> Nota: si en lugar de la UI se quiere una prueba que se ejecute automáticamente y termine, el recurso correcto es un `Job` (no un `Deployment`) y deben eliminarse los probes, ya que el contenedor está diseñado para finalizar.

<!-- Imagen: Pod de Locust en estado Running tras el fix -->
<!-- ![Locust running](images/locust_running.png) -->

---

## 11. CI/CD y GitOps

### GitHub Actions

El workflow `.github/workflows/docker-build-push.yml` construye, etiqueta y publica las imágenes de cada componente (api, streamlit, airflow, locust) en DockerHub. Cada imagen se etiqueta con el **SHA corto del commit** y con `latest`. Usa detección de cambios para construir solo los componentes modificados.

Requiere configurar dos secrets en el repositorio (Settings → Secrets and variables → Actions):

| Secret | Valor |
|---|---|
| `DOCKERHUB_USERNAME` | usuario de DockerHub |
| `DOCKERHUB_TOKEN` | access token de DockerHub (Read & Write) |

### Argo CD

La `Application` en `k8s/argocd/application.yaml` sincroniza recursivamente todos los manifiestos de `k8s/` hacia el clúster, con `prune` y `selfHeal` habilitados, y crea los namespaces automáticamente.

<!-- Imagen: Workflow de GitHub Actions en verde -->
<!-- ![GitHub Actions](images/github_actions.png) -->

<!-- Imagen: Application de Argo CD sincronizada (Synced / Healthy) -->
<!-- ![Argo CD](images/argocd_synced.png) -->

---

## 12. Pruebas Basadas en Propiedades

La capa `mlops_core` y el `ModelHolder` de la API se validan con 17 propiedades de correctitud, cada una implementada como un test basado en propiedades con `Hypothesis` (mínimo 100 iteraciones). Las verificaciones de infraestructura se cubren con pruebas unitarias y de integración.

| # | Propiedad | Valida |
|---|---|---|
| 1 | Selección de día por índice acumulado | RF1.1 |
| 2 | Condición de agotamiento | RF1.6 |
| 3 | Deduplicación, idempotencia y estabilidad del hash | RF1.4 |
| 4 | Preservación de datos crudos | RF2.1 |
| 5 | Preservación de trazabilidad por lote | RF2.5, RF11.1 |
| 6 | Validación de esquema detecta toda discrepancia | RF3.1 |
| 7 | Validación de calidad y aptitud del lote | RF3.2, RF3.8 |
| 8 | Detección de drift numérico por umbral | RF3.3 |
| 9 | Detección de nuevas categorías | RF3.4 |
| 10 | Manejo robusto de categorías desconocidas | RF3.5 |
| 11 | Decisión de entrenamiento por reglas técnicas | RF3.6, RF3.7, RF4.2, RF4.4 |
| 12 | Independencia de periodicidad y conteo de lotes | RF4.3 |
| 13 | Regla de promoción (MAE con guarda de RMSE) | RF6.1–6.5 |
| 14 | Round-trip del registro de auditoría | RF4.6, RF6.7 |
| 15 | Consistencia del modelo servido bajo recarga y fallo | RF7.5, RF7.6, RF7.7 |
| 16 | Cobertura y round-trip del evento de inferencia | RF8.2, RF8.3 |
| 17 | Particionado determinista (reproducibilidad) | RF11.3 |

### Ejecutar las pruebas

```bash
# Crear entorno e instalar dependencias
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt

# Ejecutar toda la suite
pytest tests/ -v
```

<!-- Imagen: Salida de pytest con todas las pruebas en verde -->
<!-- ![Pytest](images/pytest_passed.png) -->

---

## 13. Despliegue

### Prerrequisitos
 
- Clúster de Kubernetes (kind, minikube o gestionado)
- Argo CD instalado en el clúster
- Docker Hub con las imágenes publicadas (vía GitHub Actions)
- Token de GitHub con permiso `repo` (para gitSync de Airflow)
### Paso 1 — Aplicar la Application de Argo CD
 
```bash
kubectl apply -f k8s/argocd/application.yaml
```
 
Argo CD sincroniza automáticamente todos los manifiestos de `k8s/` y crea los namespaces.
 
### Paso 2 — Instalar Airflow con Helm
 
```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
 
helm install airflow apache-airflow/airflow \
  --namespace mlops-data \
  --values k8s/airflow/values.yaml \
  --version 1.15.0 \
  --timeout 10m
```
 
### Paso 3 — Crear el secret de gitSync
 
Reemplaza `TU_GITHUB_USERNAME` y `TU_GITHUB_TOKEN` con tus propias credenciales. El token se genera en `https://github.com/settings/tokens` con permiso `repo`.
 
```bash
kubectl create secret generic airflow-gitsync-secret \
  --from-literal=GITSYNC_USERNAME=TU_GITHUB_USERNAME \
  --from-literal=GITSYNC_PASSWORD=TU_GITHUB_TOKEN \
  --from-literal=GIT_SYNC_USERNAME=TU_GITHUB_USERNAME \
  --from-literal=GIT_SYNC_PASSWORD=TU_GITHUB_TOKEN \
  -n mlops-data
```
 
### Paso 4 — Crear el bucket de MinIO
 
```bash
kubectl exec -it minio-0 -n mlops-storage -- \
  mc alias set myminio http://localhost:9000 minioadmin minioadmin123
kubectl exec -it minio-0 -n mlops-storage -- mc mb myminio/mlflow
```
 
### Paso 5 — Agregar credenciales AWS al secret de la API
 
```bash
kubectl patch secret api-secret -n mlops-api --type='json' -p='[
  {"op": "add", "path": "/data/AWS_ACCESS_KEY_ID", "value": "'$(echo -n minioadmin | base64)'"},
  {"op": "add", "path": "/data/AWS_SECRET_ACCESS_KEY", "value": "'$(echo -n minioadmin123 | base64)'"},
  {"op": "add", "path": "/data/MLFLOW_S3_ENDPOINT_URL", "value": "'$(echo -n http://minio-service.mlops-storage.svc.cluster.local:9000 | base64)'"}
]'
kubectl rollout restart deployment/inference-api -n mlops-api
```
 
### Paso 6 — Verificar todos los pods
 
```bash
kubectl get pods -A | grep mlops
```
 
### Paso 7 — Port-forwards para acceder a los servicios
 
```bash
kubectl port-forward svc/airflow-webserver 8080:8080 -n mlops-data &
kubectl port-forward svc/mlflow-service 5001:5000 -n mlops-mlflow &
kubectl port-forward svc/inference-api-service 8001:8000 -n mlops-api &
kubectl port-forward svc/streamlit-service 8501:8501 -n mlops-ui &
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-monitoring &
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-monitoring &
kubectl port-forward svc/locust-service 8089:8089 -n mlops-loadtest &
kubectl port-forward svc/argocd-server 8443:443 -n argocd &
```
 
| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 (admin/admin) |
| MLflow | http://localhost:5001 |
| FastAPI docs | http://localhost:8001/docs |
| Streamlit | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Locust | http://localhost:8089 |
| Argo CD | https://localhost:8443 (admin) |
 
### Paso 8 — Cargar el modelo champion en la API
 
```bash
curl -s -X POST http://localhost:8001/admin/reload \
  -H "Authorization: Bearer changeme-admin-token"
```
 
### Paso 9 — Ejecutar el pipeline de Airflow
 
En `http://localhost:8080` activa el DAG `mlops_pipeline` y dispáralo manualmente con ▶.


---

## 14. Evidencias

Esta sección reúne las capturas de las pruebas realizadas sobre la plataforma. Coloca las imágenes en una carpeta `images/` y reemplaza los marcadores.

### 14.1 Pipeline de Airflow

<!-- ![DAG ejecutado exitosamente](images/evidencia_dag.png) -->

### 14.2 Entrenamiento y registro en MLflow

<!-- ![Experimentos y modelo champion en MLflow](images/evidencia_mlflow.png) -->

### 14.3 API de inferencia

<!-- ![Predicción vía API](images/evidencia_api_predict.png) -->
<!-- ![Recarga de modelo /admin/reload](images/evidencia_api_reload.png) -->

### 14.4 Interfaz Streamlit

<!-- ![Inferencia desde Streamlit](images/evidencia_streamlit_pred.png) -->
<!-- ![Historial de entrenamiento](images/evidencia_streamlit_historial.png) -->

### 14.5 Observabilidad

<!-- ![Prometheus scrapeando la API](images/evidencia_prometheus.png) -->
<!-- ![Dashboard de Grafana](images/evidencia_grafana.png) -->

### 14.6 Pruebas de carga

<!-- ![Resultados de Locust](images/evidencia_locust.png) -->

### 14.7 CI/CD y GitOps

<!-- ![Pipeline de GitHub Actions](images/evidencia_ci.png) -->
<!-- ![Argo CD sincronizado](images/evidencia_argocd.png) -->

### 14.8 Pruebas automatizadas

<!-- ![Suite de pruebas en verde](images/evidencia_pytest.png) -->

---

## 15. Colaboradores

- 🧑‍💻 **Camilo Cortés** — [![GitHub](https://img.shields.io/badge/GitHub-@cccortesh95-181717?logo=github)](https://github.com/cccortesh95)
- 🧑‍💻 **Johnny Castañeda** — [![GitHub](https://img.shields.io/badge/GitHub-@Johnny--Castaneda--Marin-181717?logo=github)](https://github.com/Johnny-Castaneda-Marin)
- 🧑‍💻 **Benkos Triana** — [![GitHub](https://img.shields.io/badge/GitHub-@BenkosT-181717?logo=github)](https://github.com/BenkosT)

---

> Pontificia Universidad Javeriana — Maestría en Inteligencia Artificial — MLOps — Proyecto Final 2026-1
