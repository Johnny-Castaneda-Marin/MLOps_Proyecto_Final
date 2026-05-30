# 🏠 MLOps Proyecto Final — Real Estate Price Prediction

## 📑 Tabla de contenido

1. [Descripción general](#descripción-general)
2. [Arquitectura](#arquitectura)
3. [Requisitos previos](#requisitos-previos)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Levantar la infraestructura](#levantar-la-infraestructura)
6. [Servicios desplegados](#servicios-desplegados)
7. [Pipeline de Airflow (DAGs)](#pipeline-de-airflow-dags)
8. [Colaboradores](#colaboradores)

---

## Descripción general

Sistema completo de MLOps sobre Kubernetes que cubre el ciclo de vida de un modelo de Machine Learning para predecir precios de propiedades inmobiliarias. El sistema consume datos de forma incremental desde una API externa, los valida, decide automáticamente si reentrenar, registra experimentos en MLflow, compara el modelo candidato contra el productivo y lo promueve si mejora el desempeño.

**Dataset**: Propiedades inmobiliarias con variables como `price`, `bed`, `bath`, `house_size`, `city`, `state`, entre otras.

**Problema**: Regresión — predicción del precio de una propiedad.

**Métrica principal**: MAE (Mean Absolute Error). Se promueve un modelo si reduce el MAE al menos 3% frente al campeón actual.

---

## Arquitectura

```
Developer → GitHub → GitHub Actions → Docker Hub → Argo CD → Kubernetes
```

Dentro del clúster:

```
Data API → Airflow DAG → PostgreSQL (raw_db / clean_db)
                       → MLflow (experimentos + model registry)
                       → FastAPI (inferencia)
                       → Streamlit (UI)
                       → Locust (pruebas de carga)
                       → Prometheus + Grafana (observabilidad)
```

---

## Requisitos previos

- Docker Desktop con WSL2
- kind (`kind create cluster`)
- kubectl
- helm
- Git

Versiones usadas:

| Herramienta | Versión |
|-------------|---------|
| Docker | 29.2.1 |
| kubectl | v1.30 |
| helm | v3.20.2 |
| kind | v1.30.13 |
| Python | 3.12 |

---

## Estructura del proyecto

```
MLOps_ProyectoFinal/
├── airflow/
│   ├── dags/
│   │   ├── mlops_pipeline.py       # DAG principal
│   │   └── tasks/
│   │       ├── config.py           # Variables de configuración
│   │       ├── fetch_batch.py      # Consume la API de datos
│   │       ├── store_raw.py        # Almacena datos crudos en raw_db
│   │       ├── validate.py         # Validaciones + detección de drift + decisión
│   │       ├── preprocess.py       # Limpieza y almacenamiento en clean_db
│   │       └── train.py            # Entrenamiento, registro en MLflow y promoción
│   ├── Dockerfile
│   └── requirements.txt
├── api/                            # FastAPI — inferencia (pendiente)
├── streamlit/                      # UI Streamlit (pendiente)
├── locust/                         # Pruebas de carga (pendiente)
├── mlflow/
│   └── Dockerfile                  # MLflow con psycopg2 + boto3
├── k8s/
│   ├── namespace/
│   ├── postgres/
│   ├── minio/
│   ├── mlflow/
│   ├── airflow/
│   ├── data-api/
│   ├── api/
│   ├── streamlit/
│   ├── locust/
│   ├── prometheus/
│   ├── grafana/
│   └── argocd/
└── .github/
    └── workflows/                  # GitHub Actions (pendiente)
```

---

## Levantar la infraestructura

### 1. Crear el clúster

```bash
kind create cluster --name kind
kubectl get nodes
```

### 2. Crear el namespace

```bash
kubectl apply -f k8s/namespace/namespace.yaml
```

### 3. PostgreSQL

```bash
kubectl apply -f k8s/postgres/
kubectl wait --for=condition=Ready pod/postgres-0 -n mlops --timeout=120s

# Crear bases de datos
kubectl exec -it postgres-0 -n mlops -- psql -U mlops_user -d mlops_db -c "CREATE DATABASE raw_db;"
kubectl exec -it postgres-0 -n mlops -- psql -U mlops_user -d mlops_db -c "CREATE DATABASE clean_db;"
kubectl exec -it postgres-0 -n mlops -- psql -U mlops_user -d mlops_db -c "CREATE DATABASE airflow_db;"
```

### 4. MinIO

```bash
kubectl apply -f k8s/minio/
kubectl wait --for=condition=Ready pod/minio-0 -n mlops --timeout=120s
kubectl apply -f k8s/minio/job-create-bucket.yaml
kubectl logs job/minio-create-bucket -n mlops --follow
```

### 5. MLflow

La imagen de MLflow necesita `psycopg2` y `boto3`. Constrúyela y cárgala en kind:

```bash
docker build -t mlops-mlflow:latest mlflow/
kind load docker-image mlops-mlflow:latest --name kind
kubectl apply -f k8s/mlflow/
kubectl wait --for=condition=Ready pod -l app=mlflow -n mlops --timeout=120s
```

### 6. Data API

```bash
kubectl apply -f k8s/data-api/
kubectl wait --for=condition=Ready pod -l app=data-api -n mlops --timeout=120s
```

Verificar que la API responde:

```bash
kubectl port-forward svc/data-api-service 8000:80 -n mlops &
curl "http://localhost:8000/health"
```

### 7. Airflow

La imagen de Airflow incluye las dependencias de ML (scikit-learn, xgboost, lightgbm, mlflow, etc.):

```bash
docker build -t mlops-airflow:latest airflow/
kind load docker-image mlops-airflow:latest --name kind

helm repo add apache-airflow https://airflow.apache.org
helm repo update

helm install airflow apache-airflow/airflow \
  --namespace mlops \
  --values k8s/airflow/values.yaml \
  --version 1.15.0 \
  --timeout 10m
```

Copiar los DAGs al scheduler y webserver:

```bash
SCHEDULER=$(kubectl get pods -n mlops -l component=scheduler -o jsonpath='{.items[0].metadata.name}')
WEBSERVER=$(kubectl get pods -n mlops -l component=webserver -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n mlops $SCHEDULER -c scheduler -- mkdir -p /opt/airflow/dags/tasks
kubectl exec -n mlops $WEBSERVER -- mkdir -p /opt/airflow/dags/tasks

for f in mlops_pipeline.py; do
  kubectl cp airflow/dags/$f mlops/$SCHEDULER:/opt/airflow/dags/$f -c scheduler
  kubectl cp airflow/dags/$f mlops/$WEBSERVER:/opt/airflow/dags/$f
done

for f in config.py fetch_batch.py store_raw.py validate.py preprocess.py train.py; do
  kubectl cp airflow/dags/tasks/$f mlops/$SCHEDULER:/opt/airflow/dags/tasks/$f -c scheduler
  kubectl cp airflow/dags/tasks/$f mlops/$WEBSERVER:/opt/airflow/dags/tasks/$f
done

# Forzar reserialización en el webserver
kubectl exec -n mlops $WEBSERVER -- airflow dags reserialize
```

Acceder a la UI:

```bash
kubectl port-forward svc/airflow-webserver 8080:8080 -n mlops
```

- URL: `http://localhost:8080`
- Usuario: `admin`
- Password: `admin`

### 8. Argo CD

```bash
kubectl apply -f k8s/argocd/application.yaml
kubectl port-forward svc/argocd-server -n argocd 8443:443
```

Obtener contraseña:

```bash
kubectl get secret argocd-initial-admin-secret -n argocd \
  -o jsonpath="{.data.password}" | base64 -d && echo
```

- URL: `https://localhost:8443`
- Usuario: `admin`

---

## Servicios desplegados

| Servicio | Propósito | Puerto |
|----------|-----------|--------|
| `postgres-service` | Bases raw_db, clean_db, airflow_db | `5432` |
| `minio-service` | Artifact store para MLflow | `9000/9001` |
| `mlflow-service` | Tracking server y model registry | `5000` |
| `data-api-service` | API externa de datos del docente | `80` |
| `airflow-webserver` | UI de Airflow | `8080` |
| `api` | FastAPI inferencia (pendiente) | `8000` |
| `streamlit` | Interfaz gráfica (pendiente) | `8501` |
| `locust` | Pruebas de carga (pendiente) | `8089` |
| `prometheus` | Métricas (pendiente) | `9090` |
| `grafana` | Dashboards (pendiente) | `3000` |

---

## Pipeline de Airflow (DAGs)

### DAG: `mlops_pipeline`

Schedule: `@daily` (ejecutar manualmente por ahora)

#### Flujo de tareas

```
start
  └── fetch_batch          # Consume un lote de la API externa
        └── store_raw      # Guarda datos crudos en raw_db (bulk insert)
              └── validate_schema    # Verifica columnas y tipos
                    └── validate_quality   # Nulos, duplicados, precios inválidos
                          └── detect_drift       # Compara distribuciones con histórico
                                └── decide_training    # Bifurcación: entrenar o no
                                      ├── preprocess_data → train_and_promote → log_result → end
                                      └── skip_training → log_result → end
```

#### Criterios de decisión de entrenamiento

El DAG entrena si se cumple al menos uno de estos criterios:

- Se detectó drift en variables numéricas (cambio > 10% en media)
- El lote nuevo tiene más de 1,000 registros nuevos
- El lote aumenta el volumen histórico en más del 5%

No entrena si:
- Total acumulado < 1,000 registros
- El lote tiene problemas graves de calidad

#### Modelos entrenados

| Modelo | Hiperparámetros |
|--------|----------------|
| Ridge | `alpha=1.0` |
| Random Forest | `n_estimators=100`, `max_depth=10` |
| Gradient Boosting | `n_estimators=100`, `max_depth=5` |

#### Criterio de promoción

Se promueve el candidato si su `val_mae` mejora al menos **3%** frente al campeón actual en MLflow. El modelo productivo se identifica con el alias `champion`.

#### Ejecutar el pipeline manualmente

```bash
SCHEDULER=$(kubectl get pods -n mlops -l component=scheduler -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n mlops $SCHEDULER -c scheduler -- airflow dags trigger mlops_pipeline
```

#### Reiniciar la API de datos (volver al lote 1)

```bash
kubectl port-forward svc/data-api-service 8000:80 -n mlops &
curl "http://localhost:8000/restart_data_generation?group_number=1"
```

---

## Variables de entorno (Airflow)

| Variable | Valor |
|----------|-------|
| `DATA_API_URL` | `http://data-api-service` |
| `POSTGRES_RAW_CONN` | `postgresql://mlops_user:mlops1234@postgres-service:5432/raw_db` |
| `POSTGRES_CLEAN_CONN` | `postgresql://mlops_user:mlops1234@postgres-service:5432/clean_db` |
| `MLFLOW_TRACKING_URI` | `http://mlflow-service:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio-service:9000` |
| `AWS_ACCESS_KEY_ID` | `minioadmin` |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin123` |

---

## 👥 Colaboradores

- 🧑‍💻 **Camilo Cortés** — [![GitHub](https://img.shields.io/badge/GitHub-@cccortesh95-181717?logo=github)](https://github.com/cccortesh95)
- 🧑‍💻 **Johnny Castañeda** — [![GitHub](https://img.shields.io/badge/GitHub-@Johnny--Castaneda--Marin-181717?logo=github)](https://github.com/Johnny-Castaneda-Marin)
- 🧑‍💻 **Benkos Triana** — [![GitHub](https://img.shields.io/badge/GitHub-@BenkosT-181717?logo=github)](https://github.com/BenkosT)
