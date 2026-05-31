# Dockerfile para la API de inferencia FastAPI (RF7, RF10, RF13, RF14)
# Build context: raíz del repositorio (docker build -f api/Dockerfile .)
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para psycopg2 y compilación
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependencias de Python
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API y el paquete mlops_core
COPY api/ ./api/
COPY mlops_core/ ./mlops_core/

# Variable de entorno para que Python encuentre los módulos
ENV PYTHONPATH=/app

# Puerto de la API
EXPOSE 8000

# Health check integrado en el contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Ejecutar con uvicorn (sin --reload en producción)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
