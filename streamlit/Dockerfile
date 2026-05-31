# Dockerfile para la UI Streamlit (RF9, RF13, RF14)
# Build context: raíz del repositorio (docker build -f streamlit/Dockerfile .)
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para psycopg2 (training history)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependencias de Python
COPY streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación Streamlit
COPY streamlit/ ./streamlit/

# Variable de entorno para que Python encuentre los módulos
ENV PYTHONPATH=/app

# Puerto por defecto de Streamlit
EXPOSE 8501

# Health check integrado en el contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Ejecutar Streamlit sin abrir navegador y escuchando en todas las interfaces
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
