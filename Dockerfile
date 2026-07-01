FROM python:3.10-slim

# Instalar dependencias del sistema y Java (necesario para PySpark)
RUN apt-get update && \
    apt-get install -y default-jre procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

WORKDIR /app

# Copiar archivos de dependencias
COPY pyproject.toml requirements-docker.txt ./

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar código fuente
COPY src/ src/
RUN pip install -e .

# Copiar script de ejecución
COPY run_pipeline.py .

# Ejecutar el pipeline al levantar el contenedor
CMD ["python", "run_pipeline.py"]
