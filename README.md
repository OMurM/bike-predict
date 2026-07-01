# Bike Predict (Mundimoto) 🏍️

Este proyecto es una pipeline de Machine Learning End-to-End (`Data Engineering` + `MLOps`) diseñada para rashear, procesar y predecir el precio de mercado de motos de segunda mano extraídas desde la web de Mundimoto.

Toda la arquitectura funciona sobre **Apache Spark (PySpark)** y está **100% dockerizada**, lo que significa que puedes ejecutar todo el flujo en local sin necesidad de configuraciones complejas de Java o Hadoop.

## 🏗️ Arquitectura del Proyecto

El flujo de datos sigue la **Arquitectura Medallón** (Medallion Architecture) almacenando los datos en formato **Delta Lake**:

1. **Scraping**: Extrae información real de motos de segunda mano desde la API pública.
2. **Capa Bronze (Raw)**: Ingesta los datos crudos en formato Delta. Sirve como fuente histórica inmutable de verdad.
3. **Capa Silver (Cleansed)**: Limpia y normaliza los datos (ej. extraer números de texto, filtrar valores nulos, normalizar tipos de moto).
4. **Capa Gold (Features)**: Agrupa y formatea los datos específicamente para que sean digeribles por el modelo de ML (agrupación de kilómetros, cálculo de edad real de la moto).
5. **Machine Learning (MLflow)**: Entrena un modelo `RandomForestRegressor` utilizando las features de la tabla Gold para predecir la variable objetivo: `precio`.

## 🛠️ Tecnologías

* **Lenguaje:** Python 3.10
* **Data Processing:** PySpark 3.5.1 + Delta Spark 3.1.0
* **Machine Learning:** PySpark MLlib (Random Forest)
* **MLOps Tracking:** MLflow (Métricas, Parámetros, Model Registry)
* **Entorno:** Docker & Docker Compose

## 🚀 Cómo ejecutar el proyecto (Local)

Al estar dockerizado, el entorno incluye todas las dependencias necesarias de Java, Hadoop, Spark y MLflow. Además, utiliza volúmenes locales para que cualquier cambio que hagas en el código (`src/` o `run_pipeline.py`) se refleje instantáneamente sin tener que reconstruir la imagen.

### 1. Levantar la infraestructura

```bash
docker compose up -d
```
Esto creará el contenedor y montará las siguientes carpetas de tu ordenador:
- `spark-warehouse/`: (Ignorada en Git) Base de datos local (Derby) y archivos parquet/delta.
- `mlruns/`: (Subida a Git) Donde se guardan las métricas y los modelos físicos empaquetados.
- `src/` y `run_pipeline.py`: Tu código fuente en vivo.

### 2. Ejecutar el Pipeline Completo

Una vez el contenedor está encendido, simplemente ejecuta:

```bash
docker compose exec pipeline python run_pipeline.py
```

El script hará el scraping, pasará por las capas Bronze, Silver y Gold, y finalmente entrenará el modelo. Si quieres omitir alguna fase, puedes comentar las líneas correspondientes en el archivo `run_pipeline.py`.

### 3. Ver los Resultados (MLflow UI)

Para ver el dashboard con el histórico de modelos entrenados, las gráficas de rendimiento (RMSE, MAE, R2) y para descargar el modelo empaquetado para producción, abre tu navegador y entra a:

👉 **[http://localhost:5000](http://localhost:5000)**

## 📊 Estructura del Código

```text
├── docker-compose.yml       # Orquestador del contenedor Spark + MLflow
├── Dockerfile               # Configuración de Ubuntu + Java + Python
├── run_pipeline.py          # Script principal (Orquestador End-to-End)
├── mlruns/                  # Modelos de Machine Learning generados
├── src/
│   └── motos_ml/
│       ├── scraping/        # Lógica de requests y parseo de datos
│       ├── ingestion/       # Escritura en capa Bronze (Esquemas estrictos)
│       ├── transforms/      # Transformaciones Silver (limpieza) y Gold (features)
│       ├── ml/              # Pipeline de PySpark MLlib y registro en MLflow
│       ├── config.py        # Configuraciones globales (Catálogo, Hiperparámetros)
│       └── dto.py           # Data Transfer Objects (Pydantic schemas)
```
