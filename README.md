# Bike Price Predictor 🏍️

Pipeline End-to-End de Machine Learning para extraer, procesar y predecir el precio de mercado de motos de segunda mano.

El proyecto está diseñado sobre **Apache Spark (PySpark)** utilizando la **Arquitectura Medallón** con Delta Lake. Todo el entorno está dockerizado para que sea fácil de levantar en cualquier máquina sin tener que pelearse con configuraciones de Java o Hadoop.

## 🏗️ Arquitectura

El flujo de datos se divide en las siguientes fases:

1. **Scraping**: Extracción de datos crudos desde un marketplace público de motos.
2. **Capa Bronze**: Ingesta de los datos en bruto en formato Delta (nuestra fuente inmutable).
3. **Capa Silver**: Limpieza de datos (parseo de strings a números, filtrado de nulos, estandarización de variables).
4. **Capa Gold**: Feature engineering (cálculo de la edad de la moto, agrupación por rangos de kilómetros) para preparar los datos para el algoritmo.
5. **Machine Learning**: Entrenamiento de un modelo `RandomForestRegressor` usando PySpark MLlib. Todo el tracking de experimentos, métricas y guardado del modelo físico se gestiona con **MLflow**.

## 🛠️ Tech Stack

* Python 3.10
* PySpark 3.5.1 + Delta Spark 3.1.0
* MLflow (MLOps)
* Docker & Docker Compose

## 🚀 Cómo usarlo en local

Como el proyecto corre sobre Docker, incluye volúmenes mapeados. Esto significa que puedes editar el código en tu editor favorito y los cambios se reflejarán dentro del contenedor en tiempo real.

### 1. Levantar el entorno

```bash
docker compose up -d
```
Este comando arranca el contenedor de Spark y monta tu código fuente.

### 2. Ejecutar el pipeline

Una vez levantado, ejecuta el orquestador principal:

```bash
docker compose exec pipeline python run_pipeline.py
```

El script pasará por todas las capas (Scraping -> Bronze -> Silver -> Gold -> ML). Si estás debugeando, puedes comentar las fases que no necesites en `run_pipeline.py`.

### 3. Ver los modelos (MLflow)

El tracking de MLflow se expone en el puerto 5000. Para ver el histórico de entrenamientos, comparar métricas (RMSE, MAE, R2) o descargarte el modelo final, entra en:

👉 **http://localhost:5000**
