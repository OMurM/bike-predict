# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Entrenamiento del modelo
# MAGIC
# MAGIC Entrena un RandomForestRegressor sobre la tabla gold.
# MAGIC Trackea parámetros, métricas y modelo en MLflow.
# MAGIC Se aborta automáticamente si hay menos de `min_rows_to_train` filas.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.insert(0, "../src")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from motos_ml.config import DeltaConfig, MLConfig
from motos_ml.ml.training import train_and_log

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Entrenamiento

# COMMAND ----------

delta_cfg = DeltaConfig()
ml_cfg = MLConfig()

print(f"Experimento MLflow: {ml_cfg.experiment_name}")
print(f"Gold table:         {delta_cfg.gold_full}")
print(f"Mínimo filas:       {ml_cfg.min_rows_to_train}")

# COMMAND ----------

train_and_log(spark, delta_cfg, ml_cfg)
print("✅ Entrenamiento finalizado. Revisa MLflow para métricas y modelo registrado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ver últimas ejecuciones en MLflow

# COMMAND ----------

import mlflow

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(ml_cfg.experiment_name)

if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=5,
    )
    for run in runs:
        m = run.data.metrics
        print(
            f"Run {run.info.run_id[:8]}  "
            f"RMSE={m.get('rmse', 0):.2f}  "
            f"MAE={m.get('mae', 0):.2f}  "
            f"R2={m.get('r2', 0):.4f}"
        )
else:
    print("Experimento aún no creado en MLflow.")