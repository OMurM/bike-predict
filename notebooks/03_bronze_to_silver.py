# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Bronze → Silver
# MAGIC
# MAGIC Limpia, normaliza y deduplica los datos bronze.
# MAGIC - Cast de tipos correctos
# MAGIC - Normalización de marca/modelo
# MAGIC - Deduplicación por url_anuncio
# MAGIC - Filtros de rango (precio, año, km)

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.insert(0, "../src")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from motos_ml.config import DeltaConfig
from motos_ml.transforms.silver_cleaning import bronze_to_silver, write_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Transformación

# COMMAND ----------

delta_cfg = DeltaConfig()
df_silver = bronze_to_silver(spark, delta_cfg)

print(f"Filas silver: {df_silver.count()}")
display(df_silver.limit(20)) # type: ignore

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Distribución de datos limpios

# COMMAND ----------

from pyspark.sql import functions as F

print("--- Marcas (top 10) ---")
display(df_silver.groupBy("marca").count().orderBy(F.desc("count")).limit(10)) # type: ignore

print("--- Tipos ---")
display(df_silver.groupBy("tipo").count().orderBy(F.desc("count"))) # type: ignore

print("--- Estadísticas de precio ---")
display(df_silver.select( # type: ignore
    F.min("precio").alias("min"),
    F.percentile_approx("precio", 0.25).alias("p25"),
    F.percentile_approx("precio", 0.50).alias("p50"),
    F.percentile_approx("precio", 0.75).alias("p75"),
    F.max("precio").alias("max"),
    F.avg("precio").alias("media"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Escritura en Delta silver

# COMMAND ----------

write_silver(df_silver, delta_cfg)
print(f"✅ Silver actualizado: {delta_cfg.silver_full}")