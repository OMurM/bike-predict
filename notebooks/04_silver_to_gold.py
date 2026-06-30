# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Silver → Gold (Features)
# MAGIC
# MAGIC Construye el dataset de features listo para ML:
# MAGIC - edad_anios: current_year - anio
# MAGIC - km_bucket: tramos de km
# MAGIC - tipo_normalizado: lowercase y agrupado
# MAGIC - log_precio: target transformado (opcional)

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.insert(0, "../src")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from motos_ml.config import DeltaConfig
from motos_ml.transforms.gold_features import silver_to_gold, write_gold

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Construcción de features

# COMMAND ----------

delta_cfg = DeltaConfig()
df_gold = silver_to_gold(spark, delta_cfg)

print(f"Filas gold: {df_gold.count()}")
display(df_gold.limit(20)) # type: ignore

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validación de features

# COMMAND ----------

from pyspark.sql import functions as F

# Comprobamos que no hay nulos en columnas clave de ML
key_cols = ["marca", "tipo_normalizado", "km_bucket", "anio", "edad_anios", "km", "precio"]
for col in key_cols:
    nulls = df_gold.filter(F.col(col).isNull()).count()
    status = "✅" if nulls == 0 else f"⚠️  {nulls} nulos"
    print(f"  {col}: {status}")

# COMMAND ----------

print("--- Distribución km_bucket ---")
display(df_gold.groupBy("km_bucket").count().orderBy("km_bucket")) # type: ignore

print("--- Distribución tipo_normalizado ---")
display(df_gold.groupBy("tipo_normalizado").count().orderBy(F.desc("count"))) # type: ignore

print("--- Correlación edad_anios / precio ---")
print(df_gold.stat.corr("edad_anios", "precio"))

print("--- Correlación km / precio ---")
print(df_gold.stat.corr("km", "precio"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Escritura en Delta gold

# COMMAND ----------

write_gold(df_gold, delta_cfg)
print(f"✅ Gold actualizado: {delta_cfg.gold_full}")