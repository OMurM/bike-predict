# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Ingesta Bronze — API mundimoto
# MAGIC
# MAGIC Scrapea todas las páginas de la API y persiste en la tabla Delta bronze
# MAGIC particionada por `ingestion_date`. Es idempotente: se puede re-ejecutar
# MAGIC el mismo día sin duplicar datos (append por fecha).

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys, logging
sys.path.insert(0, "../src")
logging.basicConfig(level=logging.INFO)

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from motos_ml.config import ScraperConfig, DeltaConfig
from motos_ml.scraping.client import fetch_all_motorbikes
from motos_ml.scraping.parser import parse_all
from motos_ml.ingestion.bronze_writer import write_to_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración

# COMMAND ----------

scraper_cfg = ScraperConfig(
    base_url="https://api.mundimoto.com/mundimoto-api",
    max_pages=60,            # 60 × 40 = hasta 2.400 motos
    request_delay_seconds=0.8,
)
delta_cfg = DeltaConfig()

print(f"Bronze table: {delta_cfg.bronze_full}")
print(f"Max páginas:  {scraper_cfg.max_pages}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scraping

# COMMAND ----------

raw_motos = fetch_all_motorbikes(scraper_cfg)
print(f"Registros crudos obtenidos: {len(raw_motos)}")

if not raw_motos:
    raise RuntimeError("El scraping no devolvió datos. Abortando para no escribir bronze vacío.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Parseo a DTOs

# COMMAND ----------

motos = parse_all(raw_motos)
descartadas = len(raw_motos) - len(motos)
print(f"DTOs válidos:   {len(motos)}")
print(f"Descartadas:    {descartadas}")

if len(motos) == 0:
    raise RuntimeError("Ningún DTO válido tras el parseo. Revisa parser.py.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Escritura en Delta bronze

# COMMAND ----------

total_written = write_to_bronze(motos, spark, delta_cfg)
print(f"✅ Filas escritas en bronze: {total_written}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificación post-escritura

# COMMAND ----------

df_check = spark.table(delta_cfg.bronze_full)
print(f"Total filas en tabla bronze: {df_check.count()}")
display( # type: ignore
    df_check
    .groupBy("ingestion_date")
    .count()
    .orderBy("ingestion_date", ascending=False)
    .limit(10)
)