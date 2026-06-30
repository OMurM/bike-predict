# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Exploración — POC API mundimoto
# MAGIC
# MAGIC Valida que la API responde correctamente y que los DTOs se parsean bien.
# MAGIC **No escribe nada en Delta.** Úsalo para verificar antes de ingestar.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.insert(0, "../src")

from motos_ml.config import ScraperConfig
from motos_ml.scraping.client import fetch_motorbikes_page
from motos_ml.scraping.parser import parse_all

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validar que la API responde

# COMMAND ----------

config = ScraperConfig(base_url="https://api.mundimoto.com/mundimoto-api")

data = fetch_motorbikes_page(offset=0, config=config)

if not data or not data.get("motorbikes"):
    raise RuntimeError("La API no ha devuelto datos. Revisa conectividad o cambios en la URL.")

print(f"✅ API OK — motos en primera página: {len(data['motorbikes'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspeccionar un registro crudo

# COMMAND ----------

import json
print(json.dumps(data["motorbikes"][0], indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Parsear a DTOs y mostrar DataFrame

# COMMAND ----------

motos = parse_all(data["motorbikes"])
print(f"DTOs válidos: {len(motos)} / {len(data['motorbikes'])} totales")

import pandas as pd
df = pd.DataFrame([vars(m) for m in motos])
display(df) # type: ignore

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distribución rápida por tipo y marca

# COMMAND ----------

print("--- Por tipo ---")
print(df["tipo"].value_counts().to_string())

print("\n--- Por marca (top 10) ---")
print(df["marca"].value_counts().head(10).to_string())

print("\n--- Rango de precios ---")
print(df["precio"].describe().to_string())