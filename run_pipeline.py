import logging
from pyspark.sql import SparkSession

# Importar tu lógica desde src/
from motos_ml.config import ScraperConfig, DeltaConfig, MLConfig
from motos_ml.scraping.client import fetch_all_motorbikes
from motos_ml.scraping.moto_ocasion_client import fetch_all_moto_ocasion
from motos_ml.scraping.parser import parse_all, parse_moto_ocasion
from motos_ml.ingestion.bronze_writer import write_to_bronze
from motos_ml.transforms.silver_cleaning import bronze_to_silver, write_silver
from motos_ml.transforms.gold_features import silver_to_gold, write_gold
from motos_ml.ml.training import train_and_log

logging.basicConfig(level=logging.INFO)

import argparse

def run_all(skip_scraping: bool = False, origen_filter: str = "all"):
    # 1. Configurar un Spark local (necesitaría el paquete delta-spark)
    spark = SparkSession.builder \
        .appName("MotoPipelineLocal") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.warehouse.dir", "/app/spark-warehouse") \
        .config("javax.jdo.option.ConnectionURL", "jdbc:derby:/app/spark-warehouse/metastore_db;create=true") \
        .enableHiveSupport() \
        .getOrCreate()
        
    cfg_scrap = ScraperConfig(base_url="https://api.mundimoto.com/mundimoto-api", max_pages=300, request_delay_seconds=0.1)
    cfg_delta = DeltaConfig(catalog="spark_catalog")
    cfg_ml = MLConfig(origen_filter=origen_filter)

    print("🚀 Iniciando Pipeline End-to-End...")

    if not skip_scraping:
        print("🌐 Descargando datos (Scraping)...")
        # BRONZE: MUNDIMOTO
        raw_motos = fetch_all_motorbikes(cfg_scrap)
        motos_dto = parse_all(raw_motos)
        write_to_bronze(motos_dto, spark, cfg_delta, cfg_delta.bronze_full)

        # BRONZE: MOTO-OCASION
        cfg_scrap_mo = ScraperConfig(base_url="https://www.moto-ocasion.com", max_pages=15, request_delay_seconds=0.1) # Limitar a 15 páginas x ~20 = ~300 motos para pruebas
        raw_motos_mo = fetch_all_moto_ocasion(cfg_scrap_mo)
        motos_dto_mo = [parse_moto_ocasion(r) for r in raw_motos_mo if parse_moto_ocasion(r)]
        write_to_bronze(motos_dto_mo, spark, cfg_delta, cfg_delta.bronze_moto_ocasion_full)
    else:
        print("⏭️ Omitiendo scraping. Se usarán los datos existentes en Bronze.")
    
    # SILVER
    df_silver = bronze_to_silver(spark, cfg_delta)
    write_silver(df_silver, cfg_delta)
    
    # GOLD
    df_gold = silver_to_gold(spark, cfg_delta)
    write_gold(df_gold, cfg_delta)
    
    # ML TRAINING
    train_and_log(spark, cfg_delta, cfg_ml)
    
    print("✅ Pipeline ejecutado con éxito!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scraping", action="store_true", help="Salta la fase de descarga y entrena con los datos cacheados")
    parser.add_argument("--origen", type=str, default="all", choices=["all", "mundimoto", "moto-ocasion"], help="Filtra las motos a entrenar por su origen")
    args = parser.parse_args()
    
    run_all(skip_scraping=args.skip_scraping, origen_filter=args.origen)
