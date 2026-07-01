import logging
from pyspark.sql import SparkSession

# Importar tu lógica desde src/
from motos_ml.config import ScraperConfig, DeltaConfig, MLConfig
from motos_ml.scraping.client import fetch_all_motorbikes
from motos_ml.scraping.parser import parse_all
from motos_ml.ingestion.bronze_writer import write_to_bronze
from motos_ml.transforms.silver_cleaning import bronze_to_silver, write_silver
from motos_ml.transforms.gold_features import silver_to_gold, write_gold
from motos_ml.ml.training import train_and_log

logging.basicConfig(level=logging.INFO)

def run_all():
    # 1. Configurar un Spark local (necesitaría el paquete delta-spark)
    spark = SparkSession.builder \
        .appName("MotoPipelineLocal") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.warehouse.dir", "/app/spark-warehouse") \
        .config("javax.jdo.option.ConnectionURL", "jdbc:derby:/app/spark-warehouse/metastore_db;create=true") \
        .config("spark.driver.memory", "4g") \
        .enableHiveSupport() \
        .getOrCreate()
        
    cfg_scrap = ScraperConfig(base_url="https://api.mundimoto.com/mundimoto-api", max_pages=300)
    cfg_delta = DeltaConfig(catalog="spark_catalog")
    cfg_ml = MLConfig()

    print("🚀 Iniciando Pipeline End-to-End...")

    # BRONZE
    raw_motos = fetch_all_motorbikes(cfg_scrap)
    motos_dto = parse_all(raw_motos)
    write_to_bronze(motos_dto, spark, cfg_delta)
    
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
    run_all()
