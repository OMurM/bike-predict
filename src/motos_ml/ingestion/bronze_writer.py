import logging
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from motos_ml.dto import MotoDTO
from motos_ml.config import DeltaConfig

logger = logging.getLogger(__name__)


def write_to_bronze(motos: List[MotoDTO], spark: SparkSession, config: DeltaConfig, table_name: str) -> int:
    if not motos:
        logger.info("No hay motos para persistir en bronze.")
        return 0

    valid_motos = [m for m in motos if m.is_valid()]
    invalid_count = len(motos) - len(valid_motos)
    if invalid_count > 0:
        logger.warning("%d anuncios descartados por validación.", invalid_count)

    if not valid_motos:
        return 0

    rows = [vars(m) for m in valid_motos]
    # Guardamos el count antes de crear el DF para no pagar una acción extra
    count = len(rows)

    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, BooleanType
    schema = StructType([
        StructField("marca", StringType(), True),
        StructField("modelo", StringType(), True),
        StructField("anio", IntegerType(), True),
        StructField("km", IntegerType(), True),
        StructField("tipo", StringType(), True),
        StructField("precio", DoubleType(), True),
        StructField("cilindrada_cc", IntegerType(), True),
        StructField("potencia_cv", IntegerType(), True),
        StructField("ubicacion", StringType(), True),
        StructField("url_anuncio", StringType(), True),
        StructField("descripcion", StringType(), True),
        StructField("distintivo_ambiental", StringType(), True),
        StructField("num_plazas", IntegerType(), True),
        StructField("num_llaves", IntegerType(), True),
        StructField("iva_deducible", BooleanType(), True),
        StructField("origen", StringType(), True),
        StructField("ingestion_date", DateType(), True)
    ])
    df = spark.createDataFrame(rows, schema=schema)
    # Sobreescribimos ingestion_date con el valor real de Spark (no el del DTO)
    df = df.withColumn("ingestion_date", F.current_date())

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.catalog}.{config.schema}")
    df.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable(
        table_name
    )

    logger.info("Escritas %d filas en %s", count, table_name)
    return count