import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType

from motos_ml.config import DeltaConfig

logger = logging.getLogger(__name__)


def bronze_to_silver(spark: SparkSession, config: DeltaConfig) -> DataFrame:
    df = spark.table(config.bronze_full)
    try:
        df_moto_ocasion = spark.table(config.bronze_moto_ocasion_full)
        df = df.unionByName(df_moto_ocasion, allowMissingColumns=True)
        logger.info(f"Unidas tablas {config.bronze_full} y {config.bronze_moto_ocasion_full}")
    except Exception as e:
        logger.warning(f"No se pudo leer la tabla {config.bronze_moto_ocasion_full}: {e}")

    df = (
        df
        .withColumn("km", F.col("km").cast(IntegerType()))
        .withColumn("anio", F.col("anio").cast(IntegerType()))
        .withColumn("precio", F.col("precio").cast(FloatType()))
        .withColumn("cilindrada_cc", F.col("cilindrada_cc").cast(IntegerType()))
        .withColumn("marca", F.upper(F.trim(F.col("marca"))))
        .withColumn("modelo", F.trim(F.col("modelo")))
    )

    df = df.dropna(subset=["marca", "modelo", "anio", "km", "precio"])
    df = df.dropDuplicates(["url_anuncio"])

    current_year = spark.sql("SELECT year(current_date())").collect()[0][0]
    df = df.filter(
        (F.col("precio").between(100, 100_000))
        & (F.col("anio").between(1980, current_year))
        & (F.col("km") >= 0)
    )

    return df


def write_silver(df: DataFrame, config: DeltaConfig) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.silver_full)
    logger.info("Tabla silver actualizada: %s", config.silver_full)