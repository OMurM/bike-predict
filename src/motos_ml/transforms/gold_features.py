import logging
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from motos_ml.config import DeltaConfig

logger = logging.getLogger(__name__)

TIPOS_MOTO = ["naked", "trail", "scooter", "deportiva", "custom", "touring", "enduro", "otro"]


def silver_to_gold(spark: SparkSession, config: DeltaConfig) -> DataFrame:
    df = spark.table(config.silver_full)
    current_year = spark.sql("SELECT year(current_date())").collect()[0][0]

    df = (
        df
        .withColumn("edad_anios", F.lit(current_year) - F.col("anio"))
        .withColumn(
            "km_bucket",
            F.when(F.col("km") < 10_000, "0-10k")
            .when(F.col("km") < 30_000, "10k-30k")
            .when(F.col("km") < 60_000, "30k-60k")
            .otherwise("60k+")
        )
        .withColumn(
            "tipo_normalizado",
            F.when(F.lower(F.col("tipo")).isin(TIPOS_MOTO), F.lower(F.col("tipo")))
            .otherwise("otro")
        )
        .withColumn("log_precio", F.log(F.col("precio")))
        .withColumn("distintivo_ambiental", F.coalesce(F.col("distintivo_ambiental"), F.lit("desconocido")))
        .withColumn("num_plazas", F.coalesce(F.col("num_plazas"), F.lit(2)))
        .withColumn("num_llaves", F.coalesce(F.col("num_llaves"), F.lit(1)))
        .withColumn("iva_deducible", F.coalesce(F.col("iva_deducible").cast("string"), F.lit("false")))
    )

    return df.select(
        "marca", "modelo", "anio", "edad_anios",
        "km", "km_bucket", "tipo_normalizado",
        "cilindrada_cc", "potencia_cv", "ubicacion",
        "distintivo_ambiental", "num_plazas", "num_llaves", "iva_deducible",
        "origen", "precio", "log_precio", "url_anuncio"
    )


def write_gold(df: DataFrame, config: DeltaConfig) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.gold_full)
    logger.info("Tabla gold actualizada: %s", config.gold_full)