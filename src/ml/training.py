import logging
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from motos_ml.config import DeltaConfig, MLConfig

logger = logging.getLogger(__name__)

CAT_COLS = ["marca", "tipo_normalizado", "km_bucket"]
NUM_COLS = ["anio", "edad_anios", "km", "cilindrada_cc"]
TARGET = "precio"


def build_pipeline() -> Pipeline:
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CAT_COLS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in CAT_COLS
    ]
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in CAT_COLS] + NUM_COLS,
        outputCol="features",
        handleInvalid="skip",
    )
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=TARGET,
        numTrees=100,
        maxDepth=8,
        seed=42,
    )
    return Pipeline(stages=indexers + encoders + [assembler, rf])


def train_and_log(spark: SparkSession, delta_config: DeltaConfig, ml_config: MLConfig) -> None:
    df = spark.table(delta_config.gold_full).dropna(subset=NUM_COLS + [TARGET])
    total_rows = df.count()

    if total_rows < ml_config.min_rows_to_train:
        logger.warning(
            "Solo %d filas disponibles, mínimo requerido: %d. Abortando.",
            total_rows, ml_config.min_rows_to_train,
        )
        return

    train_df, test_df = df.randomSplit(
        [1 - ml_config.test_size, ml_config.test_size],
        seed=ml_config.random_state,
    )

    mlflow.set_experiment(ml_config.experiment_name)

    with mlflow.start_run():
        pipeline = build_pipeline()
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        evaluator = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction")
        rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        mae  = evaluator.setMetricName("mae").evaluate(predictions)
        r2   = evaluator.setMetricName("r2").evaluate(predictions)

        mlflow.log_params({"num_trees": 100, "max_depth": 8, "total_rows": total_rows})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model",
            registered_model_name=ml_config.model_name,
        )

        logger.info("Entrenamiento OK. RMSE=%.2f MAE=%.2f R2=%.4f", rmse, mae, r2)