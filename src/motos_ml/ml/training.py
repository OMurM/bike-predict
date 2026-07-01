import logging
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from typing import cast

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
    
    rf = RandomForestRegressor(featuresCol="features", labelCol=TARGET, seed=42)
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

    logger.info("Iniciando Hyperparameter Tuning con CrossValidator (RandomForest)...")

    # 1. Definir el Pipeline base (con Random Forest vacío)
    pipeline = build_pipeline()
    
    # 2. Extraer el RF del pipeline para construir el Grid
    rf_stage = cast(RandomForestRegressor, pipeline.getStages()[-1])

    # 3. Construir la rejilla (Grid) de hiperparámetros a probar
    # Esto probará 2x2 = 4 modelos distintos
    paramGrid = ParamGridBuilder() \
        .addGrid(rf_stage.numTrees, [100, 200]) \
        .addGrid(rf_stage.maxDepth, [8, 12]) \
        .build()

    # 4. Definir cómo se va a evaluar (queremos minimizar el RMSE o maximizar R2, CrossValidator usa RMSE por defecto si es métrica de error)
    evaluator = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="rmse")

    # 5. Configurar el CrossValidator (Evaluación cruzada de 3 pliegues)
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42
    )

    with mlflow.start_run():
        # Esto va a entrenar 4 modelos * 3 folds = 12 veces por debajo. Puede tardar un poco.
        cv_model = crossval.fit(train_df)
        
        # El CrossValidator ya ha guardado el MEJOR modelo internamente.
        best_pipeline = cast(PipelineModel, cv_model.bestModel)
        best_rf = cast(RandomForestRegressionModel, best_pipeline.stages[-1])
        
        # Hacemos las predicciones finales contra el conjunto de Test real
        predictions = best_pipeline.transform(test_df)
        
        # Calculamos todas las métricas
        final_rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        final_mae  = evaluator.setMetricName("mae").evaluate(predictions)
        final_r2   = evaluator.setMetricName("r2").evaluate(predictions)

        # === ANÁLISIS DATA SCIENCE ===
        import tempfile
        import pandas as pd
        
        # 1. Extraer 50 ejemplos al azar
        sample_preds = predictions.select(
            "marca", "modelo", "tipo_normalizado", "cilindrada_cc", 
            "anio", "km", "precio", "prediction", "url_anuncio"
        ) \
        .orderBy(F.rand(seed=42)) \
        .limit(50).toPandas()
        sample_preds["prediction"] = sample_preds["prediction"].round(2)
        
        with tempfile.NamedTemporaryFile(prefix="test_samples_", suffix=".csv", delete=False) as tmp:
            sample_preds.to_csv(tmp.name, index=False)
            mlflow.log_artifact(tmp.name, "evaluation_examples")

        # 2. Análisis de Errores (Las 100 peores predicciones)
        error_df = predictions.withColumn("error_absoluto", F.abs(F.col("precio") - F.col("prediction"))) \
                              .withColumn("error_porcentual", (F.col("error_absoluto") / F.col("precio")) * 100) \
                              .select(
                                  "marca", "modelo", "tipo_normalizado", "anio", "km", 
                                  "precio", "prediction", "error_absoluto", "error_porcentual", "url_anuncio"
                              ) \
                              .orderBy(F.col("error_absoluto").desc()) \
                              .limit(100).toPandas()
        error_df["prediction"] = error_df["prediction"].round(2)
        error_df["error_absoluto"] = error_df["error_absoluto"].round(2)
        error_df["error_porcentual"] = error_df["error_porcentual"].round(2)
        
        with tempfile.NamedTemporaryFile(prefix="worst_preds_", suffix=".csv", delete=False) as tmp:
            error_df.to_csv(tmp.name, index=False)
            mlflow.log_artifact(tmp.name, "worst_predictions")

        # 3. Feature Importance (Importancia de Variables)
        importances = best_rf.featureImportances
        try:
            attrs = predictions.schema["features"].metadata["ml_attr"]["attrs"]
            feature_names = {}
            for attr_type in ["numeric", "binary", "nominal"]:
                if attr_type in attrs:
                    for attr in attrs[attr_type]:
                        feature_names[attr["idx"]] = attr["name"]
            
            imp_list = []
            for i, imp in enumerate(importances.toArray()):
                name = feature_names.get(i, f"feature_{i}")
                imp_list.append({"Feature": name, "Importance": imp})
                
            imp_df = pd.DataFrame(imp_list).sort_values(by="Importance", ascending=False)
            imp_df["Importance"] = imp_df["Importance"].round(4)
            
            with tempfile.NamedTemporaryFile(prefix="feature_importance_", suffix=".csv", delete=False) as tmp:
                imp_df.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, "feature_importance")
        except Exception as e:
            logger.warning(f"No se pudieron extraer los nombres de las features: {e}")
        # =============================

        # Loggeamos el mejor hiperparámetro encontrado
        best_numTrees = best_rf.getNumTrees
        best_maxDepth = best_rf.getOrDefault("maxDepth")

        logger.info(f"Mejor configuración encontrada: numTrees={best_numTrees}, maxDepth={best_maxDepth}")
        logger.info(f"Métricas Finales Test -> RMSE={final_rmse:.2f} MAE={final_mae:.2f} R2={final_r2:.4f}")

        mlflow.log_params({
            "model_type": "Random Forest",
            "num_trees": best_numTrees,
            "max_depth": best_maxDepth,
            "total_rows": total_rows
        })
        
        mlflow.log_metrics({
            "rmse": final_rmse,
            "mae": final_mae,
            "r2": final_r2
        })
        
        mlflow.spark.log_model(
            spark_model=best_pipeline,
            artifact_path="model",
            registered_model_name=ml_config.model_name,
        )