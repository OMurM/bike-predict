import logging
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel, GBTRegressor, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from typing import cast

from motos_ml.config import DeltaConfig, MLConfig

logger = logging.getLogger(__name__)

CAT_COLS = ["marca", "tipo_normalizado", "km_bucket", "distintivo_ambiental", "iva_deducible", "origen"]
NUM_COLS = ["anio", "edad_anios", "km", "cilindrada_cc", "num_plazas", "num_llaves"]
TARGET = "precio"


def build_base_stages() -> list:
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
    return indexers + encoders + [assembler]


def train_and_log(spark: SparkSession, delta_config: DeltaConfig, ml_config: MLConfig) -> None:
    df = spark.table(delta_config.gold_full).dropna(subset=NUM_COLS + [TARGET])
    
    # Filtro opcional por origen
    if hasattr(ml_config, "origen_filter") and ml_config.origen_filter and ml_config.origen_filter.lower() != "all":
        df = df.filter(F.col("origen") == ml_config.origen_filter.lower())
        
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
    evaluator = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="rmse")
    base_stages = build_base_stages()

    # Definir los modelos a evaluar
    models_to_train = []
    
    # 1. Random Forest
    rf = RandomForestRegressor(featuresCol="features", labelCol=TARGET, seed=42)
    rf_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [100, 200]) \
        .addGrid(rf.maxDepth, [8, 12]) \
        .build()
    models_to_train.append(("Random Forest", rf, rf_grid))

    # 2. Gradient Boosted Trees (Descomentar para incluir en el entrenamiento)
    # gbt = GBTRegressor(featuresCol="features", labelCol=TARGET, seed=42)
    # gbt_grid = ParamGridBuilder() \
    #     .addGrid(gbt.maxIter, [50, 100]) \
    #     .addGrid(gbt.maxDepth, [5, 8]) \
    #     .build()
    # models_to_train.append(("Gradient Boost", gbt, gbt_grid))

    logger.info("Iniciando Hyperparameter Tuning con CrossValidator para %d algoritmos...", len(models_to_train))

    for model_name, estimator, param_grid in models_to_train:
        logger.info(f"Entrenando {model_name}...")
        
        pipeline = Pipeline(stages=base_stages + [estimator])
        
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42
        )

        with mlflow.start_run(run_name=model_name):
            cv_model = crossval.fit(train_df)
            best_pipeline = cast(PipelineModel, cv_model.bestModel)
            best_model_stage = best_pipeline.stages[-1]
            
            predictions = best_pipeline.transform(test_df)
            
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
            
            with tempfile.NamedTemporaryFile(prefix=f"test_samples_{model_name.replace(' ', '_')}_", suffix=".csv", delete=False) as tmp:
                sample_preds.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, "evaluation_examples")

            # 2. Análisis de Errores
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
            
            with tempfile.NamedTemporaryFile(prefix=f"worst_preds_{model_name.replace(' ', '_')}_", suffix=".csv", delete=False) as tmp:
                error_df.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, "worst_predictions")

            # 3. Feature Importance
            try:
                importances = best_model_stage.featureImportances
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
                
                with tempfile.NamedTemporaryFile(prefix=f"feature_importance_{model_name.replace(' ', '_')}_", suffix=".csv", delete=False) as tmp:
                    imp_df.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name, "feature_importance")
            except Exception as e:
                logger.warning(f"No se pudieron extraer los nombres de las features: {e}")
            # =============================

            # Extraer params dinámicamente
            logged_params = {
                "model_type": model_name,
                "total_rows": total_rows,
                "origen_filter": ml_config.origen_filter if hasattr(ml_config, "origen_filter") else "all"
            }
            if isinstance(best_model_stage, RandomForestRegressionModel):
                logged_params["num_trees"] = best_model_stage.getNumTrees
                logged_params["max_depth"] = best_model_stage.getOrDefault("maxDepth")
            elif isinstance(best_model_stage, GBTRegressionModel):
                logged_params["max_iter"] = best_model_stage.getMaxIter()
                logged_params["max_depth"] = best_model_stage.getOrDefault("maxDepth")

            logger.info(f"[{model_name}] Métricas Finales Test -> RMSE={final_rmse:.2f} MAE={final_mae:.2f} R2={final_r2:.4f}")

            mlflow.log_params(logged_params)
            
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