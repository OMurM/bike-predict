from dataclasses import dataclass


@dataclass(frozen=True)
class ScraperConfig:
    base_url: str
    max_pages: int = 20
    request_delay_seconds: float = 1.5
    user_agent: str = (
        "Mozilla/5.0 (compatible; bike-predict/0.1; +https://github.com/OMurM/bike-predict)"
    )


@dataclass(frozen=True)
class DeltaConfig:
    catalog: str = "hive_metastore"
    schema: str = "motos_ml"
    bronze_table: str = "bronze_anuncios"
    bronze_moto_ocasion_table: str = "bronze_moto_ocasion"
    silver_table: str = "silver_motos"
    gold_table: str = "gold_features"

    @property
    def bronze_full(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.bronze_table}"

    @property
    def bronze_moto_ocasion_full(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.bronze_moto_ocasion_table}"

    @property
    def silver_full(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.silver_table}"

    @property
    def gold_full(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.gold_table}"


@dataclass(frozen=True)
class MLConfig:
    experiment_name: str = "/bike-predict/price-prediction"
    model_name: str = "motos-price-regressor"
    min_rows_to_train: int = 100
    test_size: float = 0.2
    random_state: int = 42
    origen_filter: str = "all"