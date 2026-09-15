"""Smoke-score the Adult @dev model on Feature Store keys."""

from __future__ import annotations

from src.n00_shared.feature_store import score_keys
from src.n00_shared.runtime import Settings, assert_dev_ml_allowed, configure_mlflow, load_settings, mlflow_client
from src.n00_shared.serving import serving_config, upsert_endpoint


def run(settings: Settings) -> None:
    assert_dev_ml_allowed(settings)
    configure_mlflow(settings)
    client = mlflow_client()
    version = str(client.get_model_version_by_alias(settings.source_model_name, "dev").version)
    uri = f"models:/{settings.source_model_name}@dev"
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    if not spark.catalog.tableExists(settings.feature_fq):
        raise RuntimeError(f"feature store table missing: {settings.feature_fq}")
    keys = spark.table(settings.feature_fq).select(settings.id_col).limit(20)
    pred = score_keys(uri, keys, result_type="double")
    if pred.count() != keys.count():
        raise RuntimeError("smoke prediction length mismatch")
    try:
        upsert_endpoint(
            settings.endpoint_name,
            serving_config(settings.source_model_name, version, previous_version=None, canary_percent=100),
        )
        print(f"smoke pinned dev endpoint {settings.endpoint_name} to v{version}")
    except Exception as exc:
        print(f"smoke feature-store path ok; endpoint pin skipped: {exc}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
