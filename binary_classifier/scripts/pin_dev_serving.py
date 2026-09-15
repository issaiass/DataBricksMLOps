# Databricks notebook source
from pyspark.sql import SparkSession

from src.n00_shared.feature_store import ensure_online_features
from src.n00_shared.runtime import configure_mlflow, load_settings, mlflow_client
from src.n00_shared.serving import serving_config, upsert_endpoint

settings = load_settings()
spark = SparkSession.builder.getOrCreate()
ensure_online_features(spark, settings)
configure_mlflow(settings)
client = mlflow_client()
version = str(client.get_model_version_by_alias(settings.source_model_name, "dev").version)
upsert_endpoint(
    settings.endpoint_name,
    serving_config(settings.source_model_name, version, previous_version=None, canary_percent=100),
)
msg = f"pinned {settings.endpoint_name} v{version} online={settings.online_feature_fq}"
print(msg)
try:
    dbutils.notebook.exit(msg)  # noqa: F821
except Exception:
    pass
