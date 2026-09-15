"""Copy a go UCI Adult model version from ml_dev into ml_prod (no training)."""

from __future__ import annotations

import mlflow

from src.n00_shared.protocol import require_compare_go, resolve_copy_source
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings, mlflow_client, operator_param, set_task_value


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _metastore_id(spark, catalog: str):
    try:
        rows = spark.sql(f"DESCRIBE CATALOG EXTENDED `{catalog}`").collect()
        for row in rows:
            info = " ".join(str(x) for x in row)
            if "metastore" in info.lower():
                return info
    except Exception:
        return None
    return None


def _tag_map(client, name: str, version: str) -> dict:
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    if isinstance(tags, dict):
        return tags
    return {t.key: t.value for t in tags}


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    pin = operator_param("source_model_version", "")
    reason = operator_param("promotion_reason", "")
    source_uri, resolved, used_fallback = resolve_copy_source(
        settings.source_model_name,
        pin,
        reason,
        settings.allow_champion_fallback,
    )
    spark = _spark()
    src_m = _metastore_id(spark, settings.dev_catalog)
    dst_m = _metastore_id(spark, settings.prod_catalog)
    if src_m and dst_m and src_m != dst_m:
        raise RuntimeError("copy_register fail-closed: catalogs are on different metastores")
    client = mlflow_client()
    if used_fallback:
        src_version = str(client.get_model_version_by_alias(settings.source_model_name, "champion").version)
    else:
        src_version = resolved
    tags = _tag_map(client, settings.source_model_name, src_version)
    require_compare_go(tags)
    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.register_model(source_uri if not used_fallback else f"models:/{settings.source_model_name}/{src_version}", settings.dest_model_name)
    dest_version = str(mv.version)
    client.set_registered_model_alias(settings.dest_model_name, "challenger", dest_version)
    client.set_model_version_tag(settings.dest_model_name, dest_version, "source_model_name", settings.source_model_name)
    client.set_model_version_tag(settings.dest_model_name, dest_version, "source_uri", f"models:/{settings.source_model_name}/{src_version}")
    client.set_model_version_tag(settings.dest_model_name, dest_version, "source_model_version", src_version)
    if used_fallback:
        client.set_model_version_tag(settings.dest_model_name, dest_version, "source_alias", "champion")
    set_task_value("model_version", dest_version)
    print(f"copy_register dest v{dest_version} from source v{src_version}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
