"""Standalone Adult prediction job (not train).

Loads raw + Feature Store rows, scores with ``fe.score_batch`` (default) or a
Model Serving endpoint, then MERGE/writes ``predictions_table``.
``score`` is P(income > 50K).
"""

from __future__ import annotations

from src.n00_shared.dataset import MODEL_FEATURE_COLS
from src.n00_shared.feature_store import score_keys
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings
from src.n00_shared.serving import query_endpoint_scores


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def resolve_predict_mode(settings: Settings) -> str:
    raw = str(settings.predict_mode or "batch").strip().lower()
    if raw in {"serving", "endpoint", "model_serving"}:
        return "serving"
    if raw in {"batch", "score_batch", ""}:
        return "batch"
    raise RuntimeError("predict_mode must be batch (default) or serving")


def resolve_model(settings: Settings):
    from mlflow.tracking import MlflowClient

    alias = settings.batch_alias
    if alias not in {"dev", "prod"}:
        raise RuntimeError("batch_alias must be dev or prod")
    model_name = settings.source_model_name if alias == "dev" else settings.dest_model_name
    client = MlflowClient(registry_uri="databricks-uc")
    try:
        version = str(client.get_model_version_by_alias(model_name, alias).version)
    except Exception as exc:
        return model_name, alias, None, exc
    return model_name, alias, version, None


def load_raw_and_features(spark, settings: Settings):
    """Load raw keys and the Feature Store scoring frame (same contract as train)."""
    for table in (settings.raw_fq, settings.feature_fq):
        if not spark.catalog.tableExists(table):
            raise RuntimeError(f"prediction job missing table {table}")
    raw = spark.table(settings.raw_fq)
    features = spark.table(settings.feature_fq)
    if settings.id_col not in raw.columns:
        raise RuntimeError(f"raw table missing {settings.id_col}")
    missing_fs = [c for c in [settings.id_col] + list(MODEL_FEATURE_COLS) if c not in features.columns]
    if missing_fs:
        raise RuntimeError(f"feature store missing columns {missing_fs}")
    if settings.label_col in features.columns:
        raise RuntimeError(f"feature store {settings.feature_fq} must not contain {settings.label_col}")
    keys = raw.select(settings.id_col).dropDuplicates()
    featured = keys.join(features, on=settings.id_col, how="inner")
    n_raw = keys.count()
    n_feat = featured.count()
    if n_feat < 1:
        raise RuntimeError(
            f"no overlapping keys between {settings.raw_fq} and {settings.feature_fq}"
        )
    print(
        f"prediction inputs raw={settings.raw_fq} rows={n_raw} "
        f"features={settings.feature_fq} scored_keys={n_feat}"
    )
    return featured


def score_via_batch(featured, settings: Settings, *, model_name: str, alias: str):
    uri = f"models:/{model_name}@{alias}"
    keys = featured.select(settings.id_col)
    scored_raw = score_keys(uri, keys, result_type="double")
    if "prediction" in scored_raw.columns:
        scored_raw = scored_raw.withColumnRenamed("prediction", "score")
    if "score" not in scored_raw.columns:
        raise RuntimeError(f"score_batch missing score/prediction; got {scored_raw.columns}")
    print(f"scored {keys.count()} rows via fe.score_batch {uri}")
    return scored_raw, uri


def score_via_serving(spark, featured, settings: Settings, *, version: str):
    pdf = featured.select(settings.id_col, *MODEL_FEATURE_COLS).toPandas()
    ids = [str(v) for v in pdf[settings.id_col].tolist()]
    records = pdf[list(MODEL_FEATURE_COLS)].to_dict(orient="records")
    scores = query_endpoint_scores(settings.endpoint_name, records)
    rows = list(zip(ids, [float(s) for s in scores]))
    scored = spark.createDataFrame(rows, schema=f"{settings.id_col} string, score double")
    print(f"scored {len(rows)} rows via serving endpoint {settings.endpoint_name} v{version}")
    return scored, f"endpoints:/{settings.endpoint_name}"


def write_predictions(spark, scored_raw, settings: Settings, *, version: str) -> None:
    from pyspark.sql import functions as F

    scored = (
        scored_raw.withColumn("scored_at", F.current_timestamp())
        .withColumn("model_version", F.lit(version))
        .select(settings.id_col, "score", "scored_at", "model_version")
    )
    if spark.catalog.tableExists(settings.predictions_fq):
        scored.createOrReplaceTempView("incoming_scores")
        spark.sql(
            f"""
            MERGE INTO {settings.predictions_fq} AS t
            USING incoming_scores AS s
            ON t.{settings.id_col} = s.{settings.id_col}
            WHEN MATCHED THEN UPDATE SET
              t.score = s.score,
              t.scored_at = s.scored_at,
              t.model_version = s.model_version
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    else:
        scored.write.format("delta").mode("overwrite").saveAsTable(settings.predictions_fq)
    print(f"wrote predictions {settings.predictions_fq} rows={scored.count()} version={version}")


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    spark = _spark()
    mode = resolve_predict_mode(settings)
    model_name, alias, version, alias_err = resolve_model(settings)
    if version is None:
        print(f"prediction no-op: alias @{alias} not set on {model_name}: {alias_err}")
        return
    featured = load_raw_and_features(spark, settings)
    if mode == "serving":
        scored_raw, source = score_via_serving(spark, featured, settings, version=version)
    else:
        scored_raw, source = score_via_batch(featured, settings, model_name=model_name, alias=alias)
    write_predictions(spark, scored_raw, settings, version=version)
    print(f"prediction job done mode={mode} source={source} alias=@{alias}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
