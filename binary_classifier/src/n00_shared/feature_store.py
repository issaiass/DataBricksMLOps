"""Unity Catalog Feature Store helpers for the Adult example.

Feature table = FeatureEngineeringClient.create_table / write_table.
Labels stay in a separate Delta table (keys + split + label) and are the
training-set spine. Models logged with fe.log_model(training_set=...) must
be scored with fe.score_batch (keys only).
"""

from __future__ import annotations

from typing import Any

from src.n00_shared.dataset import (
    ENGINEERED_FEATURE_COLS,
    FEATURE_COLS,
    MODEL_FEATURE_COLS,
    SPLIT_COL,
)


def fe_client():
    from databricks.feature_engineering import FeatureEngineeringClient

    return FeatureEngineeringClient(model_registry_uri="databricks-uc")


def with_engineered_features(df):
    """Add derived Adult features. Input must already have the 14 census columns."""
    from pyspark.sql import functions as F

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"raw Adult frame missing census columns {missing}")
    return (
        df.withColumn("capital_net", F.col("capital_gain") - F.col("capital_loss"))
        .withColumn(
            "hours_over_40",
            F.when(F.col("hours_per_week") > 40, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("log_fnlwgt", F.log1p(F.col("fnlwgt").cast("double")))
        .withColumn(
            "education_num_sq",
            F.col("education_num").cast("double") * F.col("education_num").cast("double"),
        )
    )


def engineer_adult_pandas(pdf):
    """Local/test twin of ``with_engineered_features`` (no Spark)."""
    import numpy as np
    import pandas as pd

    out = pdf.copy()
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(f"raw Adult frame missing census columns {missing}")
    out["capital_net"] = out["capital_gain"] - out["capital_loss"]
    out["hours_over_40"] = (pd.to_numeric(out["hours_per_week"], errors="coerce") > 40).astype(int)
    out["log_fnlwgt"] = np.log1p(pd.to_numeric(out["fnlwgt"], errors="coerce").astype(float))
    edu = pd.to_numeric(out["education_num"], errors="coerce").astype(float)
    out["education_num_sq"] = edu * edu
    return out


def feature_frame(raw_df, settings) -> Any:
    featured = with_engineered_features(raw_df)
    return featured.select(settings.id_col, *[c for c in MODEL_FEATURE_COLS])


def publish_feature_table(spark, settings, raw_df) -> None:
    """Create or overwrite the UC feature table. Does not store the label."""
    from pyspark.sql import functions as F

    df = feature_frame(raw_df, settings)
    dup = df.groupBy(settings.id_col).count().filter(F.col("count") > 1).count()
    if dup:
        raise RuntimeError(f"feature store keys are not unique: {dup} duplicates")
    fe = fe_client()
    spark.sql(f"DROP TABLE IF EXISTS {settings.feature_fq}")
    fe.create_table(
        name=settings.feature_fq,
        primary_keys=[settings.id_col],
        df=df,
        description=(
            "UCI Adult Feature Store table: 14 census columns plus "
            f"{', '.join(ENGINEERED_FEATURE_COLS)}. Label is not stored here."
        ),
    )


def write_label_table(spark, raw_df, settings) -> None:
    if SPLIT_COL not in raw_df.columns:
        raise RuntimeError(f"raw table {settings.raw_fq} is missing {SPLIT_COL}")
    labels = raw_df.select(settings.id_col, SPLIT_COL, settings.label_col)
    (
        labels.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(settings.label_fq)
    )
    n = labels.count()
    print(f"labels {settings.label_fq} rows={n}")


def require_raw_and_feature_store(spark, settings, *, split: str) -> None:
    """Fail closed unless raw, Feature Store, and labels line up for ``split``."""
    from pyspark.sql import functions as F

    for table in (settings.raw_fq, settings.feature_fq, settings.label_fq):
        if not spark.catalog.tableExists(table):
            raise RuntimeError(f"missing table {table}")
    raw = spark.table(settings.raw_fq)
    fs = spark.table(settings.feature_fq)
    labels = spark.table(settings.label_fq)
    if settings.label_col in fs.columns:
        raise RuntimeError(
            f"feature store {settings.feature_fq} must not contain {settings.label_col}"
        )
    missing_fs = [c for c in [settings.id_col] + MODEL_FEATURE_COLS if c not in fs.columns]
    if missing_fs:
        raise RuntimeError(f"feature store missing columns {missing_fs}")
    missing_lbl = [c for c in (settings.id_col, SPLIT_COL, settings.label_col) if c not in labels.columns]
    if missing_lbl:
        raise RuntimeError(f"label table missing columns {missing_lbl}")
    if SPLIT_COL not in raw.columns:
        raise RuntimeError(f"raw table {settings.raw_fq} is missing {SPLIT_COL}")
    raw_keys = raw.filter(F.col(SPLIT_COL) == split).select(settings.id_col)
    raw_n = raw_keys.count()
    if raw_n < 1:
        raise RuntimeError(f"raw table has no '{split}' rows")
    joined = raw_keys.join(fs, on=settings.id_col, how="left")
    missing_join = joined.filter(F.col(MODEL_FEATURE_COLS[0]).isNull()).count()
    if missing_join:
        raise RuntimeError(
            f"{missing_join} {split} raw keys missing from feature store {settings.feature_fq}"
        )
    label_n = labels.filter(F.col(SPLIT_COL) == split).count()
    if label_n != raw_n:
        raise RuntimeError(
            f"{split} label rows {label_n} != raw rows {raw_n}; refuse mismatched spine"
        )
    print(
        f"feature-store check ok split={split} raw={settings.raw_fq} "
        f"fs={settings.feature_fq} labels={settings.label_fq} rows={raw_n}"
    )


def create_split_training_set(settings, *, split: str):
    """Join label spine for ``split`` to the Feature Store table via FeatureLookup."""
    from databricks.feature_engineering import FeatureLookup
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.getOrCreate()
    require_raw_and_feature_store(spark, settings, split=split)
    spine = (
        spark.table(settings.label_fq)
        .filter(F.col(SPLIT_COL) == split)
        .select(settings.id_col, settings.label_col)
    )
    fe = fe_client()
    return fe.create_training_set(
        df=spine,
        feature_lookups=[
            FeatureLookup(
                table_name=settings.feature_fq,
                lookup_key=settings.id_col,
                feature_names=list(MODEL_FEATURE_COLS),
            )
        ],
        label=settings.label_col,
        exclude_columns=[settings.id_col],
    )


def load_split_pandas(settings, *, split: str):
    training_set = create_split_training_set(settings, split=split)
    pdf = training_set.load_df().toPandas()
    missing = [c for c in MODEL_FEATURE_COLS if c not in pdf.columns]
    if missing:
        raise RuntimeError(f"training set missing feature columns {missing}")
    if settings.label_col not in pdf.columns:
        raise RuntimeError("training set missing label")
    return training_set, pdf


def registered_version(result, client=None, model_name: str = "") -> str:
    for attr in ("registered_model_version", "version"):
        val = getattr(result, attr, None)
        if val is not None and str(val).strip():
            return str(val)
    if client is not None and model_name:
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            return str(latest.version)
    raise RuntimeError("fe.log_model did not return a registered model version")


def score_keys(model_uri: str, keys_df, *, result_type: str = "double"):
    fe = fe_client()
    return fe.score_batch(model_uri=model_uri, df=keys_df, result_type=result_type)
