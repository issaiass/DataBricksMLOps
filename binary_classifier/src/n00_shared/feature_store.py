"""Unity Catalog Feature Store helpers for the Adult example.

Feature table = FeatureEngineeringClient.create_table / write_table.
Labels stay in a separate Delta table (keys + split + label) and are the
training-set spine. Models logged with fe.log_model(training_set=...) must
be scored with fe.score_batch (keys only).
"""

from __future__ import annotations

import time
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
            F.when(F.col("hours_per_week") > 40, F.lit(1)).otherwise(F.lit(0)).cast("long"),
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
    out["hours_over_40"] = (pd.to_numeric(out["hours_per_week"], errors="coerce") > 40).astype("int64")
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


def _online_store_state(store) -> str:
    state = getattr(store, "state", None)
    if state is None and isinstance(store, dict):
        state = store.get("state")
    return str(state or "").upper()


def _pipeline_name(obj) -> str:
    """Lakeflow pipeline name. Jobs UI may prefix ``Synced table: ``; that is not the UC table."""
    if obj is None:
        return ""
    name = getattr(obj, "name", None)
    if not name:
        spec = getattr(obj, "spec", None)
        name = getattr(spec, "name", "") or ""
    text = str(name).strip()
    lower = text.lower()
    prefix = "synced table:"
    if lower.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def _spec_update_kwargs(spec) -> dict:
    if spec is None:
        return {}
    if hasattr(spec, "as_dict"):
        data = spec.as_dict()
    elif isinstance(spec, dict):
        data = dict(spec)
    else:
        return {}
    for key in ("id", "pipeline_id"):
        data.pop(key, None)
    return {key: value for key, value in data.items() if value is not None}


def find_online_sync_pipeline_id(client, settings, desired: str) -> str:
    """Resolve the Lakeflow pipeline Databricks creates for ``publish_table``."""
    online_fq = str(settings.online_feature_fq).strip().lower()
    wanted = str(desired or "").strip().lower()
    mentioned = []
    for pipeline in client.pipelines.list_pipelines():
        name = _pipeline_name(pipeline).lower()
        pid = str(getattr(pipeline, "pipeline_id", "") or "").strip()
        if not pid:
            continue
        if wanted and name == wanted:
            return pid
        if online_fq and online_fq in name:
            mentioned.append(pipeline)
    if len(mentioned) == 1:
        return str(mentioned[0].pipeline_id)
    prefixed = [
        p
        for p in mentioned
        if _pipeline_name(p).lower().startswith(online_fq)
    ]
    if len(prefixed) == 1:
        return str(prefixed[0].pipeline_id)
    if mentioned:
        raise RuntimeError(
            f"multiple sync pipelines mention {settings.online_feature_fq}; "
            "rename or delete extras before publish"
        )
    return ""


def rename_online_sync_pipeline(settings, *, pipeline_id: str = "", client=None) -> str:
    """Replace Databricks' ``<table> <random>`` pipeline name with the Jobs UI pattern."""
    desired = str(getattr(settings, "online_sync_pipeline_name", "") or "").strip()
    if not desired:
        print("online_sync_pipeline_name empty; skip pipeline rename")
        return ""
    if client is None:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
    pid = str(pipeline_id or "").strip()
    if not pid:
        pid = find_online_sync_pipeline_id(client, settings, desired)
    if not pid:
        print(f"no sync pipeline found for {settings.online_feature_fq}; skip rename")
        return ""
    current = client.pipelines.get(pid)
    current_name = _pipeline_name(current)
    if current_name == desired:
        print(f"sync pipeline already named {desired}")
        return pid
    try:
        client.pipelines.update(pipeline_id=pid, name=desired)
    except Exception as first:
        try:
            kwargs = _spec_update_kwargs(getattr(current, "spec", None))
            kwargs["name"] = desired
            client.pipelines.update(pipeline_id=pid, **kwargs)
        except Exception as second:
            msg = str(second or first)
            if "DatabaseSyncTable" in msg or "only updates to" in msg:
                print(
                    "Databricks does not allow renaming Online Feature Store "
                    f"(DatabaseSyncTable) pipelines; left {current_name!r}. "
                    f"Desired name was {desired!r}."
                )
                return pid
            raise
    print(f"renamed sync pipeline {current_name!r} -> {desired!r}")
    return pid


def wait_online_store(fe, name: str, *, timeout_s: int = 1800, poll_s: int = 15):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = fe.get_online_store(name=name)
        state = _online_store_state(last)
        if "AVAILABLE" in state or state in {"READY", "RUNNING"}:
            return last
        if any(token in state for token in ("FAIL", "ERROR", "DELET")):
            raise RuntimeError(f"online store {name} failed: {last}")
        time.sleep(poll_s)
    raise TimeoutError(f"online store {name} never AVAILABLE: {last}")


def ensure_online_features(spark, settings) -> None:
    """Publish the offline Feature Store table to Databricks Online Feature Store.

    Model Serving automatic lookup requires this. Empty ``online_store_name`` skips.
    """
    store_name = str(getattr(settings, "online_store_name", "") or "").strip()
    if not store_name:
        print("online_store_name empty; skip online feature publish")
        return
    online_fq = str(getattr(settings, "online_feature_fq", "") or "").strip()
    if not online_fq or online_fq.startswith("."):
        raise RuntimeError("online_catalog and online_feature_table are required to publish")
    spark.sql(
        f"ALTER TABLE {settings.feature_fq} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    )
    spark.sql(
        f"ALTER TABLE {settings.feature_fq} ALTER COLUMN {settings.id_col} SET NOT NULL"
    )
    fe = fe_client()
    store = None
    try:
        store = fe.get_online_store(name=store_name)
    except Exception:
        store = None
    if store is None:
        capacity = str(getattr(settings, "online_store_capacity", "") or "CU_1").strip() or "CU_1"
        fe.create_online_store(name=store_name, capacity=capacity)
        print(f"created online store {store_name} capacity={capacity}")
    store = wait_online_store(fe, store_name)
    publish_kwargs = {
        "online_store": store,
        "source_table_name": settings.feature_fq,
        "online_table_name": online_fq,
    }
    published = None
    try:
        published = fe.publish_table(**publish_kwargs, publish_mode="TRIGGERED")
    except TypeError:
        published = fe.publish_table(**publish_kwargs)
    print(f"published {settings.feature_fq} -> {online_fq} store={store_name}")
    pipeline_id = str(getattr(published, "pipeline_id", "") or "").strip()
    rename_online_sync_pipeline(settings, pipeline_id=pipeline_id)


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
