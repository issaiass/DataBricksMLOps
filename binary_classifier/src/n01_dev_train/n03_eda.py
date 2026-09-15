"""Dev-only EDA for UCI Adult: target definition, missingness, class balance."""

from __future__ import annotations

import json

from pyspark.sql import functions as F

from src.n00_shared.dataset import (
    ADULT_CATEGORICAL,
    ADULT_NUMERIC,
    FEATURE_COLS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_DEFINITION,
    UCI_DATASET_URL,
)
from src.n00_shared.runtime import Settings, assert_dev_ml_allowed, configure_mlflow, load_settings


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run(settings: Settings) -> None:
    assert_dev_ml_allowed(settings)
    configure_mlflow(settings)
    spark = _spark()
    df = spark.table(settings.raw_fq)
    n = df.count()
    pos = df.filter(F.col(settings.label_col) == 1).count()
    cat_card = {}
    for col in ADULT_CATEGORICAL:
        if col in df.columns:
            cat_card[col] = df.select(col).distinct().count()
    num_summary = {}
    for col in ADULT_NUMERIC:
        if col in df.columns:
            stats = df.select(
                F.mean(col).alias("mean"),
                F.min(col).alias("min"),
                F.max(col).alias("max"),
            ).first()
            num_summary[col] = {
                "mean": float(stats["mean"]) if stats["mean"] is not None else None,
                "min": float(stats["min"]) if stats["min"] is not None else None,
                "max": float(stats["max"]) if stats["max"] is not None else None,
            }
    profile = {
        "dataset": "UCI Adult Census Income",
        "source": UCI_DATASET_URL,
        "target_definition": TARGET_DEFINITION,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "rows": n,
        "positives": pos,
        "negatives": n - pos,
        "positive_rate": (pos / n) if n else None,
        "dtypes": {f.name: f.dataType.simpleString() for f in df.schema.fields},
        "numeric": ADULT_NUMERIC,
        "categorical": ADULT_CATEGORICAL,
        "categorical_cardinality": cat_card,
        "numeric_summary": num_summary,
        "feature_cols": FEATURE_COLS,
        "id_col": settings.id_col,
        "label_col": settings.label_col,
        "leakage_label_in_features": settings.label_col in FEATURE_COLS,
        "dup_keys": df.groupBy(settings.id_col).count().filter(F.col("count") > 1).count(),
    }
    if profile["leakage_label_in_features"]:
        raise RuntimeError("label leaked into feature list")
    missingness = {}
    for col in df.columns:
        missingness[col] = df.select(F.mean(F.col(col).isNull().cast("double"))).first()[0]
    profile["missingness"] = missingness
    rows = [(k, json.dumps(v, default=str)) for k, v in profile.items()]
    out = spark.createDataFrame(rows, ["metric", "value"])
    (
        out.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(settings.eda_fq)
    )
    print(f"eda wrote {settings.eda_fq} adult pos_rate={profile['positive_rate']}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
