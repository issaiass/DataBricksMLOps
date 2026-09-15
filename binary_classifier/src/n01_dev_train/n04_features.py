"""Materialize Adult Feature Store table + label spine.

Reads the seeded raw table (adult.data / adult.test split), engineers extra
features, and publishes them with FeatureEngineeringClient. Labels stay in a
separate Delta table so the Feature Store table has no target leakage.
"""

from __future__ import annotations

from pyspark.sql import functions as F

from src.n00_shared.dataset import ENGINEERED_FEATURE_COLS, SPLIT_COL
from src.n00_shared.feature_store import (
    ensure_online_features,
    publish_feature_table,
    write_label_table,
)
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    spark = _spark()
    if not spark.catalog.tableExists(settings.raw_fq):
        raise RuntimeError(f"missing raw table {settings.raw_fq}; run seed first")
    df = spark.table(settings.raw_fq)
    if SPLIT_COL in df.columns:
        raw = df.withColumn(
            SPLIT_COL,
            F.when(F.col(SPLIT_COL).isin("train", "holdout"), F.col(SPLIT_COL)).otherwise(
                F.lit("holdout")
            ),
        )
    elif settings.event_time_col:
        from pyspark.sql.window import Window

        wdf = df.withColumn(
            "_rn",
            F.row_number().over(
                Window.orderBy(F.col(settings.event_time_col).asc(), F.col(settings.id_col))
            ),
        )
        n = wdf.count()
        cut = int(n * 0.8)
        raw = wdf.withColumn(
            SPLIT_COL, F.when(F.col("_rn") <= cut, F.lit("train")).otherwise(F.lit("holdout"))
        ).drop("_rn")
    else:
        raw = df.withColumn(
            SPLIT_COL,
            F.when((F.hash(F.col(settings.id_col)) % 10) < 8, F.lit("train")).otherwise(
                F.lit("holdout")
            ),
        )
    write_label_table(spark, raw, settings)
    publish_feature_table(spark, settings, raw)
    ensure_online_features(spark, settings)
    print(
        f"feature store {settings.feature_fq} pk={settings.id_col} "
        f"engineered={ENGINEERED_FEATURE_COLS} labels {settings.label_fq}"
    )


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
