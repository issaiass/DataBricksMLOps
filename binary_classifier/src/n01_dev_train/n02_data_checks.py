"""Fail-closed checks on the seeded UCI Adult raw table."""

from __future__ import annotations

from pyspark.sql import functions as F

from src.n00_shared.dataset import FEATURE_COLS, LABEL_COL, SPLIT_COL
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings, set_task_value


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    spark = _spark()
    table = settings.raw_fq
    if not spark.catalog.tableExists(table):
        raise RuntimeError(f"missing table {table}")
    df = spark.table(table)
    required = [settings.id_col] + FEATURE_COLS + [settings.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing Adult columns {missing}")
    if settings.label_col != LABEL_COL:
        raise RuntimeError(f"label_col must be {LABEL_COL} (1=>50K, 0=<=50K)")
    labels = {int(r[0]) for r in df.select(settings.label_col).distinct().collect() if r[0] is not None}
    if not labels.issubset({0, 1}):
        raise RuntimeError(f"Adult label must be binary 0/1, got {labels}")
    if 0 not in labels or 1 not in labels:
        raise RuntimeError(f"Adult table must contain both classes <=50K(0) and >50K(1), got {labels}")
    n = df.count()
    if n < settings.min_table_rows:
        raise RuntimeError(f"row count {n} < min_table_rows {settings.min_table_rows}")
    dup = df.groupBy(settings.id_col).count().filter(F.col("count") > 1).count()
    if dup:
        raise RuntimeError(f"duplicate keys: {dup}")
    for col in [settings.id_col, settings.label_col]:
        null_rate = df.select(F.mean(F.col(col).isNull().cast("double"))).first()[0]
        if null_rate is None or float(null_rate) > settings.max_null_rate:
            raise RuntimeError(f"null rate on {col} is {null_rate}")
    if SPLIT_COL in df.columns:
        splits = {r[0] for r in df.select(SPLIT_COL).distinct().collect()}
        if "train" not in splits or "holdout" not in splits:
            raise RuntimeError(f"UCI official split missing train/holdout, got {splits}")
    if settings.event_time_col:
        if settings.event_time_col not in df.columns:
            raise RuntimeError(f"event_time_col {settings.event_time_col} missing")
        future = df.filter(F.col(settings.event_time_col) > F.current_timestamp()).count()
        if future:
            raise RuntimeError("future event_time rows in train slice")
    set_task_value("data_checks_ok", "true")
    set_task_value("raw_rows", str(n))
    print(f"data_checks ok table={table} rows={n} adult_binary_label={settings.label_col}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
