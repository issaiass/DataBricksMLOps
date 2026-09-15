"""Seed Unity Catalog with the UCI Adult Census Income tables.

Downloads adult.data (train) and adult.test (holdout) from
https://archive.ics.uci.edu/dataset/2/adult
and writes the raw Delta table used by data checks / features.
"""

from __future__ import annotations

from src.n00_shared.dataset import LABEL_COL, POSITIVE_CLASS, load_adult_pandas
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    spark = _spark()
    pdf = load_adult_pandas()
    if settings.label_col != LABEL_COL:
        raise RuntimeError(f"expected label {LABEL_COL}, got {settings.label_col}")
    df = spark.createDataFrame(pdf)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(settings.raw_fq)
    )
    n = df.count()
    pos = df.filter(f"{LABEL_COL} = 1").count()
    print(
        f"seeded {settings.raw_fq} rows={n} positives(>50K)={pos} "
        f"negatives(<=50K)={n - pos} positive_class={POSITIVE_CLASS}"
    )


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
