"""Profile Adult feature/prediction tables and write monitor_status.gate_ok."""

from __future__ import annotations

from datetime import datetime, timezone

from src.n00_shared.dataset import MODEL_FEATURE_COLS
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings, job_run_id


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    spark = _spark()
    reason = "ok"
    gate_ok = True
    try:
        if not spark.catalog.tableExists(settings.feature_fq):
            raise RuntimeError("Adult feature store table missing")
        feats = spark.table(settings.feature_fq)
        missing = [c for c in MODEL_FEATURE_COLS if c not in feats.columns]
        if missing:
            raise RuntimeError(f"Adult feature contract mismatch: {missing}")
        n = feats.count()
        if n < settings.min_table_rows:
            raise RuntimeError(f"feature rows {n} < {settings.min_table_rows}")
        if spark.catalog.tableExists(settings.predictions_fq):
            pred_n = spark.table(settings.predictions_fq).count()
            if pred_n == 0:
                reason = "predictions empty"
                gate_ok = True
        else:
            reason = "predictions table not created yet"
    except Exception as exc:
        gate_ok = False
        reason = str(exc)
    status = spark.createDataFrame(
        [
            {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "gate_ok": gate_ok,
                "reason": reason,
                "job_run_id": job_run_id(),
            }
        ]
    )
    (
        status.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(settings.monitor_fq)
    )
    print(f"monitor gate_ok={gate_ok} reason={reason}")
    if not gate_ok:
        print("enqueue dev train via Jobs API with cooldown (not implemented in-process)")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
