"""Absolute evaluate on Adult holdout (adult.test). mode=dev or prod_gate.

Holdout keys come from the label spine; features are joined from Feature Store
via the model's fe.log_model lineage (score_batch).
"""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.n00_shared.dataset import MODEL_FEATURE_COLS, NEGATIVE_CLASS, POSITIVE_CLASS, SPLIT_COL
from src.n00_shared.feature_store import require_raw_and_feature_store, score_keys
from src.n00_shared.runtime import (
    Settings,
    configure_mlflow,
    get_task_value,
    load_settings,
    mlflow_client,
    set_task_value,
)


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _latest_monitor_ok(spark, settings: Settings) -> None:
    try:
        exists = spark.catalog.tableExists(settings.monitor_fq)
    except Exception:
        exists = False
    if not exists:
        return
    row = (
        spark.table(settings.monitor_fq)
        .orderBy("as_of", ascending=False)
        .limit(1)
        .collect()
    )
    if not row:
        return
    if str(row[0]["gate_ok"]).lower() != "true":
        raise RuntimeError(f"monitor gate blocked evaluate: {row[0]['reason']}")


def _model_uri_and_name(settings: Settings) -> tuple[str, str, str]:
    if settings.evaluate_mode == "prod_gate":
        version = get_task_value("copy_register", "model_version")
        return settings.dest_model_name, version, f"models:/{settings.dest_model_name}/{version}"
    version = get_task_value("train", "model_version")
    return settings.source_model_name, version, f"models:/{settings.source_model_name}/{version}"


def run(settings: Settings) -> None:
    from pyspark.sql import functions as F

    if settings.evaluate_mode not in {"dev", "prod_gate"}:
        raise RuntimeError(f"unsupported evaluate mode {settings.evaluate_mode}")
    if settings.evaluate_mode == "dev" and settings.catalog == settings.prod_catalog:
        raise RuntimeError("dev evaluate must not use prod_catalog")
    configure_mlflow(settings)
    spark = _spark()
    _latest_monitor_ok(spark, settings)
    require_raw_and_feature_store(spark, settings, split="holdout")
    model_name, version, uri = _model_uri_and_name(settings)
    labels = (
        spark.table(settings.label_fq)
        .filter(F.col(SPLIT_COL) == "holdout")
        .select(settings.id_col, settings.label_col)
    )
    n = labels.count()
    if n < settings.min_eval_rows:
        raise RuntimeError(f"eval rows {n} < min_eval_rows {settings.min_eval_rows}")
    if settings.label_col not in labels.columns:
        raise RuntimeError("empty or missing labels")
    keys = labels.select(settings.id_col)
    scored = score_keys(uri, keys, result_type="double")
    joined = labels.join(scored, on=settings.id_col, how="inner")
    pred_col = "prediction" if "prediction" in joined.columns else None
    if pred_col is None:
        raise RuntimeError(f"score_batch missing prediction column; got {joined.columns}")
    pdf = joined.select(settings.label_col, pred_col).toPandas()
    if pdf[settings.label_col].isna().all():
        raise RuntimeError("empty or missing labels")
    y = pdf[settings.label_col].astype(int)
    import numpy as np

    scores = np.array(pdf[pred_col])
    if scores.ndim > 1:
        scores = scores[:, -1]
    if len(scores) == 0:
        raise RuntimeError("empty metrics")
    preds = (scores >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "f1": float(f1_score(y, preds)),
        "accuracy": float(accuracy_score(y, preds)),
        "eval_rows": float(n),
        "positive_rate": float((y == 1).mean()),
    }
    if any(v is None or v != v for v in metrics.values()):
        raise RuntimeError("missing/empty metrics")
    if metrics["roc_auc"] < settings.eval_min_roc_auc:
        raise RuntimeError(f"roc_auc {metrics['roc_auc']} below {settings.eval_min_roc_auc}")
    client = mlflow_client()
    for k, v in metrics.items():
        client.set_model_version_tag(model_name, version, k, str(v))
    client.set_model_version_tag(model_name, version, "feature_contract", ",".join(MODEL_FEATURE_COLS))
    if settings.evaluate_mode == "prod_gate":
        pass
    set_task_value("eval_roc_auc", str(metrics["roc_auc"]))
    set_task_value("model_version", version)
    print(
        f"evaluate mode={settings.evaluate_mode} adult {POSITIVE_CLASS}=1 "
        f"{NEGATIVE_CLASS}=0 {metrics}"
    )


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
