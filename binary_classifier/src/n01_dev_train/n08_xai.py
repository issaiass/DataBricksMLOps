"""Global importances for the UCI Adult challenger (dev only; not a promotion gate)."""

from __future__ import annotations

from src.n00_shared.dataset import MODEL_FEATURE_COLS, POSITIVE_CLASS
from src.n00_shared.feature_store import load_split_pandas
from src.n00_shared.runtime import Settings, assert_dev_ml_allowed, configure_mlflow, load_settings, mlflow_client


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run(settings: Settings) -> None:
    assert_dev_ml_allowed(settings)
    configure_mlflow(settings)
    spark = _spark()
    client = mlflow_client()
    version = client.get_model_version_by_alias(settings.source_model_name, "challenger").version
    uri = f"models:/{settings.source_model_name}/{version}"
    import mlflow

    _, holdout = load_split_pandas(settings, split="holdout")
    sample = holdout[MODEL_FEATURE_COLS].head(200)
    y = holdout[settings.label_col].head(200)
    rows = []
    try:
        from sklearn.inspection import permutation_importance

        raw = mlflow.sklearn.load_model(uri)
        imp = permutation_importance(raw, sample, y, n_repeats=3, random_state=settings.train_seed)
        for name, mean in zip(MODEL_FEATURE_COLS, imp.importances_mean):
            rows.append((str(version), name, float(mean), "permutation"))
    except Exception as exc:
        rows.append((str(version), "error", 0.0, str(exc)))
    try:
        import shap

        raw = mlflow.sklearn.load_model(uri)
        background = sample.sample(min(50, len(sample)), random_state=settings.train_seed)
        explainer = shap.Explainer(raw.predict_proba, background)
        shap.Explainer  # keep import used
        _ = explainer
    except Exception:
        pass
    out = spark.createDataFrame(rows, ["model_version", "feature", "importance", "method"])
    (
        out.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(settings.xai_fq)
    )
    client.set_model_version_tag(settings.source_model_name, str(version), "xai_table", settings.xai_fq)
    print(f"xai wrote {settings.xai_fq} for {uri} (Adult P({POSITIVE_CLASS}))")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
