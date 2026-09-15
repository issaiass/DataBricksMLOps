"""Train XGBoost on UCI Adult (dev only). Score is P(income > 50K).

Loads the official train split from the raw/label spine joined to the Feature
Store table via FeatureLookup, then logs with fe.log_model(training_set=...).
"""

from __future__ import annotations

import json

import mlflow
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.n00_shared.dataset import (
    ADULT_CATEGORICAL,
    ADULT_MODEL_NUMERIC,
    FEATURE_CONTRACT,
    MODEL_FEATURE_COLS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_DEFINITION,
    UCI_DATASET_URL,
)
from src.n00_shared.feature_store import fe_client, load_split_pandas, registered_version
from src.n00_shared.runtime import (
    Settings,
    assert_dev_ml_allowed,
    configure_mlflow,
    load_settings,
    mlflow_client,
    set_task_value,
    timestamped_run_name,
)


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _pipeline(seed: int) -> Pipeline:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, ADULT_MODEL_NUMERIC),
            ("cat", categorical, ADULT_CATEGORICAL),
        ]
    )
    clf = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def run(settings: Settings) -> None:
    assert_dev_ml_allowed(settings)
    configure_mlflow(settings)
    spark = _spark()
    training_set, pdf = load_split_pandas(settings, split="train")
    X = pdf[MODEL_FEATURE_COLS]
    y = pdf[settings.label_col]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale = (neg / pos) if pos else 1.0
    model = _pipeline(settings.train_seed)
    model.named_steps["clf"].set_params(scale_pos_weight=scale)

    mlflow.xgboost.autolog(log_models=False, log_input_examples=False)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False)
    fe = fe_client()

    with mlflow.start_run(run_name=timestamped_run_name()) as run:
        model.fit(X, y)
        example = X.head(5)
        signature = infer_signature(example, model.predict_proba(example)[:, 1])
        mlflow.log_param("dataset", "UCI Adult Census Income")
        mlflow.log_param("dataset_url", UCI_DATASET_URL)
        mlflow.log_param("target_definition", TARGET_DEFINITION)
        mlflow.log_param("positive_class", POSITIVE_CLASS)
        mlflow.log_param("negative_class", NEGATIVE_CLASS)
        mlflow.log_param("training_table", settings.feature_fq)
        mlflow.log_param("training_label_table", settings.label_fq)
        mlflow.log_param("training_raw_table", settings.raw_fq)
        try:
            hist = spark.sql(f"DESCRIBE HISTORY {settings.feature_fq}").limit(1).collect()
            if hist:
                mlflow.log_param("training_table_version", str(hist[0]["version"]))
        except Exception:
            mlflow.log_param("training_table_version", "unknown")
        mlflow.log_dict(FEATURE_CONTRACT, "feature_contract.json")
        mlflow.log_param("feature_columns", json.dumps(MODEL_FEATURE_COLS))
        mlflow.log_param("label_col", settings.label_col)
        mlflow.log_param("train_seed", settings.train_seed)
        mlflow.log_param("feature_store", "databricks.feature_engineering")
        mlflow.log_metric("train_rows", float(len(pdf)))
        log_kwargs = {
            "model": model,
            "artifact_path": "model",
            "flavor": mlflow.sklearn,
            "training_set": training_set,
            "registered_model_name": settings.source_model_name,
        }
        try:
            info = fe.log_model(
                **log_kwargs, signature=signature, input_example=example
            )
        except TypeError:
            info = fe.log_model(**log_kwargs)
        client = mlflow_client()
        version = registered_version(info, client=client, model_name=settings.source_model_name)
        client.set_registered_model_alias(settings.source_model_name, "challenger", version)
        client.set_registered_model_alias(settings.source_model_name, "dev", version)
        client.set_model_version_tag(settings.source_model_name, version, "eda_table", settings.eda_fq)
        client.set_model_version_tag(settings.source_model_name, version, "train_run_id", run.info.run_id)
        set_task_value("model_version", version)
        set_task_value("mlflow_run_id", run.info.run_id)
        print(
            f"registered {settings.source_model_name} v{version} "
            f"aliases=@challenger,@dev feature_store={settings.feature_fq}"
        )


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
