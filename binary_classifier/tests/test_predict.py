from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from src.n00_shared.runtime import Settings
from src.n00_shared.serving import flatten_endpoint_scores
from src.n04_ops.n01_batch import resolve_predict_mode


def _settings(**kwargs) -> Settings:
    base = dict(
        dev_catalog="ml_dev",
        prod_catalog="ml_prod",
        catalog="ml_dev",
        schema="adult_income",
        model_name="adult_income_clf",
        source_model_name="ml_dev.adult_income.adult_income_clf",
        dest_model_name="ml_prod.adult_income.adult_income_clf",
        raw_table="adult_raw",
        feature_table="adult_features",
        label_table="adult_labels",
        predictions_table="adult_predictions",
        inference_table="adult_inference_logs",
        eda_profile_table="adult_eda_profile",
        xai_explanation_table="adult_xai",
        monitor_status_table="adult_monitor_status",
        label_col="income_gt_50k",
        id_col="row_id",
        event_time_col="",
        compare_metric="roc_auc",
        compare_higher_is_better=True,
        compare_margin=0.0,
        min_eval_rows=200,
        min_table_rows=200,
        max_null_rate=0.4,
        label_window="P365D",
        eval_min_roc_auc=0.7,
        train_seed=42,
        allow_champion_fallback=False,
        allow_canary=False,
        canary_percent=10,
        canary_warmup_s=120,
        endpoint_name="adult-income-clf-dev",
        online_store_name="adult-income-ofs-dev",
        online_catalog="ml_dev",
        online_feature_table="adult_features_online",
        online_store_capacity="CU_1",
        online_sync_pipeline_name="[dev issaiass] adult-adult_income_clf-features-pipeline",
        env_manager="local",
        secret_scope="mlops-adult",
        train_run_as_sp="train-sp",
        job_run_as_sp="job-sp",
        approver_identities="human@example.com",
        experiment_path="/Shared/mlops/binary_classifier/[dev issaiass] uc-adult-xgb",
        evaluate_mode="dev",
        batch_alias="dev",
    )
    base.update(kwargs)
    return Settings(**base)


def test_predict_mode_defaults_to_batch():
    assert resolve_predict_mode(_settings()) == "batch"
    assert resolve_predict_mode(_settings(predict_mode="score_batch")) == "batch"


def test_predict_mode_serving_aliases():
    assert resolve_predict_mode(_settings(predict_mode="serving")) == "serving"
    assert resolve_predict_mode(_settings(predict_mode="endpoint")) == "serving"


def test_predict_mode_unknown_refuses():
    with pytest.raises(RuntimeError, match="predict_mode"):
        resolve_predict_mode(_settings(predict_mode="spark_udf"))


def test_flatten_endpoint_scores_shapes():
    assert flatten_endpoint_scores({"predictions": [0.1, 0.9]}) == [0.1, 0.9]
    assert flatten_endpoint_scores({"predictions": [{"score": 0.4}]}) == [0.4]
    assert flatten_endpoint_scores({"predictions": [{"prediction": [0.2, 0.8]}]}) == [0.8]


def test_flatten_endpoint_scores_empty_refuses():
    with pytest.raises(RuntimeError, match="empty"):
        flatten_endpoint_scores(None)
