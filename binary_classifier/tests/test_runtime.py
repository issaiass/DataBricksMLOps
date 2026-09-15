import os

import pytest

from src.n00_shared.runtime import (
    as_bool,
    assert_dev_ml_allowed,
    baked,
    fq,
    get_task_value,
    job_run_id,
    load_settings,
    operator_param,
    set_task_value,
)
from tests.conftest import make_settings


def test_as_bool_truthy_and_falsey():
    assert as_bool("true")
    assert as_bool("YES")
    assert as_bool("1")
    assert not as_bool("false")
    assert not as_bool("0")
    assert not as_bool("")


def test_fq_joins_catalog_schema_table():
    assert fq("ml_dev", "adult_income", "adult_raw") == "ml_dev.adult_income.adult_raw"


def test_settings_fq_properties_and_active_model():
    settings = make_settings()
    assert settings.raw_fq == "ml_dev.adult_income.adult_raw"
    assert settings.feature_fq.endswith("adult_features")
    assert settings.active_model == settings.source_model_name
    prod = make_settings(evaluate_mode="prod_gate", catalog="ml_prod")
    assert prod.active_model == prod.dest_model_name


def test_assert_dev_ml_allowed_refuses_prod_catalog():
    with pytest.raises(RuntimeError, match="must not use prod_catalog"):
        assert_dev_ml_allowed(make_settings(catalog="ml_prod"))
    with pytest.raises(RuntimeError, match="must be distinct"):
        assert_dev_ml_allowed(make_settings(dev_catalog="ml_prod", catalog="ml_dev"))
    assert_dev_ml_allowed(make_settings())


def test_baked_prefers_env_and_refuses_missing(monkeypatch):
    monkeypatch.setenv("DEV_CATALOG", "ml_dev")
    assert baked("DEV_CATALOG") == "ml_dev"
    monkeypatch.delenv("MISSING_BAKED_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_BAKED_KEY"):
        baked("MISSING_BAKED_KEY")
    assert baked("MISSING_BAKED_KEY", "fallback") == "fallback"


def test_operator_param_reads_env(monkeypatch):
    monkeypatch.setenv("SOURCE_MODEL_VERSION", "9")
    assert operator_param("source_model_version") == "9"
    monkeypatch.delenv("PROMOTION_REASON", raising=False)
    assert operator_param("promotion_reason", "") == ""


def test_task_values_round_trip_via_env(monkeypatch):
    monkeypatch.setenv("MLOPS_TASK_model_version", "12")
    assert get_task_value("copy_register", "model_version") == "12"
    set_task_value("model_version", "12")
    with pytest.raises(RuntimeError, match="missing task value"):
        get_task_value("missing_task", "no_such_key")
    assert get_task_value("missing_task", "no_such_key", "d") == "d"


def test_job_run_id_from_env(monkeypatch):
    monkeypatch.setenv("DB_JOB_ID", "job-42")
    assert job_run_id() == "job-42"


def test_load_settings_from_baked_env(monkeypatch):
    env = {
        "DEV_CATALOG": "ml_dev",
        "PROD_CATALOG": "ml_prod",
        "CATALOG": "ml_dev",
        "SCHEMA": "adult_income",
        "MODEL_NAME": "adult_income_clf",
        "SOURCE_MODEL_NAME": "ml_dev.adult_income.adult_income_clf",
        "DEST_MODEL_NAME": "ml_prod.adult_income.adult_income_clf",
        "RAW_TABLE": "adult_raw",
        "FEATURE_TABLE": "adult_features",
        "LABEL_TABLE": "adult_labels",
        "PREDICTIONS_TABLE": "adult_predictions",
        "INFERENCE_TABLE": "adult_inference_logs",
        "EDA_PROFILE_TABLE": "adult_eda_profile",
        "XAI_EXPLANATION_TABLE": "adult_xai",
        "MONITOR_STATUS_TABLE": "adult_monitor_status",
        "LABEL_COL": "income_gt_50k",
        "ID_COL": "row_id",
        "EVENT_TIME_COL": "",
        "COMPARE_METRIC": "roc_auc",
        "COMPARE_HIGHER_IS_BETTER": "true",
        "COMPARE_MARGIN": "0.01",
        "MIN_EVAL_ROWS": "200",
        "MIN_TABLE_ROWS": "200",
        "MAX_NULL_RATE": "0.4",
        "LABEL_WINDOW": "P365D",
        "EVAL_MIN_ROC_AUC": "0.7",
        "TRAIN_SEED": "42",
        "ALLOW_CHAMPION_FALLBACK": "false",
        "ALLOW_CANARY": "false",
        "CANARY_PERCENT": "10",
        "CANARY_WARMUP_S": "120",
        "ENDPOINT_NAME": "adult-income-clf-dev",
        "ONLINE_STORE_NAME": "adult-income-ofs-dev",
        "ONLINE_CATALOG": "ml_dev",
        "ONLINE_FEATURE_TABLE": "adult_features_online",
        "ONLINE_STORE_CAPACITY": "CU_1",
        "ONLINE_SYNC_PIPELINE_NAME": "[dev issaiass] adult-adult_income_clf-features-pipeline",
        "ENV_MANAGER": "local",
        "SECRET_SCOPE": "mlops-adult",
        "TRAIN_RUN_AS_SP": "train-sp",
        "JOB_RUN_AS_SP": "job-sp",
        "APPROVER_IDENTITIES": "human@example.com",
        "EXPERIMENT_PATH": "/Shared/mlops/binary_classifier/dev",
        "EVALUATE_MODE": "dev",
        "BATCH_ALIAS": "dev",
        "PREDICT_MODE": "batch",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    loaded = load_settings()
    assert loaded.catalog == "ml_dev"
    assert loaded.online_store_name == "adult-income-ofs-dev"
    assert loaded.online_sync_pipeline_name == "[dev issaiass] adult-adult_income_clf-features-pipeline"
    assert loaded.allow_champion_fallback is False
    assert loaded.compare_margin == 0.01
    assert os.environ["EVALUATE_MODE"] == "dev"
