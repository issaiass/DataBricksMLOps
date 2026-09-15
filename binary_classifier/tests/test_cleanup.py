from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from src.n00_shared.cleanup import (
    ALL_KINDS,
    DEFAULT_KINDS,
    assert_catalog_drop_allowed,
    planned_actions,
    resolve_delete_kinds,
)
from src.n00_shared.runtime import Settings


def _settings() -> Settings:
    return Settings(
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
        env_manager="local",
        secret_scope="mlops-adult",
        train_run_as_sp="train-sp",
        job_run_as_sp="job-sp",
        approver_identities="human@example.com",
        experiment_path="/Shared/mlops/binary_classifier/[dev issaiass] uc-adult-xgb",
        evaluate_mode="dev",
        batch_alias="dev",
    )


def test_empty_delete_kinds_means_example_objects_not_catalog():
    assert resolve_delete_kinds("") == DEFAULT_KINDS
    assert resolve_delete_kinds("all") == DEFAULT_KINDS
    assert "catalogs" not in DEFAULT_KINDS
    assert resolve_delete_kinds("everything") == ALL_KINDS


def test_blank_selection_still_deletes_endpoint():
    assert resolve_delete_kinds(", ,") == ("endpoints",)


def test_subset_always_includes_endpoint():
    kinds = resolve_delete_kinds("tables,models")
    assert kinds == ("endpoints", "tables", "models")


def test_unknown_kind_refuses():
    with pytest.raises(ValueError, match="unknown delete_kinds"):
        resolve_delete_kinds("clusters")


def test_protected_catalog_refused():
    with pytest.raises(RuntimeError, match="protected catalog"):
        assert_catalog_drop_allowed("system")


def test_planned_actions_default_covers_example_objects():
    settings = _settings()
    actions = planned_actions(
        settings,
        resolve_delete_kinds(""),
        volume_name="mlops_files",
        functions=["ml_dev.adult_income.demo_fn"],
        views=["ml_dev.adult_income.demo_view"],
        extra_tables=["ml_dev.adult_income.leftover_table"],
        extra_volumes=["ml_dev.adult_income.extra_vol"],
        extra_models=["ml_dev.adult_income.other_model"],
    )
    kinds = [k for k, _ in actions]
    assert kinds[0] == "endpoints"
    assert ("endpoints", "adult-income-clf-dev") in actions
    assert ("functions", "ml_dev.adult_income.demo_fn") in actions
    assert ("views", "ml_dev.adult_income.demo_view") in actions
    assert ("tables", "ml_dev.adult_income.adult_raw") in actions
    assert ("tables", "ml_dev.adult_income.leftover_table") in actions
    assert ("volumes", "ml_dev.adult_income.mlops_files") in actions
    assert ("volumes", "ml_dev.adult_income.extra_vol") in actions
    assert ("models", "ml_dev.adult_income.adult_income_clf") in actions
    assert ("models", "ml_dev.adult_income.other_model") in actions
    assert (
        "experiments",
        "/Shared/mlops/binary_classifier/[dev issaiass] uc-adult-xgb",
    ) in actions
    assert ("schemas", "ml_dev.adult_income") in actions
    assert ("catalogs", "ml_dev") not in actions


def test_protected_schema_refused():
    from src.n00_shared.cleanup import assert_schema_drop_allowed

    with pytest.raises(RuntimeError, match="protected schema"):
        assert_schema_drop_allowed("ml_dev", "information_schema")
    with pytest.raises(RuntimeError, match="empty"):
        assert_schema_drop_allowed("ml_dev", "  ")


def test_planned_actions_everything_includes_catalog():
    settings = _settings()
    actions = planned_actions(
        settings,
        resolve_delete_kinds("everything"),
        volume_name="mlops_files",
    )
    assert ("catalogs", "ml_dev") in actions
    assert ("schemas", "ml_dev.adult_income") in actions


def test_example_volume_requires_name():
    from src.n00_shared.cleanup import example_volume

    with pytest.raises(RuntimeError, match="VOLUME_NAME"):
        example_volume(_settings(), "")


def test_planned_actions_prod_uses_dest_model():
    settings = _settings()
    settings.catalog = "ml_prod"
    settings.evaluate_mode = "prod_gate"
    settings.endpoint_name = "adult-income-clf-prod"
    actions = dict(
        planned_actions(
            settings, ("endpoints", "models"), volume_name="mlops_files"
        )
    )
    assert actions["endpoints"] == "adult-income-clf-prod"
    assert actions["models"] == "ml_prod.adult_income.adult_income_clf"
