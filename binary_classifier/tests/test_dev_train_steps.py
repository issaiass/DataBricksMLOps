from unittest.mock import MagicMock
import sys
import types

import pytest

from src.n00_shared.runtime import Settings
from tests.conftest import FakeMlflowClient, make_settings


def _patch_common(monkeypatch, mod, client):
    monkeypatch.setattr(mod, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(mod, "mlflow_client", lambda: client)
    monkeypatch.setattr(mod, "set_task_value", lambda *_a, **_k: None)


def test_seed_refuses_wrong_label(monkeypatch, settings: Settings):
    from src.n01_dev_train import n01_seed as seed

    monkeypatch.setattr(seed, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(seed, "_spark", lambda: MagicMock())
    monkeypatch.setattr(seed, "load_adult_pandas", lambda: MagicMock())
    with pytest.raises(RuntimeError, match="expected label"):
        seed.run(make_settings(label_col="income"))


def test_data_checks_missing_table(monkeypatch, settings: Settings):
    from src.n01_dev_train import n02_data_checks as checks

    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    monkeypatch.setattr(checks, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(checks, "_spark", lambda: spark)
    with pytest.raises(RuntimeError, match="missing table"):
        checks.run(settings)


def test_eda_refuses_prod_catalog():
    from src.n01_dev_train import n03_eda as eda

    with pytest.raises(RuntimeError, match="must not use prod_catalog"):
        eda.run(make_settings(catalog="ml_prod"))


def test_features_missing_raw(monkeypatch, settings: Settings):
    from src.n01_dev_train import n04_features as features

    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    monkeypatch.setattr(features, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(features, "_spark", lambda: spark)
    with pytest.raises(RuntimeError, match="missing raw table"):
        features.run(settings)


def test_train_refuses_prod_catalog():
    from src.n01_dev_train import n05_train as train

    with pytest.raises(RuntimeError, match="must not use prod_catalog"):
        train.run(make_settings(catalog="ml_prod"))


def test_train_pipeline_has_xgb_step():
    import sklearn.pipeline as sp

    if sp.Pipeline.__module__ != "sklearn.pipeline":
        pytest.skip("real scikit-learn is not installed")
    from src.n01_dev_train import n05_train as train

    pipe = train._pipeline(42)
    assert "clf" in pipe.named_steps


def test_evaluate_model_uri_dev_vs_prod_gate(monkeypatch, settings: Settings):
    from src.n01_dev_train import n06_evaluate as evaluate

    monkeypatch.setattr(
        evaluate,
        "get_task_value",
        lambda task, key, default=None: {"copy_register": "21", "train": "4"}[task],
    )
    name, version, uri = evaluate._model_uri_and_name(settings)
    assert name == settings.source_model_name
    assert version == "4"
    assert uri.endswith("/4")
    prod = make_settings(evaluate_mode="prod_gate")
    name, version, uri = evaluate._model_uri_and_name(prod)
    assert name == prod.dest_model_name
    assert version == "21"


def test_evaluate_unsupported_mode_and_prod_catalog(monkeypatch, settings: Settings):
    pyspark = types.ModuleType("pyspark")
    sql = types.ModuleType("pyspark.sql")
    functions = types.ModuleType("pyspark.sql.functions")
    pyspark.sql = sql
    sql.functions = functions
    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)
    from src.n01_dev_train import n06_evaluate as evaluate

    with pytest.raises(RuntimeError, match="unsupported evaluate mode"):
        evaluate.run(make_settings(evaluate_mode="compare"))
    with pytest.raises(RuntimeError, match="must not use prod_catalog"):
        evaluate.run(make_settings(catalog="ml_prod", evaluate_mode="dev"))


def test_compare_tags_only_never_sets_champion(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n01_dev_train import n07_compare as compare

    fake_client.tags[(settings.source_model_name, "5")] = {"roc_auc": "0.9"}
    _patch_common(monkeypatch, compare, fake_client)
    monkeypatch.setattr(compare, "assert_dev_ml_allowed", lambda _s: None)
    monkeypatch.setattr(compare, "get_task_value", lambda *_a, **_k: "5")
    monkeypatch.setattr(compare, "MlflowException", RuntimeError)
    compare.run(settings)
    assert ("compare_result", "go") in [(k, v) for _n, _ver, k, v in fake_client.tag_calls]
    assert all(alias != "champion" for _n, alias, _v in fake_client.alias_calls)
    assert fake_client.alias_calls == []


def test_compare_no_go_still_succeeds(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n01_dev_train import n07_compare as compare

    fake_client.tags[(settings.source_model_name, "5")] = {"roc_auc": "0.5"}
    fake_client.aliases[(settings.source_model_name, "champion")] = "1"
    fake_client.tags[(settings.source_model_name, "1")] = {"roc_auc": "0.9"}
    _patch_common(monkeypatch, compare, fake_client)
    monkeypatch.setattr(compare, "assert_dev_ml_allowed", lambda _s: None)
    monkeypatch.setattr(compare, "get_task_value", lambda *_a, **_k: "5")
    monkeypatch.setattr(compare, "MlflowException", RuntimeError)
    compare.run(settings)
    assert any(k == "compare_result" and v == "no-go" for _n, _ver, k, v in fake_client.tag_calls)
    assert fake_client.alias_calls == []


def test_xai_refuses_prod_catalog():
    from src.n01_dev_train import n08_xai as xai

    with pytest.raises(RuntimeError, match="must not use prod_catalog"):
        xai.run(make_settings(catalog="ml_prod"))


def test_promotion_ready_go_sets_champion(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n01_dev_train import n09_promotion_ready as ready

    fake_client.tags[(settings.source_model_name, "6")] = {"compare_result": "go"}
    _patch_common(monkeypatch, ready, fake_client)
    monkeypatch.setattr(ready, "assert_dev_ml_allowed", lambda _s: None)
    monkeypatch.setattr(ready, "get_task_value", lambda *_a, **_k: "6")
    monkeypatch.setattr(ready, "job_run_id", lambda: "train-run-99")
    ready.run(settings)
    assert (settings.source_model_name, "champion", "6") in fake_client.alias_calls
    tags = {(k, v) for _n, _ver, k, v in fake_client.tag_calls}
    assert ("ready", "true") in tags
    assert ("go_version", "6") in tags


def test_promotion_ready_no_go_does_not_move_champion(
    monkeypatch, settings: Settings, fake_client: FakeMlflowClient
):
    from src.n01_dev_train import n09_promotion_ready as ready

    fake_client.tags[(settings.source_model_name, "6")] = {"compare_result": "no-go"}
    fake_client.aliases[(settings.source_model_name, "champion")] = "1"
    _patch_common(monkeypatch, ready, fake_client)
    monkeypatch.setattr(ready, "assert_dev_ml_allowed", lambda _s: None)
    monkeypatch.setattr(ready, "get_task_value", lambda *_a, **_k: "6")
    ready.run(settings)
    assert fake_client.alias_calls == []
    assert fake_client.aliases[(settings.source_model_name, "champion")] == "1"
    assert ("ready", "false") in {(k, v) for _n, _ver, k, v in fake_client.tag_calls}


def test_smoke_refuses_prod_catalog():
    from src.n01_dev_train import n10_smoke as smoke

    with pytest.raises(RuntimeError, match="must not use prod_catalog"):
        smoke.run(make_settings(catalog="ml_prod"))
