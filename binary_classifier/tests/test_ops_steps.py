from unittest.mock import MagicMock

import pytest

from src.n00_shared.runtime import Settings
from tests.conftest import FakeMlflowClient, make_settings


def test_batch_alias_must_be_dev_or_prod(settings: Settings):
    from src.n04_ops import n01_batch as batch

    with pytest.raises(RuntimeError, match="batch_alias"):
        batch.resolve_model(make_settings(batch_alias="challenger"))


def test_batch_resolve_model_missing_alias_returns_error(monkeypatch, settings: Settings):
    from src.n04_ops import n01_batch as batch

    class _Client:
        def get_model_version_by_alias(self, *_a, **_k):
            raise RuntimeError("no alias")

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **_k: _Client())
    name, alias, version, err = batch.resolve_model(settings)
    assert version is None
    assert err is not None
    assert alias == "dev"
    assert name == settings.source_model_name


def test_batch_run_noop_when_alias_missing(monkeypatch, settings: Settings):
    from src.n04_ops import n01_batch as batch

    monkeypatch.setattr(batch, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(batch, "_spark", lambda: MagicMock())
    monkeypatch.setattr(
        batch,
        "resolve_model",
        lambda _s: (_s.source_model_name, "dev", None, RuntimeError("missing")),
    )
    load = MagicMock()
    monkeypatch.setattr(batch, "load_raw_and_features", load)
    batch.run(settings)
    load.assert_not_called()


def test_batch_load_raw_missing_table(settings: Settings):
    from src.n04_ops import n01_batch as batch

    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    with pytest.raises(RuntimeError, match="missing table"):
        batch.load_raw_and_features(spark, settings)


def test_monitor_writes_gate_ok_false_when_features_missing(monkeypatch, settings: Settings):
    from src.n04_ops import n02_monitor as monitor

    spark = MagicMock()
    spark.catalog.tableExists.return_value = False
    writer = MagicMock()
    status_df = MagicMock()
    status_df.write.format.return_value.mode.return_value.option.return_value.saveAsTable = writer
    spark.createDataFrame.return_value = status_df
    monkeypatch.setattr(monitor, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(monitor, "_spark", lambda: spark)
    monkeypatch.setattr(monitor, "job_run_id", lambda: "job-1")
    monitor.run(settings)
    row = spark.createDataFrame.call_args[0][0][0]
    assert row["gate_ok"] is False
    assert "missing" in row["reason"].lower() or "feature" in row["reason"].lower()
    writer.assert_called_once()


def test_cleanup_sql_ident_and_qualify(settings: Settings):
    from src.n04_ops import n03_cleanup as cleanup

    assert cleanup._sql_ident("ml_dev.adult_income.adult_raw") == "`ml_dev`.`adult_income`.`adult_raw`"
    assert cleanup._qualify(settings, "leftover") == "ml_dev.adult_income.leftover"
    assert cleanup._qualify(settings, "ml_dev.adult_income.full") == "ml_dev.adult_income.full"


def test_cleanup_unsupported_kind(settings: Settings):
    from src.n04_ops import n03_cleanup as cleanup

    with pytest.raises(RuntimeError, match="unsupported cleanup kind"):
        cleanup._delete_kind(MagicMock(), settings, "clusters", "x")


def test_cleanup_collect_names_skips_temp_and_sql_errors(settings: Settings):
    from src.n04_ops import n03_cleanup as cleanup

    spark = MagicMock()
    spark.sql.side_effect = RuntimeError("no schema")
    assert cleanup._collect_names(spark, "SHOW TABLES", settings) == []

    class _Row:
        def __init__(self, mapping):
            self._m = mapping

        def asDict(self, recursive=True):
            return self._m

        def __bool__(self):
            return True

        def __getitem__(self, idx):
            return None

    spark.sql.side_effect = None
    spark.sql.return_value.collect.return_value = [
        _Row({"isTemporary": True, "name": "tmp"}),
        _Row({"tableName": "extra", "isTemporary": False}),
    ]
    assert cleanup._collect_names(spark, "SHOW TABLES", settings) == [
        "ml_dev.adult_income.extra"
    ]
