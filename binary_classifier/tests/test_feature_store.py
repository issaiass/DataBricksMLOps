from types import SimpleNamespace

import pandas as pd
import pytest

from src.n00_shared.dataset import FEATURE_COLS
from src.n00_shared.feature_store import (
    engineer_adult_pandas,
    find_online_sync_pipeline_id,
    registered_version,
    rename_online_sync_pipeline,
)


def test_engineer_adult_pandas_refuses_missing_census_columns():
    with pytest.raises(RuntimeError, match="missing census columns"):
        engineer_adult_pandas(pd.DataFrame({"age": [1]}))


def test_registered_version_prefers_result_attr():
    assert registered_version(SimpleNamespace(registered_model_version="11")) == "11"
    assert registered_version(SimpleNamespace(version="3")) == "3"


def test_registered_version_falls_back_to_client_search():
    class _V:
        def __init__(self, version):
            self.version = version

    class _Client:
        def search_model_versions(self, _q):
            return [_V("1"), _V("4"), _V("2")]

    assert registered_version(SimpleNamespace(), client=_Client(), model_name="m") == "4"


def test_registered_version_refuses_when_absent():
    with pytest.raises(RuntimeError, match="did not return"):
        registered_version(SimpleNamespace())


def test_online_feature_fq_uses_online_catalog(settings):
    assert settings.online_feature_fq == "ml_dev.adult_income.adult_features_online"


def test_feature_cols_do_not_include_label():
    from src.n00_shared.dataset import LABEL_COL, MODEL_FEATURE_COLS

    assert LABEL_COL not in FEATURE_COLS
    assert LABEL_COL not in MODEL_FEATURE_COLS


class _Pipe:
    def __init__(self, pipeline_id, name, spec=None):
        self.pipeline_id = pipeline_id
        self.name = name
        self.spec = spec or SimpleNamespace(name=name)


class _Pipelines:
    def __init__(self, items, *, get_obj=None, update_error=None):
        self.items = items
        self.get_obj = get_obj
        self.update_error = update_error
        self.updated = []

    def list_pipelines(self):
        return list(self.items)

    def get(self, pipeline_id):
        return self.get_obj

    def update(self, **kwargs):
        if self.update_error:
            self.updated.append(kwargs)
            raise self.update_error
        self.updated.append(kwargs)


class _Ws:
    def __init__(self, pipelines):
        self.pipelines = pipelines


def test_find_online_sync_pipeline_prefers_job_style_name(settings):
    desired = settings.online_sync_pipeline_name
    client = _Ws(
        _Pipelines(
            [
                _Pipe("old", "ml_dev.adult_income.adult_features_online sbqSDo"),
                _Pipe("named", desired),
            ]
        )
    )
    assert find_online_sync_pipeline_id(client, settings, desired) == "named"


def test_find_online_sync_pipeline_matches_databricks_default_name(settings):
    client = _Ws(
        _Pipelines([_Pipe("p1", "ml_dev.adult_income.adult_features_online sbqSDo")])
    )
    assert (
        find_online_sync_pipeline_id(client, settings, settings.online_sync_pipeline_name)
        == "p1"
    )


def test_find_online_sync_pipeline_matches_jobs_ui_synced_table_label(settings):
    client = _Ws(
        _Pipelines(
            [
                _Pipe(
                    "p1",
                    "Synced table: ml_dev.adult_income.adult_features_online sbqSDo",
                )
            ]
        )
    )
    assert (
        find_online_sync_pipeline_id(client, settings, settings.online_sync_pipeline_name)
        == "p1"
    )


def test_rename_online_sync_pipeline_updates_default_name(settings):
    current = _Pipe(
        "p1", "Synced table: ml_dev.adult_income.adult_features_online sbqSDo"
    )
    pipes = _Pipelines([current], get_obj=current)
    got = rename_online_sync_pipeline(settings, pipeline_id="p1", client=_Ws(pipes))
    assert got == "p1"
    assert pipes.updated[0]["name"] == "[dev issaiass] adult-adult_income_clf-features-pipeline"


def test_rename_online_sync_pipeline_skips_when_name_empty(settings):
    settings.online_sync_pipeline_name = ""
    pipes = _Pipelines([])
    assert rename_online_sync_pipeline(settings, pipeline_id="p1", client=_Ws(pipes)) == ""
    assert pipes.updated == []


def test_rename_online_sync_pipeline_warns_when_databricks_blocks_sync_table(settings):
    current = _Pipe(
        "p1", "Synced table: ml_dev.adult_income.adult_features_online sbqSDo"
    )
    pipes = _Pipelines(
        [current],
        get_obj=current,
        update_error=RuntimeError(
            "DatabaseSyncTable Pipeline p1: only updates to sinks, name, tags, "
            "notifications, budgetPolicyId and dbrVersion are allowed."
        ),
    )
    got = rename_online_sync_pipeline(settings, pipeline_id="p1", client=_Ws(pipes))
    assert got == "p1"
