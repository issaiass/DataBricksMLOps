from types import SimpleNamespace

import pandas as pd
import pytest

from src.n00_shared.dataset import FEATURE_COLS
from src.n00_shared.feature_store import engineer_adult_pandas, registered_version


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


def test_feature_cols_do_not_include_label():
    from src.n00_shared.dataset import LABEL_COL, MODEL_FEATURE_COLS

    assert LABEL_COL not in FEATURE_COLS
    assert LABEL_COL not in MODEL_FEATURE_COLS
