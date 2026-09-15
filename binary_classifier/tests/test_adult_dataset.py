from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.n00_shared.dataset import (
    ENGINEERED_FEATURE_COLS,
    FEATURE_COLS,
    LABEL_COL,
    MODEL_FEATURE_COLS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    SPLIT_COL,
    encode_income_label,
    normalize_adult_pandas,
)
from src.n00_shared.feature_store import engineer_adult_pandas


def test_encode_gt50k_is_positive_one():
    s = pd.Series([">50K", "<=50K", ">50K.", "<=50K.", " >50K "])
    out = encode_income_label(s)
    assert list(out) == [1, 0, 1, 0, 1]


def test_encode_rejects_unknown_class():
    with pytest.raises(ValueError, match="unexpected Adult income"):
        encode_income_label(pd.Series([">50K", "unknown"]))


def test_normalize_renames_uci_hyphens_and_drops_raw_income():
    pdf = pd.DataFrame(
        {
            "age": [39, 25],
            "workclass": [" Private", "?"],
            "fnlwgt": [77516, 226802],
            "education": ["Bachelors", "11th"],
            "education-num": [13, 7],
            "marital-status": ["Never-married", "Never-married"],
            "occupation": ["Adm-clerical", "Machine-op-inspct"],
            "relationship": ["Not-in-family", "Own-child"],
            "race": ["White", "Black"],
            "sex": ["Male", "Male"],
            "capital-gain": [2174, 0],
            "capital-loss": [0, 0],
            "hours-per-week": [40, 40],
            "native-country": ["United-States", "United-States"],
            "income": [">50K", "<=50K"],
        }
    )
    out = normalize_adult_pandas(pdf, split="train")
    assert LABEL_COL in out.columns
    assert "income" not in out.columns
    assert list(out[LABEL_COL]) == [1, 0]
    assert out.loc[0, "education_num"] == 13
    assert out.loc[0, "marital_status"] == "Never-married"
    assert pd.isna(out.loc[1, "workclass"])
    assert out[SPLIT_COL].unique().tolist() == ["train"]
    for col in FEATURE_COLS:
        assert col in out.columns


def test_bundled_adult_files_parse():
    from src.n00_shared.dataset import load_adult_pandas

    data_dir = ROOT / "data"
    if not (data_dir / "adult.data").is_file():
        pytest.skip("vendored Adult files not present")
    out = load_adult_pandas()
    assert len(out) >= 200
    assert set(out[SPLIT_COL].unique()) == {"train", "holdout"}
    assert out[LABEL_COL].isin([0, 1]).all()


def test_engineer_adult_pandas_adds_feature_store_columns():
    import numpy as np

    pdf = pd.DataFrame(
        {
            "age": [39],
            "workclass": ["Private"],
            "fnlwgt": [float(np.e - 1)],
            "education": ["Bachelors"],
            "education_num": [4],
            "marital_status": ["Never-married"],
            "occupation": ["Adm-clerical"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "capital_gain": [100],
            "capital_loss": [30],
            "hours_per_week": [45],
            "native_country": ["United-States"],
        }
    )
    out = engineer_adult_pandas(pdf)
    assert out.loc[0, "capital_net"] == 70
    assert int(out.loc[0, "hours_over_40"]) == 1
    assert abs(out.loc[0, "log_fnlwgt"] - 1.0) < 1e-6
    assert out.loc[0, "education_num_sq"] == 16
    for col in ENGINEERED_FEATURE_COLS:
        assert col in out.columns
    assert set(ENGINEERED_FEATURE_COLS).issubset(MODEL_FEATURE_COLS)


def test_positive_negative_class_literals():
    assert POSITIVE_CLASS == ">50K"
    assert NEGATIVE_CLASS == "<=50K"


def test_normalize_accepts_class_column_and_refuses_missing_income():
    pdf = pd.DataFrame(
        {
            "age": [39],
            "workclass": ["Private"],
            "fnlwgt": [1],
            "education": ["Bachelors"],
            "education_num": [13],
            "marital_status": ["Never-married"],
            "occupation": ["Adm-clerical"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "capital_gain": [0],
            "capital_loss": [0],
            "hours_per_week": [40],
            "native_country": ["United-States"],
            "class": [">50K"],
        }
    )
    out = normalize_adult_pandas(pdf)
    assert list(out[LABEL_COL]) == [1]
    with pytest.raises(ValueError, match="missing the income"):
        normalize_adult_pandas(pd.DataFrame({"age": [1]}))


def test_feature_contract_types():
    from src.n00_shared.dataset import ADULT_MODEL_NUMERIC, FEATURE_CONTRACT, MODEL_FEATURE_COLS

    assert set(FEATURE_CONTRACT) == set(MODEL_FEATURE_COLS)
    for col in ADULT_MODEL_NUMERIC:
        assert FEATURE_CONTRACT[col] == "double"
