"""UCI Adult Census Income — binary classification.

Source: https://archive.ics.uci.edu/dataset/2/adult
Citation: Becker, B. & Kohavi, R. (1996). Adult. UCI ML Repository. DOI: 10.24432/C5XW20

Target (two classes):
  >50K  → 1  (positive)
  <=50K → 0  (negative)

UCI adult.test labels include a trailing period (``>50K.``); that is stripped
before encoding. Missing categoricals are stored as ``?`` in the raw files.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from zipfile import ZipFile

# Official UCI column order (adult.names / adult.data).
UCI_RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

ADULT_NUMERIC = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
ADULT_CATEGORICAL = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
FEATURE_COLS = ADULT_NUMERIC + ADULT_CATEGORICAL
ENGINEERED_FEATURE_COLS = [
    "capital_net",
    "hours_over_40",
    "log_fnlwgt",
    "education_num_sq",
]
MODEL_FEATURE_COLS = FEATURE_COLS + ENGINEERED_FEATURE_COLS
ADULT_MODEL_NUMERIC = ADULT_NUMERIC + ENGINEERED_FEATURE_COLS
FEATURE_CONTRACT = {
    c: ("double" if c in ADULT_MODEL_NUMERIC else "string") for c in MODEL_FEATURE_COLS
}

LABEL_COL = "income_gt_50k"
ID_COL = "row_id"
SPLIT_COL = "split"
POSITIVE_CLASS = ">50K"
NEGATIVE_CLASS = "<=50K"
TARGET_DEFINITION = (
    "Predict whether annual income exceeds $50K/yr (UCI Adult / Census Income). "
    "income_gt_50k=1 if class is >50K else 0 if class is <=50K."
)

UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/2/adult"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
# Serverless often cannot resolve archive.ics.uci.edu; GitHub + vendored files do.
GITHUB_DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-train.csv"
GITHUB_TEST_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-test.csv"


def _snake(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def encode_income_label(series):
    """Map UCI income strings to {0, 1}. ``>50K`` / ``>50K.`` → 1; ``<=50K`` → 0."""
    cleaned = series.astype(str).str.strip().str.replace(".", "", regex=False)
    unknown = ~cleaned.isin([POSITIVE_CLASS, NEGATIVE_CLASS, POSITIVE_CLASS.lower(), NEGATIVE_CLASS.lower()])
    if bool(unknown.any()):
        bad = sorted(cleaned[unknown].unique().tolist())[:8]
        raise ValueError(f"unexpected Adult income labels {bad}")
    return cleaned.str.lower().eq(POSITIVE_CLASS.lower()).astype(int)


def normalize_adult_pandas(pdf, split: Optional[str] = None):
    import pandas as pd

    rename = {c: _snake(c) for c in pdf.columns}
    out = pdf.rename(columns=rename).copy()
    if "class" in out.columns and "income" not in out.columns:
        out["income"] = out["class"]
    if "education_num" not in out.columns and "educationnum" in out.columns:
        out["education_num"] = out["educationnum"]
    for col in ADULT_CATEGORICAL + ["income"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace({"nan": pd.NA, "None": pd.NA, "?": pd.NA, "": pd.NA})
    if "income" not in out.columns:
        raise ValueError("Adult frame is missing the income / class column")
    out[LABEL_COL] = encode_income_label(out["income"])
    for col in ADULT_NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if split is not None:
        out[SPLIT_COL] = split
    for col in ADULT_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].astype(object)
            out.loc[pd.isna(out[col]), col] = None
    keep = [c for c in FEATURE_COLS if c in out.columns]
    extra = [LABEL_COL]
    if SPLIT_COL in out.columns:
        extra = [SPLIT_COL, LABEL_COL]
    return out[keep + extra].reset_index(drop=True)


def _read_uci_csv(raw: bytes, *, skiprows: int = 0):
    import pandas as pd

    text = raw.decode("utf-8", errors="replace")
    first = text.lstrip().splitlines()[:1]
    if skiprows == 0 and first and first[0].startswith("|"):
        skiprows = 1
    return pd.read_csv(
        StringIO(text),
        header=None,
        names=UCI_RAW_COLUMNS,
        skipinitialspace=True,
        skiprows=skiprows,
        na_values=["?"],
    )


def _bundle_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _http_get(url: str) -> bytes:
    with urlopen(url, timeout=60) as resp:
        return resp.read()


def _from_uci_zip() -> tuple:
    blob = _http_get(UCI_ZIP_URL)
    with ZipFile(BytesIO(blob)) as zf:
        names = {name.split("/")[-1].lower(): name for name in zf.namelist()}
        if "adult.data" not in names or "adult.test" not in names:
            raise FileNotFoundError(f"adult.zip missing data files: {list(names)}")
        train = _read_uci_csv(zf.read(names["adult.data"]))
        test = _read_uci_csv(zf.read(names["adult.test"]), skiprows=1)
    return train, test


def _from_uci_flat_files() -> tuple:
    train = _read_uci_csv(_http_get(UCI_DATA_URL))
    test = _read_uci_csv(_http_get(UCI_TEST_URL), skiprows=1)
    return train, test


def _from_github_raw() -> tuple:
    train = _read_uci_csv(_http_get(GITHUB_DATA_URL))
    test = _read_uci_csv(_http_get(GITHUB_TEST_URL))
    return train, test


def _from_bundled_files() -> tuple:
    data_dir = _bundle_data_dir()
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"missing vendored Adult files under {data_dir}")
    train = _read_uci_csv(train_path.read_bytes())
    test = _read_uci_csv(test_path.read_bytes())
    return train, test


def load_adult_pandas():
    """Load UCI Adult train+test with official ``split`` (vendored files, then HTTP)."""
    import pandas as pd

    last_err: Optional[Exception] = None
    train = test = None
    for loader in (_from_bundled_files, _from_github_raw, _from_uci_zip, _from_uci_flat_files):
        try:
            train, test = loader()
            break
        except Exception as exc:
            last_err = exc
    if train is None or test is None:
        raise RuntimeError(
            f"failed to load UCI Adult (bundled data/, GitHub, or {UCI_DATASET_URL}): {last_err}"
        )
    train_n = normalize_adult_pandas(train, split="train")
    holdout = normalize_adult_pandas(test, split="holdout")
    out = pd.concat([train_n, holdout], ignore_index=True)
    out.insert(0, ID_COL, range(1, len(out) + 1))
    return out[ [ID_COL] + FEATURE_COLS + [SPLIT_COL, LABEL_COL] ]
