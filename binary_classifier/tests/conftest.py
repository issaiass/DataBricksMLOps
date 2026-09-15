from pathlib import Path
import sys
import types

import pytest

# Local pytest must not require Databricks extras. Stub missing optional imports
# so step modules can be imported without UC / MLflow / sklearn.
def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


try:
    import mlflow  # noqa: F401
except ImportError:
    mlflow = _ensure_module("mlflow")
    exceptions = _ensure_module("mlflow.exceptions")

    class MlflowException(Exception):
        pass

    exceptions.MlflowException = MlflowException
    mlflow.exceptions = exceptions
    tracking = _ensure_module("mlflow.tracking")

    class MlflowClient:
        def __init__(self, *args, **kwargs):
            pass

    tracking.MlflowClient = MlflowClient
    mlflow.tracking = tracking
    mlflow.set_registry_uri = lambda *args, **kwargs: None
    mlflow.register_model = lambda *args, **kwargs: types.SimpleNamespace(version="1")
    mlflow.set_experiment = lambda *args, **kwargs: None
    mlflow.start_run = lambda *args, **kwargs: types.SimpleNamespace(
        info=types.SimpleNamespace(run_id="run")
    )
    sklearn_flavor = _ensure_module("mlflow.sklearn")
    xgb_flavor = _ensure_module("mlflow.xgboost")
    sklearn_flavor.autolog = lambda *args, **kwargs: None
    xgb_flavor.autolog = lambda *args, **kwargs: None
    mlflow.sklearn = sklearn_flavor
    mlflow.xgboost = xgb_flavor
    models = _ensure_module("mlflow.models")
    models.infer_signature = lambda *args, **kwargs: None
    mlflow.models = models
    deployments = _ensure_module("mlflow.deployments")
    deployments.get_deploy_client = lambda *args, **kwargs: None
    mlflow.deployments = deployments

try:
    import sklearn  # noqa: F401
except ImportError:
    sklearn = _ensure_module("sklearn")
    metrics = _ensure_module("sklearn.metrics")
    metrics.accuracy_score = lambda *args, **kwargs: 0.0
    metrics.f1_score = lambda *args, **kwargs: 0.0
    metrics.roc_auc_score = lambda *args, **kwargs: 0.0
    sklearn.metrics = metrics
    compose = _ensure_module("sklearn.compose")
    compose.ColumnTransformer = type("ColumnTransformer", (), {"__init__": lambda *a, **k: None})
    sklearn.compose = compose
    impute = _ensure_module("sklearn.impute")
    impute.SimpleImputer = type("SimpleImputer", (), {"__init__": lambda *a, **k: None})
    sklearn.impute = impute
    pipeline = _ensure_module("sklearn.pipeline")
    pipeline.Pipeline = type("Pipeline", (), {"__init__": lambda *a, **k: None})
    sklearn.pipeline = pipeline
    preprocessing = _ensure_module("sklearn.preprocessing")
    preprocessing.OneHotEncoder = type("OneHotEncoder", (), {"__init__": lambda *a, **k: None})
    sklearn.preprocessing = preprocessing
    inspection = _ensure_module("sklearn.inspection")
    inspection.permutation_importance = lambda *args, **kwargs: None
    sklearn.inspection = inspection

try:
    import pyspark  # noqa: F401
except ImportError:
    pyspark = _ensure_module("pyspark")
    sql = _ensure_module("pyspark.sql")
    functions = _ensure_module("pyspark.sql.functions")
    window = _ensure_module("pyspark.sql.window")
    window.Window = type("Window", (), {"orderBy": staticmethod(lambda *a, **k: None)})
    sql.functions = functions
    sql.window = window
    pyspark.sql = sql
    functions.col = lambda *args, **kwargs: None
    functions.when = lambda *args, **kwargs: None
    functions.lit = lambda *args, **kwargs: None
    functions.hash = lambda *args, **kwargs: None
    functions.mean = lambda *args, **kwargs: None
    functions.min = lambda *args, **kwargs: None
    functions.max = lambda *args, **kwargs: None
    functions.current_timestamp = lambda *args, **kwargs: None
    functions.row_number = lambda *args, **kwargs: None

    class SparkSession:
        class builder:
            @staticmethod
            def getOrCreate():
                return None

    sql.SparkSession = SparkSession

try:
    import xgboost  # noqa: F401
except ImportError:
    xgboost = _ensure_module("xgboost")

    class XGBClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def set_params(self, **kwargs):
            return self

    xgboost.XGBClassifier = XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.n00_shared.runtime import Settings


def make_settings(**kwargs) -> Settings:
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
        env_manager="local",
        secret_scope="mlops-adult",
        train_run_as_sp="train-sp",
        job_run_as_sp="job-sp",
        approver_identities="human@example.com",
        experiment_path="/Shared/mlops/binary_classifier/[dev issaiass] uc-adult-xgb",
        evaluate_mode="dev",
        batch_alias="dev",
        predict_mode="batch",
    )
    base.update(kwargs)
    return Settings(**base)


@pytest.fixture
def settings():
    return make_settings()


class FakeModelVersion:
    def __init__(self, version, tags=None):
        self.version = str(version)
        self.tags = dict(tags or {})


class FakeMlflowClient:
    """In-memory UC registry stand-in. No network, no writes."""

    def __init__(self):
        self.aliases: dict[tuple[str, str], str] = {}
        self.tags: dict[tuple[str, str], dict[str, str]] = {}
        self.alias_calls: list[tuple[str, str, str]] = []
        self.tag_calls: list[tuple[str, str, str, str]] = []

    def get_model_version(self, name: str, version: str):
        return FakeModelVersion(version, self.tags.get((name, str(version)), {}))

    def get_model_version_by_alias(self, name: str, alias: str):
        key = (name, alias)
        if key not in self.aliases:
            raise RuntimeError(f"alias {alias} missing on {name}")
        version = self.aliases[key]
        return FakeModelVersion(version, self.tags.get((name, str(version)), {}))

    def set_registered_model_alias(self, name: str, alias: str, version: str):
        version = str(version)
        self.alias_calls.append((name, alias, version))
        self.aliases[(name, alias)] = version

    def set_model_version_tag(self, name: str, version: str, key: str, value: str):
        version = str(version)
        self.tag_calls.append((name, version, key, str(value)))
        self.tags.setdefault((name, version), {})[key] = str(value)


@pytest.fixture
def fake_client():
    return FakeMlflowClient()
