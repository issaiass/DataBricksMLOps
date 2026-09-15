"""Bundle-baked configuration. Catalog/schema/model/mode come from task env, not operator params.

Adult tables/models are named via bundle variables (never hardcoded in these helpers).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

MLFLOW_RUN_NAME_PREFIX = "uci-adult-xgb"


_KEYS = (
    "DEV_CATALOG",
    "PROD_CATALOG",
    "CATALOG",
    "SCHEMA",
    "MODEL_NAME",
    "SOURCE_MODEL_NAME",
    "DEST_MODEL_NAME",
    "RAW_TABLE",
    "FEATURE_TABLE",
    "LABEL_TABLE",
    "PREDICTIONS_TABLE",
    "INFERENCE_TABLE",
    "EDA_PROFILE_TABLE",
    "XAI_EXPLANATION_TABLE",
    "MONITOR_STATUS_TABLE",
    "LABEL_COL",
    "ID_COL",
    "EVENT_TIME_COL",
    "COMPARE_METRIC",
    "COMPARE_HIGHER_IS_BETTER",
    "COMPARE_MARGIN",
    "MIN_EVAL_ROWS",
    "MIN_TABLE_ROWS",
    "MAX_NULL_RATE",
    "LABEL_WINDOW",
    "EVAL_MIN_ROC_AUC",
    "TRAIN_SEED",
    "ALLOW_CHAMPION_FALLBACK",
    "ALLOW_CANARY",
    "CANARY_PERCENT",
    "CANARY_WARMUP_S",
    "ENDPOINT_NAME",
    "ENV_MANAGER",
    "SECRET_SCOPE",
    "TRAIN_RUN_AS_SP",
    "JOB_RUN_AS_SP",
    "APPROVER_IDENTITIES",
    "EXPERIMENT_PATH",
    "EVALUATE_MODE",
    "BATCH_ALIAS",
    "PREDICT_MODE",
)


def _spark():
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except Exception:
        return None


def _widget(name: str) -> Optional[str]:
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is None or "dbutils" not in ip.user_ns:
            return None
        dbutils = ip.user_ns["dbutils"]
        try:
            val = dbutils.widgets.get(name)
        except Exception:
            try:
                val = dbutils.widgets.get(name.lower())
            except Exception:
                return None
        if val is None or str(val).strip() == "":
            return None
        return str(val)
    except Exception:
        return None


def baked(name: str, default: Optional[str] = None) -> str:
    """Prefer process env, then deploy-time notebook base_parameters. Widgets are last resort."""
    key = name.upper()
    val = os.environ.get(key)
    if val is not None and str(val).strip() != "":
        return str(val)
    spark = _spark()
    if spark is not None:
        for conf_key in (f"spark.mlops.{key.lower()}", f"spark.mlops.{name}"):
            try:
                conf_val = spark.conf.get(conf_key, None)
            except Exception:
                conf_val = None
            if conf_val is not None and str(conf_val).strip() != "":
                return str(conf_val)
    widget = _widget(key) or _widget(name)
    if widget is not None:
        return widget
    if default is not None:
        return default
    raise RuntimeError(f"{key} must be baked by the Databricks job (task env / base_parameters)")


def operator_param(name: str, default: str = "") -> str:
    """Only source_model_version and promotion_reason may come from job/widget params."""
    env = os.environ.get(name.upper())
    if env is not None and str(env).strip() != "":
        return str(env)
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None and "dbutils" in ip.user_ns:
            dbutils = ip.user_ns["dbutils"]
            try:
                dbutils.widgets.text(name, default)
                return str(dbutils.widgets.get(name) or default)
            except Exception:
                return default
    except Exception:
        return default
    return default


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fq(catalog: str, schema: str, table: str) -> str:
    return f"{catalog}.{schema}.{table}"


@dataclass
class Settings:
    dev_catalog: str
    prod_catalog: str
    catalog: str
    schema: str
    model_name: str
    source_model_name: str
    dest_model_name: str
    raw_table: str
    feature_table: str
    label_table: str
    predictions_table: str
    inference_table: str
    eda_profile_table: str
    xai_explanation_table: str
    monitor_status_table: str
    label_col: str
    id_col: str
    event_time_col: str
    compare_metric: str
    compare_higher_is_better: bool
    compare_margin: float
    min_eval_rows: int
    min_table_rows: int
    max_null_rate: float
    label_window: str
    eval_min_roc_auc: float
    train_seed: int
    allow_champion_fallback: bool
    allow_canary: bool
    canary_percent: int
    canary_warmup_s: int
    endpoint_name: str
    env_manager: str
    secret_scope: str
    train_run_as_sp: str
    job_run_as_sp: str
    approver_identities: str
    experiment_path: str
    evaluate_mode: str
    batch_alias: str
    predict_mode: str = "batch"

    @property
    def raw_fq(self) -> str:
        return fq(self.catalog, self.schema, self.raw_table)

    @property
    def feature_fq(self) -> str:
        return fq(self.catalog, self.schema, self.feature_table)

    @property
    def label_fq(self) -> str:
        return fq(self.catalog, self.schema, self.label_table)

    @property
    def predictions_fq(self) -> str:
        return fq(self.catalog, self.schema, self.predictions_table)

    @property
    def inference_fq(self) -> str:
        return fq(self.catalog, self.schema, self.inference_table)

    @property
    def eda_fq(self) -> str:
        return fq(self.catalog, self.schema, self.eda_profile_table)

    @property
    def xai_fq(self) -> str:
        return fq(self.catalog, self.schema, self.xai_explanation_table)

    @property
    def monitor_fq(self) -> str:
        return fq(self.catalog, self.schema, self.monitor_status_table)

    @property
    def active_model(self) -> str:
        if self.evaluate_mode == "prod_gate":
            return self.dest_model_name
        return self.source_model_name


def load_settings() -> Settings:
    return Settings(
        dev_catalog=baked("DEV_CATALOG"),
        prod_catalog=baked("PROD_CATALOG"),
        catalog=baked("CATALOG"),
        schema=baked("SCHEMA"),
        model_name=baked("MODEL_NAME"),
        source_model_name=baked("SOURCE_MODEL_NAME"),
        dest_model_name=baked("DEST_MODEL_NAME"),
        raw_table=baked("RAW_TABLE"),
        feature_table=baked("FEATURE_TABLE"),
        label_table=baked("LABEL_TABLE"),
        predictions_table=baked("PREDICTIONS_TABLE"),
        inference_table=baked("INFERENCE_TABLE"),
        eda_profile_table=baked("EDA_PROFILE_TABLE"),
        xai_explanation_table=baked("XAI_EXPLANATION_TABLE"),
        monitor_status_table=baked("MONITOR_STATUS_TABLE"),
        label_col=baked("LABEL_COL"),
        id_col=baked("ID_COL"),
        event_time_col=baked("EVENT_TIME_COL", ""),
        compare_metric=baked("COMPARE_METRIC"),
        compare_higher_is_better=as_bool(baked("COMPARE_HIGHER_IS_BETTER")),
        compare_margin=float(baked("COMPARE_MARGIN")),
        min_eval_rows=int(baked("MIN_EVAL_ROWS")),
        min_table_rows=int(baked("MIN_TABLE_ROWS")),
        max_null_rate=float(baked("MAX_NULL_RATE")),
        label_window=baked("LABEL_WINDOW"),
        eval_min_roc_auc=float(baked("EVAL_MIN_ROC_AUC")),
        train_seed=int(baked("TRAIN_SEED")),
        allow_champion_fallback=as_bool(baked("ALLOW_CHAMPION_FALLBACK")),
        allow_canary=as_bool(baked("ALLOW_CANARY")),
        canary_percent=int(baked("CANARY_PERCENT")),
        canary_warmup_s=int(baked("CANARY_WARMUP_S")),
        endpoint_name=baked("ENDPOINT_NAME"),
        env_manager=baked("ENV_MANAGER", "local"),
        secret_scope=baked("SECRET_SCOPE"),
        train_run_as_sp=baked("TRAIN_RUN_AS_SP"),
        job_run_as_sp=baked("JOB_RUN_AS_SP"),
        approver_identities=baked("APPROVER_IDENTITIES"),
        experiment_path=baked("EXPERIMENT_PATH"),
        evaluate_mode=baked("EVALUATE_MODE"),
        batch_alias=baked("BATCH_ALIAS"),
        predict_mode=(operator_param("predict_mode", "") or baked("PREDICT_MODE", "batch")).strip().lower(),
    )


def assert_dev_ml_allowed(settings: Settings) -> None:
    if str(settings.catalog) == str(settings.prod_catalog):
        raise RuntimeError("train/eda/xai/compare must not use prod_catalog")
    if str(settings.dev_catalog) == str(settings.prod_catalog):
        raise RuntimeError("dev_catalog and prod_catalog must be distinct")


def mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    return MlflowClient(registry_uri="databricks-uc")


def timestamped_run_name(
    base: str = MLFLOW_RUN_NAME_PREFIX,
    when: datetime | None = None,
) -> str:
    """Unique MLflow run name: ``uci-adult-xgb-140926-232856`` (DDMMYY-HHMMSS)."""
    stamp = (when or datetime.now()).strftime("%d%m%y-%H%M%S")
    return f"{base}-{stamp}"


def configure_mlflow(settings: Settings) -> None:
    """Bind Tracking to the target experiment: ``[<target> <user>] uc-adult-xgb``."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(settings.experiment_path)


def set_task_value(key: str, value: Any) -> None:
    os.environ[f"MLOPS_TASK_{key}"] = str(value)
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None and "dbutils" in ip.user_ns:
            ip.user_ns["dbutils"].jobs.taskValues.set(key=key, value=str(value))
    except Exception:
        pass


def get_task_value(task_key: str, key: str, default: Optional[str] = None) -> str:
    env = os.environ.get(f"MLOPS_TASK_{key}")
    if env:
        return env
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None and "dbutils" in ip.user_ns:
            val = ip.user_ns["dbutils"].jobs.taskValues.get(
                taskKey=task_key, key=key, debugValue=default
            )
            if val is not None:
                return str(val)
    except Exception:
        pass
    if default is not None:
        return default
    raise RuntimeError(f"missing task value {task_key}.{key}")


def current_task_key() -> str:
    for name in ("DB_JOB_TASK_KEY", "TASK_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None and "dbutils" in ip.user_ns:
            ctx = ip.user_ns["dbutils"].notebook.entry_point.getDbutils().notebook().getContext()
            for attr in ("taskKey", "task_key"):
                getter = getattr(ctx, attr, None)
                if getter:
                    got = getter()
                    if hasattr(got, "get"):
                        return str(got.get() or "")
                    return str(got or "")
    except Exception:
        return ""
    return ""


def job_run_id() -> str:
    return os.environ.get("DB_JOB_ID") or os.environ.get("DATABRICKS_JOB_RUN_ID") or ""
