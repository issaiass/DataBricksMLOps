"""Rewrite job YAML: serverless notebook tasks use base_parameters, not spark_env_vars."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = """            base_parameters: &mlops_params
              BUNDLE_ROOT: ${workspace.file_path}
              DEV_CATALOG: ${var.dev_catalog}
              PROD_CATALOG: ${var.prod_catalog}
              CATALOG: ${var.catalog}
              SCHEMA: ${var.schema}
              MODEL_NAME: ${var.model_name}
              SOURCE_MODEL_NAME: ${var.source_model_name}
              DEST_MODEL_NAME: ${var.dest_model_name}
              EVALUATE_MODE: {evaluate_mode}
              BATCH_ALIAS: ${var.batch_alias}
              PREDICT_MODE: ${var.predict_mode}
              ALLOW_CHAMPION_FALLBACK: ${var.allow_champion_fallback}
              ALLOW_CANARY: ${var.allow_canary}
              EXPERIMENT_PATH: ${var.experiment_path}
              ENDPOINT_NAME: ${var.endpoint_name}
              ONLINE_STORE_NAME: ${var.online_store_name}
              ONLINE_CATALOG: ${var.online_catalog}
              ONLINE_FEATURE_TABLE: ${var.online_feature_table}
              ONLINE_STORE_CAPACITY: ${var.online_store_capacity}
              ONLINE_SYNC_PIPELINE_NAME: ${var.online_sync_pipeline_name}
              JOB_RUN_AS_SP: ${var.job_run_as_sp}
              TRAIN_RUN_AS_SP: ${var.train_run_as_sp}
              APPROVER_IDENTITIES: ${var.approver_identities}
              COMPARE_METRIC: ${var.compare_metric}
              COMPARE_HIGHER_IS_BETTER: ${var.compare_higher_is_better}
              COMPARE_MARGIN: ${var.compare_margin}
              MIN_EVAL_ROWS: ${var.min_eval_rows}
              MIN_TABLE_ROWS: ${var.min_table_rows}
              MAX_NULL_RATE: ${var.max_null_rate}
              LABEL_WINDOW: ${var.label_window}
              EVAL_MIN_ROC_AUC: ${var.eval_min_roc_auc}
              TRAIN_SEED: ${var.train_seed}
              CANARY_PERCENT: ${var.canary_percent}
              CANARY_WARMUP_S: ${var.canary_warmup_s}
              ENV_MANAGER: ${var.env_manager}
              SECRET_SCOPE: ${var.secret_scope}
              RAW_TABLE: ${var.raw_table}
              FEATURE_TABLE: ${var.feature_table}
              LABEL_TABLE: ${var.label_table}
              PREDICTIONS_TABLE: ${var.predictions_table}
              INFERENCE_TABLE: ${var.inference_table}
              EDA_PROFILE_TABLE: ${var.eda_profile_table}
              XAI_EXPLANATION_TABLE: ${var.xai_explanation_table}
              MONITOR_STATUS_TABLE: ${var.monitor_status_table}
              LABEL_COL: ${var.label_col}
              ID_COL: ${var.id_col}
              EVENT_TIME_COL: ${var.event_time_col}
"""

ENV = """        - environment_key: ml_env
          spec:
            client: "4"
            dependencies:
{deps}
"""

CLASSIC = """      job_clusters:
        - job_cluster_key: classic_ml
          new_cluster:
            spark_version: ${{var.classic_spark_version}}
            node_type_id: ${{var.classic_node_type_id}}
            autoscale:
              min_workers: ${{var.classic_min_workers}}
              max_workers: ${{var.classic_max_workers}}
            data_security_mode: ${{var.classic_data_security_mode}}
            spark_env_vars:
              PYTHONPATH: ${{workspace.file_path}}
"""

PERMS = """      permissions:
        - service_principal_name: ${var.ci_trigger_sp}
          level: CAN_MANAGE_RUN
        - group_name: ${var.approver_group}
          level: CAN_VIEW
"""

ML = """              - mlflow==3.3.2
              - scikit-learn==1.6.1
              - xgboost==2.1.4
              - pandas==2.2.3
              - databricks-feature-engineering>=0.16.0
              - "-e ${workspace.file_path}"
"""
TRAIN_DEPS = ML + """              - shap==0.47.2
              - matplotlib==3.10.1
"""
SKLEARN = """              - scikit-learn==1.6.1
              - pandas==2.2.3
              - databricks-feature-engineering>=0.16.0
              - "-e ${workspace.file_path}"
"""


def task(key, notebook_var, python_var, depends=None, extra_params=None, first=False, max_retries=None, run_if=None, job="train"):
    lines = [f"        - task_key: {key}"]
    if max_retries is not None:
        lines.append(f"          max_retries: {max_retries}")
    if depends:
        lines.append("          depends_on:")
        for d in depends:
            lines.append(f"            - task_key: {d}")
    if run_if:
        lines.append(f"          run_if: {run_if}")
    lines.append(f"          environment_key: ${{var.env_key_{job}_{key}}}")
    # Serverless-only workspaces: omit job_cluster_key. Cleanup keeps classic via YAML.
    lines.append("          notebook_task:")
    lines.append(f"            notebook_path: ${{var.{notebook_var}}}")
    if extra_params:
        lines.append("            base_parameters:")
        lines.append("              <<: *mlops_params")
        for k, v in extra_params.items():
            lines.append(f"              {k}: {v}")
    elif first:
        lines.append(PARAMS.replace("{evaluate_mode}", "${var.evaluate_mode}"))
    else:
        lines.append("            base_parameters: *mlops_params")
    lines.append(f"            # spark_python_task:")
    lines.append(f"            #   python_file: ${{var.{python_var}}}")
    return "\n".join(lines)


def first_params(evaluate_mode: str) -> str:
    return PARAMS.replace("{evaluate_mode}", evaluate_mode)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8")
    print(path)


def main() -> None:
    # --- train ---
    train_tasks = []
    seq = [
        ("seed", "seed_notebook_path", "seed_python_path", None),
        ("data_checks", "data_checks_notebook_path", "data_checks_python_path", ["seed"]),
        ("eda", "eda_notebook_path", "eda_python_path", ["data_checks"]),
        ("features", "features_notebook_path", "features_python_path", ["eda"]),
        ("train", "train_notebook_path", "train_python_path", ["features"]),
        ("evaluate", "evaluate_notebook_path", "evaluate_python_path", ["train"]),
        ("compare", "compare_notebook_path", "compare_python_path", ["evaluate"]),
        ("xai", "xai_notebook_path", "xai_python_path", ["evaluate"]),
        ("promotion_ready", "promotion_ready_notebook_path", "promotion_ready_python_path", ["compare"]),
        ("smoke", "smoke_notebook_path", "smoke_python_path", ["compare"]),
    ]
    for i, (key, nb, py, dep) in enumerate(seq):
        if i == 0:
            block = f"""        - task_key: {key}
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.{nb}}}
{first_params("${var.evaluate_mode}")}            # spark_python_task:
            #   python_file: ${{var.{py}}}"""
        else:
            block = task(key, nb, py, depends=dep, first=False)
        train_tasks.append(block)

    write(
        ROOT / "resources/dev/train.job.yml",
        f"""# Dev-only train graph. Do not include this file in the prod target.
resources:
  jobs:
    train:
      name: ${{var.train_job_name}}
      max_concurrent_runs: 1
      tags:
        example: binary_classifier
        dataset: adult
      schedule:
        quartz_cron_expression: "0 0 2 * * ?"
        timezone_id: UTC
        pause_status: PAUSED
      run_as:
        service_principal_name: ${{var.train_run_as_sp}}
      environments:
{ENV.format(deps=TRAIN_DEPS)}{PERMS}      tasks:
{chr(10).join(train_tasks)}
""",
    )

    write(
        ROOT / "resources/prod/promotion_gate.job.yml",
        f"""# Prod-only. copy_register → evaluate mode=prod_gate. No approval/deploy.
resources:
  jobs:
    promotion_gate:
      name: ${{var.promotion_gate_job_name}}
      max_concurrent_runs: 1
      tags:
        example: binary_classifier
        dataset: adult
      run_as:
        service_principal_name: ${{var.job_run_as_sp}}
      parameters:
        - name: source_model_version
          default: ""
        - name: promotion_reason
          default: ""
      environments:
{ENV.format(deps=ML)}{PERMS}      tasks:
        - task_key: copy_register
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.copy_register_notebook_path}}
{first_params("prod_gate")}            # spark_python_task:
            #   python_file: ${{var.copy_register_python_path}}
        - task_key: evaluate
          depends_on:
            - task_key: copy_register
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.evaluate_notebook_path}}
            base_parameters: *mlops_params
            # spark_python_task:
            #   python_file: ${{var.evaluate_python_path}}
""",
    )

    write(
        ROOT / "resources/prod/promotion_cutover.job.yml",
        f"""# Prod-only cutover. Dest model_version is not a job parameter.
resources:
  jobs:
    promotion_cutover:
      name: ${{var.promotion_cutover_job_name}}
      max_concurrent_runs: 1
      tags:
        example: binary_classifier
        dataset: adult
      run_as:
        service_principal_name: ${{var.job_run_as_sp}}
      parameters:
        - name: source_model_version
          default: ""
        - name: promotion_reason
          default: ""
      environments:
{ENV.format(deps=ML)}{PERMS}      tasks:
        - task_key: approval_check
          max_retries: 0
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.approval_notebook_path}}
{first_params("prod_gate")}            # spark_python_task:
            #   python_file: ${{var.approval_python_path}}
        - task_key: deploy
          depends_on:
            - task_key: approval_check
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.deploy_notebook_path}}
            base_parameters: *mlops_params
            # spark_python_task:
            #   python_file: ${{var.deploy_python_path}}
        - task_key: canary_start
          depends_on:
            - task_key: deploy
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.canary_notebook_path}}
            base_parameters:
              <<: *mlops_params
              CANARY_ACTION: canary_start
            # spark_python_task:
            #   python_file: ${{var.canary_python_path}}
        - task_key: metrics_gate
          depends_on:
            - task_key: canary_start
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.canary_notebook_path}}
            base_parameters:
              <<: *mlops_params
              CANARY_ACTION: metrics_gate
            # spark_python_task:
            #   python_file: ${{var.canary_python_path}}
        - task_key: to_100
          depends_on:
            - task_key: metrics_gate
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.canary_notebook_path}}
            base_parameters:
              <<: *mlops_params
              CANARY_ACTION: to_100
            # spark_python_task:
            #   python_file: ${{var.canary_python_path}}
        - task_key: rollback
          depends_on:
            - task_key: deploy
            - task_key: canary_start
            - task_key: metrics_gate
            - task_key: to_100
          run_if: AT_LEAST_ONE_FAILED
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.rollback_notebook_path}}
            base_parameters: *mlops_params
            # spark_python_task:
            #   python_file: ${{var.rollback_python_path}}
""",
    )

    for name, job_key, job_name_var, notebook, python, extra_deps in [
        ("batch.job.yml", "batch", "batch_job_name", "batch_notebook_path", "batch_python_path", ML),
        ("monitor.job.yml", "monitor", "monitor_job_name", "monitor_notebook_path", "monitor_python_path", """              - mlflow==3.3.2
              - scikit-learn==1.6.1
              - pandas==2.2.3
"""),
    ]:
        cron = "0 0 7 * * ?" if job_key == "batch" else "0 30 7 * * ?"
        extra_job_params = ""
        if job_key == "batch":
            extra_job_params = """      parameters:
        - name: predict_mode
          default: ${var.predict_mode}
"""
        write(
            ROOT / "resources/shared" / name,
            f"""resources:
  jobs:
    {job_key}:
      name: ${{var.{job_name_var}}}
      max_concurrent_runs: 1
      tags:
        example: binary_classifier
        dataset: adult
      schedule:
        quartz_cron_expression: "{cron}"
        timezone_id: UTC
        pause_status: PAUSED
{extra_job_params}      run_as:
        service_principal_name: ${{var.job_run_as_sp}}
      environments:
{ENV.format(deps=extra_deps)}{PERMS}      tasks:
        - task_key: {job_key}
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.{notebook}}}
{first_params("${var.evaluate_mode}")}            # spark_python_task:
            #   python_file: ${{var.{python}}}
""",
        )

    write(
        ROOT / "resources/shared/feature_refresh.job.yml",
        f"""# Feature ETL for the active catalog. Not training; does not register models.
resources:
  jobs:
    feature_refresh:
      name: ${{var.feature_refresh_job_name}}
      max_concurrent_runs: 1
      tags:
        example: binary_classifier
        dataset: adult
      schedule:
        quartz_cron_expression: "0 0 5 * * ?"
        timezone_id: UTC
        pause_status: PAUSED
      run_as:
        service_principal_name: ${{var.job_run_as_sp}}
      environments:
{ENV.format(deps=SKLEARN)}{PERMS}      tasks:
        - task_key: seed
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.seed_notebook_path}}
{first_params("${var.evaluate_mode}")}            # spark_python_task:
            #   python_file: ${{var.seed_python_path}}
        - task_key: data_checks
          depends_on:
            - task_key: seed
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.data_checks_notebook_path}}
            base_parameters: *mlops_params
            # spark_python_task:
            #   python_file: ${{var.data_checks_python_path}}
        - task_key: features
          depends_on:
            - task_key: data_checks
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.features_notebook_path}}
            base_parameters: *mlops_params
            # spark_python_task:
            #   python_file: ${{var.features_python_path}}
""",
    )


    write(
        ROOT / "resources/shared/cleanup.job.yml",
        f"""# Manual teardown of objects this example created on the active target.
# No schedule — run with: databricks bundle run cleanup --target <dev|prod>
resources:
  jobs:
    cleanup:
      name: ${{var.cleanup_job_name}}
      max_concurrent_runs: 1
      tags:
        example: binary_classifier
        dataset: adult
      parameters:
        - name: delete_kinds
          default: ""
      run_as:
        service_principal_name: ${{var.job_run_as_sp}}
      environments:
{ENV.format(deps=ML)}{PERMS}      tasks:
        - task_key: cleanup
          environment_key: ml_env
          notebook_task:
            notebook_path: ${{var.cleanup_notebook_path}}
{first_params("${{var.evaluate_mode}}")}              VOLUME_NAME: ${{var.volume_name}}
            # spark_python_task:
            #   python_file: ${{var.cleanup_python_path}}
""",
    )


if __name__ == "__main__":
    main()
