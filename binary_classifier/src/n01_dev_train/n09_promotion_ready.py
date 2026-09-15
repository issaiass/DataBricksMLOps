"""Mark a go Adult model promotion-ready (sets ml_dev @champion on go only)."""

from __future__ import annotations

from src.n00_shared.runtime import (
    Settings,
    assert_dev_ml_allowed,
    configure_mlflow,
    get_task_value,
    job_run_id,
    load_settings,
    mlflow_client,
    set_task_value,
)


def _tag_map(client, name: str, version: str) -> dict:
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    if isinstance(tags, dict):
        return tags
    return {t.key: t.value for t in tags}


def run(settings: Settings) -> None:
    assert_dev_ml_allowed(settings)
    configure_mlflow(settings)
    client = mlflow_client()
    version = get_task_value("compare", "model_version")
    tags = _tag_map(client, settings.source_model_name, version)
    go = str(tags.get("compare_result") or "") == "go"
    if go:
        client.set_registered_model_alias(settings.source_model_name, "champion", version)
        client.set_model_version_tag(settings.source_model_name, version, "ready", "true")
        client.set_model_version_tag(settings.source_model_name, version, "go_version", str(version))
        run_id = job_run_id()
        client.set_model_version_tag(settings.source_model_name, version, "train_run_id", run_id)
        set_task_value("ready", "true")
        set_task_value("go_version", str(version))
        set_task_value("train_run_id", run_id)
        print(f"promotion_ready go version={version}")
        return
    client.set_model_version_tag(settings.source_model_name, version, "ready", "false")
    set_task_value("ready", "false")
    print("promotion_ready no-go; champion unchanged; job succeeds")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
