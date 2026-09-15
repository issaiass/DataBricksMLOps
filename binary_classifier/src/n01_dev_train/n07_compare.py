"""Compare Adult challenger vs champion on absolute-eval tags only (no aliases)."""

from __future__ import annotations

from mlflow.exceptions import MlflowException

from src.n00_shared.protocol import compare_decision
from src.n00_shared.runtime import Settings, assert_dev_ml_allowed, configure_mlflow, get_task_value, load_settings, mlflow_client, set_task_value


def _tag_map(client, name: str, version: str) -> dict:
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    if isinstance(tags, dict):
        return tags
    return {t.key: t.value for t in tags}


def _metric_from_tags(tags: dict, metric: str):
    raw = tags.get(metric)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def run(settings: Settings) -> None:
    assert_dev_ml_allowed(settings)
    configure_mlflow(settings)
    client = mlflow_client()
    version = get_task_value("evaluate", "model_version")
    tags = _tag_map(client, settings.source_model_name, version)
    challenger_metric = _metric_from_tags(tags, settings.compare_metric)
    champion_exists = True
    champion_metric = None
    try:
        champ = client.get_model_version_by_alias(settings.source_model_name, "champion")
        champ_tags = _tag_map(client, settings.source_model_name, champ.version)
        champion_metric = _metric_from_tags(champ_tags, settings.compare_metric)
    except MlflowException:
        champion_exists = False
    except Exception:
        champion_exists = False
    result, first_version = compare_decision(
        challenger_metric,
        champion_metric,
        settings.compare_higher_is_better,
        settings.compare_margin,
        champion_exists,
    )
    client.set_model_version_tag(settings.source_model_name, version, "compare_result", result)
    if first_version:
        client.set_model_version_tag(settings.source_model_name, version, "first_version", "true")
    set_task_value("compare_result", result)
    set_task_value("model_version", version)
    print(f"compare tags-only result={result} first_version={first_version}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
