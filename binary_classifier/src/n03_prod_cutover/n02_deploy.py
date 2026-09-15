"""Pin the approved Adult dest version for serving / @champion (and @prod if no canary)."""

from __future__ import annotations

from mlflow.exceptions import MlflowException

from src.n03_prod_cutover.n01_approval import resolve_cutover_dest_version
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings, mlflow_client, set_task_value
from src.n00_shared.serving import serving_config, upsert_endpoint


def _alias_version(client, name: str, alias: str):
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None
    except Exception:
        return None


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    client = mlflow_client()
    dest = settings.dest_model_name
    version = resolve_cutover_dest_version(settings)
    previous = _alias_version(client, dest, "prod")
    client.set_registered_model_alias(dest, "champion", version)
    if previous:
        client.set_model_version_tag(dest, version, "previous_prod_version", previous)
        set_task_value("previous_prod_version", previous)
    else:
        client.set_model_version_tag(dest, version, "previous_prod_version", "")
        set_task_value("previous_prod_version", "")
    first_or_no_canary = (not previous) or (not settings.allow_canary)
    if first_or_no_canary:
        client.set_registered_model_alias(dest, "prod", version)
        upsert_endpoint(
            settings.endpoint_name,
            serving_config(dest, version, previous_version=None, canary_percent=100),
        )
        print(f"deploy set @prod and serving 100% to v{version}")
    else:
        upsert_endpoint(
            settings.endpoint_name,
            serving_config(
                dest,
                version,
                previous_version=previous,
                canary_percent=settings.canary_percent,
            ),
        )
        print(f"deploy left @prod on {previous}; serving canary {settings.canary_percent}% -> v{version}")
    set_task_value("model_version", version)


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
