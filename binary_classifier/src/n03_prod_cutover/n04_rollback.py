"""Restore previous Adult @prod / serving pin after a failed deploy or canary."""

from __future__ import annotations

from src.n03_prod_cutover.n01_approval import resolve_cutover_dest_version
from src.n00_shared.runtime import Settings, configure_mlflow, get_task_value, load_settings, mlflow_client
from src.n00_shared.serving import serving_config, upsert_endpoint


def _tag_map(client, name: str, version: str) -> dict:
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    if isinstance(tags, dict):
        return tags
    return {t.key: t.value for t in tags}


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    client = mlflow_client()
    dest = settings.dest_model_name
    try:
        current = resolve_cutover_dest_version(settings)
    except Exception:
        current = get_task_value("deploy", "model_version", "")
    tags = _tag_map(client, dest, current) if current else {}
    previous = tags.get("previous_prod_version") or get_task_value("deploy", "previous_prod_version", "")
    if not previous:
        print("rollback: no previous @prod; aliases unchanged; revert serving if needed")
        return
    client.set_registered_model_alias(dest, "champion", previous)
    client.set_registered_model_alias(dest, "prod", previous)
    upsert_endpoint(
        settings.endpoint_name,
        serving_config(dest, previous, previous_version=None, canary_percent=100),
    )
    print(f"rollback restored @champion/@prod and serving to v{previous}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
