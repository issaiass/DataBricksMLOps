"""Optional canary traffic for the Adult prod endpoint (does not flip @prod until to_100)."""

from __future__ import annotations

import time

from src.n03_prod_cutover.n01_approval import resolve_cutover_dest_version
from src.n00_shared.runtime import Settings, configure_mlflow, current_task_key, load_settings, mlflow_client
from src.n00_shared.serving import serving_config, upsert_endpoint, wait_endpoint_ready


def _tag_map(client, name: str, version: str) -> dict:
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    if isinstance(tags, dict):
        return tags
    return {t.key: t.value for t in tags}


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    task = current_task_key() or __import__("os").environ.get("CANARY_ACTION", "")
    client = mlflow_client()
    dest = settings.dest_model_name
    version = resolve_cutover_dest_version(settings)
    tags = _tag_map(client, dest, version)
    previous = str(tags.get("previous_prod_version") or "")
    if not settings.allow_canary or not previous:
        print(f"{task or 'canary'}: skipped (allow_canary={settings.allow_canary} previous={previous})")
        return
    if task in {"", "canary_start", "canary"}:
        upsert_endpoint(
            settings.endpoint_name,
            serving_config(
                dest,
                version,
                previous_version=previous,
                canary_percent=settings.canary_percent,
            ),
        )
        time.sleep(min(settings.canary_warmup_s, 30))
        print(f"canary_start {settings.canary_percent}% on v{version}")
        return
    if task == "metrics_gate":
        wait_endpoint_ready(settings.endpoint_name)
        print("metrics_gate: endpoint READY (wire live canary SLOs when traffic exists)")
        return
    if task == "to_100":
        client.set_registered_model_alias(dest, "prod", version)
        client.set_registered_model_alias(dest, "champion", version)
        upsert_endpoint(
            settings.endpoint_name,
            serving_config(dest, version, previous_version=None, canary_percent=100),
        )
        print(f"to_100 set @prod/@champion and serving 100% to v{version}")
        return
    raise RuntimeError(f"unknown canary task {task}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
