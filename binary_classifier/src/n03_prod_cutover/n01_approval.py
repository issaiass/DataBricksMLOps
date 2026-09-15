"""Read-only approval gate for the Adult dest @challenger version."""

from __future__ import annotations

from src.n00_shared.protocol import challenger_pin_matches, check_approval_read
from src.n00_shared.runtime import Settings, configure_mlflow, load_settings, mlflow_client, operator_param, set_task_value


def _tag_map(client, name: str, version: str) -> dict:
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    if isinstance(tags, dict):
        return tags
    return {t.key: t.value for t in tags}


def resolve_cutover_dest_version(settings: Settings) -> str:
    pin = operator_param("source_model_version", "").strip()
    if not pin:
        raise RuntimeError("cutover requires the same source_model_version pin as the gate")
    client = mlflow_client()
    challenger = client.get_model_version_by_alias(settings.dest_model_name, "challenger")
    tags = _tag_map(client, settings.dest_model_name, challenger.version)
    if not challenger_pin_matches(tags.get("source_model_version"), pin):
        raise RuntimeError("dest @challenger source_model_version tag does not match the pin")
    return str(challenger.version)


def run(settings: Settings) -> None:
    configure_mlflow(settings)
    version = resolve_cutover_dest_version(settings)
    client = mlflow_client()
    tags = _tag_map(client, settings.dest_model_name, version)
    check_approval_read(
        tags.get("approval_check"),
        tags.get("approved_by"),
        settings.approver_identities,
        settings.job_run_as_sp,
    )
    set_task_value("model_version", version)
    print(f"approval_check read-only ok dest v{version}")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
