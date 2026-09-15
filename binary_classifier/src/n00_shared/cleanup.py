"""Select which Adult-example UC / serving / Tracking objects a cleanup run may drop.

Empty delete_kinds means every kind this example creates. The serving endpoint
is always included, even when the operator names a subset.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from src.n00_shared.runtime import Settings, fq

ALL_KINDS: Tuple[str, ...] = (
    "endpoints",
    "functions",
    "views",
    "tables",
    "volumes",
    "models",
    "experiments",
    "schemas",
    "catalogs",
)

KIND_ALIASES = {
    "endpoint": "endpoints",
    "endpoints": "endpoints",
    "serving": "endpoints",
    "serving_endpoint": "endpoints",
    "serving_endpoints": "endpoints",
    "function": "functions",
    "functions": "functions",
    "view": "views",
    "views": "views",
    "table": "tables",
    "tables": "tables",
    "volume": "volumes",
    "volumes": "volumes",
    "model": "models",
    "models": "models",
    "registered_model": "models",
    "registered_models": "models",
    "experiment": "experiments",
    "experiments": "experiments",
    "schema": "schemas",
    "schemas": "schemas",
    "catalog": "catalogs",
    "catalogs": "catalogs",
}

PROTECTED_CATALOGS = frozenset(
    {
        "system",
        "hive_metastore",
        "samples",
        "information_schema",
        "spark_catalog",
    }
)

PROTECTED_SCHEMAS = frozenset({"information_schema", "default"})

Action = Tuple[str, str]


DEFAULT_KINDS: Tuple[str, ...] = tuple(k for k in ALL_KINDS if k != "catalogs")


def resolve_delete_kinds(raw: str) -> Tuple[str, ...]:
    """Parse operator CSV. Empty / all / * → every example kind except catalogs.

    Catalogs are admin-created (not a bundle resource) and opt-in via
    ``delete_kinds=catalogs`` or ``everything``. Endpoint is always included.
    """
    text = (raw or "").strip()
    if text.lower() in {"", "all", "*"}:
        selected = set(DEFAULT_KINDS)
    elif text.lower() == "everything":
        selected = set(ALL_KINDS)
    else:
        selected = set()
        unknown: List[str] = []
        for part in text.split(","):
            token = part.strip().lower()
            if not token:
                continue
            kind = KIND_ALIASES.get(token)
            if kind is None:
                unknown.append(token)
            else:
                selected.add(kind)
        if unknown:
            raise ValueError(
                "unknown delete_kinds "
                f"{unknown}; allowed: {', '.join(ALL_KINDS)}"
            )
        if not selected:
            selected = {"endpoints"}
    selected.add("endpoints")
    return tuple(kind for kind in ALL_KINDS if kind in selected)


def assert_catalog_drop_allowed(catalog: str) -> None:
    name = str(catalog or "").strip()
    if not name:
        raise RuntimeError("catalog name is empty; refuse catalog drop")
    if name.lower() in PROTECTED_CATALOGS:
        raise RuntimeError(f"refuse dropping protected catalog {name}")


def assert_schema_drop_allowed(catalog: str, schema: str) -> None:
    assert_catalog_drop_allowed(catalog)
    name = str(schema or "").strip()
    if not name:
        raise RuntimeError("schema name is empty; refuse schema drop")
    if name.lower() in PROTECTED_SCHEMAS:
        raise RuntimeError(f"refuse dropping protected schema {name}")


def example_tables(settings: Settings) -> List[str]:
    return [
        settings.raw_fq,
        settings.feature_fq,
        settings.label_fq,
        settings.predictions_fq,
        settings.inference_fq,
        settings.eda_fq,
        settings.xai_fq,
        settings.monitor_fq,
    ]


def example_volume(settings: Settings, volume_name: str) -> str:
    name = str(volume_name or "").strip()
    if not name:
        raise RuntimeError("VOLUME_NAME must be baked for volume cleanup")
    return fq(settings.catalog, settings.schema, name)


def example_schema(settings: Settings) -> str:
    assert_schema_drop_allowed(settings.catalog, settings.schema)
    return f"{settings.catalog}.{settings.schema}"


def example_experiment(settings: Settings) -> str:
    path = str(settings.experiment_path or "").strip()
    if not path:
        raise RuntimeError("EXPERIMENT_PATH must be baked for experiment cleanup")
    return path


def _dedupe(names: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in names:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def planned_actions(
    settings: Settings,
    kinds: Sequence[str],
    *,
    volume_name: str,
    functions: Iterable[str] = (),
    views: Iterable[str] = (),
    extra_tables: Iterable[str] = (),
    extra_volumes: Iterable[str] = (),
    extra_models: Iterable[str] = (),
) -> List[Action]:
    """Targets this example created on the *active* catalog / workspace."""
    chosen = set(kinds)
    actions: List[Action] = []
    if "endpoints" in chosen:
        actions.append(("endpoints", settings.endpoint_name))
    if "functions" in chosen:
        for fn in _dedupe(functions):
            actions.append(("functions", fn))
    if "views" in chosen:
        for view in _dedupe(views):
            actions.append(("views", view))
    if "tables" in chosen:
        for table in _dedupe([*example_tables(settings), *extra_tables]):
            actions.append(("tables", table))
    if "volumes" in chosen:
        for volume in _dedupe([example_volume(settings, volume_name), *extra_volumes]):
            actions.append(("volumes", volume))
    if "models" in chosen:
        for model in _dedupe([settings.active_model, *extra_models]):
            actions.append(("models", model))
    if "experiments" in chosen:
        actions.append(("experiments", example_experiment(settings)))
    if "schemas" in chosen:
        actions.append(("schemas", example_schema(settings)))
    if "catalogs" in chosen:
        assert_catalog_drop_allowed(settings.catalog)
        actions.append(("catalogs", settings.catalog))
    return actions
