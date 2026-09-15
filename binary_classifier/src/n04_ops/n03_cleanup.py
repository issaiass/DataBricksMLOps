"""Tear down Adult-example objects created by this bundle on the active target.

Manual job only (no schedule). Empty delete_kinds drops the serving endpoint,
schema functions/views, example + leftover tables/volumes, the registered
model, the target MLflow experiment, and the example schema. Catalog drop is
opt-in (``delete_kinds=catalogs`` or ``everything``) because catalogs are
admin-created, not a bundle resource. The serving endpoint is always deleted.
Bundle jobs and workspace files are left for `databricks bundle destroy`.
"""

from __future__ import annotations

from src.n00_shared.cleanup import planned_actions, resolve_delete_kinds
from src.n00_shared.runtime import Settings, baked, fq, load_settings, operator_param
from src.n00_shared.serving import delete_endpoint


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _sql_ident(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    return ".".join(f"`{p.replace('`', '')}`" for p in parts)


def _qualify(settings: Settings, raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    if text.count(".") >= 2:
        return text
    return fq(settings.catalog, settings.schema, text.split(".")[-1])


def _collect_names(spark, statement: str, settings: Settings) -> list[str]:
    try:
        rows = spark.sql(statement).collect()
    except Exception:
        return []
    names: list[str] = []
    for row in rows:
        mapping = row.asDict(recursive=True) if hasattr(row, "asDict") else {}
        is_temp = mapping.get("isTemporary") or mapping.get("is_temporary")
        if is_temp:
            continue
        raw = (
            mapping.get("function")
            or mapping.get("functionName")
            or mapping.get("viewName")
            or mapping.get("tableName")
            or mapping.get("volumeName")
            or mapping.get("name")
            or (row[0] if row else "")
        )
        qual = _qualify(settings, str(raw))
        if qual:
            names.append(qual)
    return names


def _schema_ident(settings: Settings) -> str:
    return f"{_sql_ident(settings.catalog)}.{_sql_ident(settings.schema)}"


def _list_schema_functions(spark, settings: Settings) -> list[str]:
    return _collect_names(
        spark,
        f"SHOW USER FUNCTIONS IN {_schema_ident(settings)}",
        settings,
    )


def _list_schema_views(spark, settings: Settings) -> list[str]:
    return _collect_names(spark, f"SHOW VIEWS IN {_schema_ident(settings)}", settings)


def _list_schema_tables(spark, settings: Settings) -> list[str]:
    return _collect_names(spark, f"SHOW TABLES IN {_schema_ident(settings)}", settings)


def _list_schema_volumes(spark, settings: Settings) -> list[str]:
    return _collect_names(spark, f"SHOW VOLUMES IN {_schema_ident(settings)}", settings)


def _list_schema_models(settings: Settings) -> list[str]:
    from src.n00_shared.runtime import mlflow_client

    prefix = f"{settings.catalog}.{settings.schema}."
    names: list[str] = []
    try:
        client = mlflow_client()
        page_token = None
        while True:
            kwargs = {"max_results": 100}
            if page_token:
                kwargs["page_token"] = page_token
            page = client.search_registered_models(**kwargs)
            items = list(page) if page is not None else []
            for rm in items:
                name = str(getattr(rm, "name", "") or "")
                if name.startswith(prefix):
                    names.append(name)
            page_token = getattr(page, "token", None) or getattr(page, "next_page_token", None)
            if not page_token:
                break
    except Exception:
        return []
    return names


def _drop_sql(spark, statement: str) -> str:
    spark.sql(statement)
    return statement


def _drop_model(name: str) -> str:
    from src.n00_shared.runtime import mlflow_client

    client = mlflow_client()
    try:
        client.delete_registered_model(name)
        return f"deleted registered model {name}"
    except Exception as exc:
        text = str(exc).lower()
        if "not found" in text or "does not exist" in text or "resource_does_not_exist" in text:
            return f"registered model {name} already absent"
        raise


def _drop_experiment(path: str) -> str:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        exp = client.get_experiment_by_name(path)
        if exp is None:
            return f"experiment {path} already absent"
        if str(getattr(exp, "lifecycle_stage", "")).lower() == "deleted":
            return f"experiment {path} already absent"
        client.delete_experiment(exp.experiment_id)
        return f"deleted experiment {path} id={exp.experiment_id}"
    except Exception as exc:
        text = str(exc).lower()
        if "not found" in text or "does not exist" in text or "resource_does_not_exist" in text:
            return f"experiment {path} already absent"
        raise


def _drop_table(spark, settings: Settings, name: str) -> str:
    if name == settings.feature_fq:
        try:
            from databricks.feature_engineering import FeatureEngineeringClient

            FeatureEngineeringClient(model_registry_uri="databricks-uc").drop_table(name=name)
            return f"dropped feature table {name}"
        except Exception:
            pass
    return _drop_sql(spark, f"DROP TABLE IF EXISTS {_sql_ident(name)}")


def _delete_kind(spark, settings: Settings, kind: str, name: str) -> str:
    if kind == "endpoints":
        return delete_endpoint(name)
    if kind == "functions":
        return _drop_sql(spark, f"DROP FUNCTION IF EXISTS {_sql_ident(name)}")
    if kind == "views":
        return _drop_sql(spark, f"DROP VIEW IF EXISTS {_sql_ident(name)}")
    if kind == "tables":
        return _drop_table(spark, settings, name)
    if kind == "volumes":
        return _drop_sql(spark, f"DROP VOLUME IF EXISTS {_sql_ident(name)}")
    if kind == "models":
        return _drop_model(name)
    if kind == "experiments":
        return _drop_experiment(name)
    if kind == "schemas":
        return _drop_sql(spark, f"DROP SCHEMA IF EXISTS {_sql_ident(name)} CASCADE")
    if kind == "catalogs":
        return _drop_sql(spark, f"DROP CATALOG IF EXISTS {_sql_ident(name)} CASCADE")
    raise RuntimeError(f"unsupported cleanup kind {kind}")


def run(settings: Settings) -> None:
    kinds = resolve_delete_kinds(operator_param("delete_kinds", ""))
    volume_name = baked("VOLUME_NAME", "mlops_files")
    spark = _spark()
    functions = _list_schema_functions(spark, settings) if "functions" in kinds else []
    views = _list_schema_views(spark, settings) if "views" in kinds else []
    extra_tables = _list_schema_tables(spark, settings) if "tables" in kinds else []
    extra_volumes = _list_schema_volumes(spark, settings) if "volumes" in kinds else []
    extra_models = _list_schema_models(settings) if "models" in kinds else []
    actions = planned_actions(
        settings,
        kinds,
        volume_name=volume_name,
        functions=functions,
        views=views,
        extra_tables=extra_tables,
        extra_volumes=extra_volumes,
        extra_models=extra_models,
    )
    print(f"cleanup kinds={','.join(kinds)} catalog={settings.catalog} actions={len(actions)}")
    errors: list[str] = []
    for kind, name in actions:
        try:
            result = _delete_kind(spark, settings, kind, name)
            print(f"ok {kind} {name}: {result}")
        except Exception as exc:
            msg = f"failed {kind} {name}: {exc}"
            print(msg)
            errors.append(msg)
    if errors:
        raise RuntimeError("cleanup failed: " + " | ".join(errors))
    print("cleanup finished")


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
