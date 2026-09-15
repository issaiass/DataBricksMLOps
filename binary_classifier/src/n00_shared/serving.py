"""Model Serving helpers: pin entity_version and wait until READY.

Adult endpoints serve P(income > 50K) from the registered sklearn pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def served_entity_name(model_name: str, version: str) -> str:
    short = str(model_name).split(".")[-1]
    return f"{short}-{version}"


def serving_config(
    entity_name: str,
    version: str,
    *,
    previous_version: Optional[str] = None,
    canary_percent: int = 0,
    workload_size: str = "Small",
) -> Dict[str, Any]:
    entities: List[Dict[str, Any]] = [
        {
            "name": served_entity_name(entity_name, version),
            "entity_name": entity_name,
            "entity_version": str(version),
            "workload_size": workload_size,
            "scale_to_zero_enabled": True,
        }
    ]
    routes = [
        {
            "served_model_name": served_entity_name(entity_name, version),
            "traffic_percentage": int(canary_percent) if previous_version else 100,
        }
    ]
    if previous_version and canary_percent < 100:
        entities.append(
            {
                "name": served_entity_name(entity_name, previous_version),
                "entity_name": entity_name,
                "entity_version": str(previous_version),
                "workload_size": workload_size,
                "scale_to_zero_enabled": True,
            }
        )
        routes = [
            {
                "served_model_name": served_entity_name(entity_name, version),
                "traffic_percentage": int(canary_percent),
            },
            {
                "served_model_name": served_entity_name(entity_name, previous_version),
                "traffic_percentage": 100 - int(canary_percent),
            },
        ]
    elif previous_version and canary_percent >= 100:
        routes = [
            {
                "served_model_name": served_entity_name(entity_name, version),
                "traffic_percentage": 100,
            }
        ]
    return {"served_entities": entities, "traffic_config": {"routes": routes}}


def flatten_endpoint_scores(payload) -> list[float]:
    """Parse a Model Serving / MLflow deployments predict payload into P(>50K) scores."""
    if payload is None:
        raise RuntimeError("empty serving response")
    if isinstance(payload, dict):
        preds = payload.get("predictions", payload.get("prediction"))
        if preds is None:
            raise RuntimeError(f"serving payload missing predictions: {list(payload)}")
    else:
        preds = payload
    if isinstance(preds, (int, float, str)):
        preds = [preds]
    out: list[float] = []
    for item in preds:
        if isinstance(item, dict):
            if "score" in item:
                out.append(float(item["score"]))
            elif "prediction" in item:
                val = item["prediction"]
                if isinstance(val, (list, tuple)):
                    out.append(float(val[1] if len(val) > 1 else val[0]))
                else:
                    out.append(float(val))
            else:
                nums = [v for v in item.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if not nums:
                    raise RuntimeError(f"cannot parse score from {item}")
                out.append(float(nums[0]))
        elif isinstance(item, (list, tuple)):
            out.append(float(item[1] if len(item) > 1 else item[0]))
        else:
            out.append(float(item))
    return out


def query_endpoint_scores(endpoint_name: str, records: list[dict], *, chunk_size: int = 256) -> list[float]:
    """Score feature-row dicts on an existing endpoint. Does not create or update serving."""
    name = str(endpoint_name).strip()
    if not name:
        raise RuntimeError("endpoint_name is required for serving predict")
    if not records:
        return []
    wait_endpoint_ready(name, timeout_s=300, poll_s=10)
    scores: list[float] = []
    step = max(int(chunk_size), 1)
    last_err: Exception | None = None
    for start in range(0, len(records), step):
        chunk = records[start : start + step]
        try:
            from mlflow.deployments import get_deploy_client

            payload = get_deploy_client("databricks").predict(
                endpoint=name, inputs={"dataframe_records": chunk}
            )
            scores.extend(flatten_endpoint_scores(payload))
            last_err = None
            continue
        except Exception as exc:
            last_err = exc
        try:
            from databricks.sdk import WorkspaceClient

            resp = WorkspaceClient().serving_endpoints.query(name=name, dataframe_records=chunk)
            payload = resp.as_dict() if hasattr(resp, "as_dict") else resp
            scores.extend(flatten_endpoint_scores(payload))
            last_err = None
        except Exception as inner:
            raise RuntimeError(
                f"serving predict failed on {name} rows[{start}:{start + len(chunk)}]: {inner}"
            ) from (last_err or inner)
    if len(scores) != len(records):
        raise RuntimeError(f"serving returned {len(scores)} scores for {len(records)} rows")
    return scores


def wait_endpoint_ready(endpoint_name: str, timeout_s: int = 1800, poll_s: int = 15) -> None:
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = client.get_endpoint(endpoint_name)
        state = (last or {}).get("state") or {}
        ready = str(state.get("ready") or "")
        updating = str(state.get("config_update") or "")
        if ready == "READY" and updating in {"", "NOT_UPDATING"}:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"endpoint {endpoint_name} never READY: {last}")


def upsert_endpoint(endpoint_name: str, config: Dict[str, Any]) -> None:
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    try:
        client.get_endpoint(endpoint_name)
        client.update_endpoint(endpoint=endpoint_name, config=config)
    except Exception:
        client.create_endpoint(name=endpoint_name, config=config)
    wait_endpoint_ready(endpoint_name)


def _missing_endpoint(exc: BaseException) -> bool:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "resource_does_not_exist" in text
        or "does not exist" in text
        or "not found" in text
        or "notfound" in name
        or "resourcedoesnotexist" in name
    )


def delete_endpoint(endpoint_name: str) -> str:
    """Delete a Model Serving endpoint. Missing endpoint is a no-op."""
    name = str(endpoint_name).strip()
    if not name:
        raise RuntimeError("endpoint_name is required for cleanup")
    try:
        from databricks.sdk import WorkspaceClient

        WorkspaceClient().serving_endpoints.delete(name=name)
        return f"deleted endpoint {name}"
    except Exception as exc:
        if _missing_endpoint(exc):
            return f"endpoint {name} already absent"
        try:
            from mlflow.deployments import get_deploy_client

            get_deploy_client("databricks").delete_endpoint(name)
            return f"deleted endpoint {name}"
        except Exception as inner:
            if _missing_endpoint(inner):
                return f"endpoint {name} already absent"
            raise RuntimeError(f"failed to delete endpoint {name}: {inner}") from inner
