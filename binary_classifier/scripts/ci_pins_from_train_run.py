"""Read promotion_ready task values from a Databricks Jobs get-run JSON payload.

CI MUST pin source_model_version from that train run's go_version (never live
@champion, never an empty pin). ready=false is not promotable and is not an
infra failure. Missing ready, or ready=true without go_version, fails closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


TASK_KEY = "promotion_ready"


class PinError(ValueError):
    """Fail closed: cannot start promotion_gate."""


def _as_map(values: Any) -> dict[str, str]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {str(k): str(v) for k, v in values.items() if k is not None}
    if isinstance(values, list):
        out: dict[str, str] = {}
        for item in values:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key") or item.get("name")
            if key is None:
                continue
            val = item.get("value", item.get("val"))
            if val is None:
                continue
            out[str(key)] = str(val)
        return out
    return {}


def _task_maps(task: Mapping[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key in ("values", "task_values", "taskValues"):
        merged.update(_as_map(task.get(key)))
    status = task.get("status")
    if isinstance(status, Mapping):
        for key in ("values", "task_values"):
            merged.update(_as_map(status.get(key)))
    return merged


def _iter_tasks(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks: list[Mapping[str, Any]] = []
    raw = payload.get("tasks") or payload.get("task_runs") or []
    if isinstance(raw, list):
        tasks.extend(t for t in raw if isinstance(t, Mapping))
    history = payload.get("repair_history")
    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, Mapping):
                continue
            nested = entry.get("tasks") or []
            if isinstance(nested, list):
                tasks.extend(t for t in nested if isinstance(t, Mapping))
    return tasks


def promotion_ready_values(payload: Mapping[str, Any], task_key: str = TASK_KEY) -> dict[str, str]:
    found: Optional[dict[str, str]] = None
    for task in _iter_tasks(payload):
        if str(task.get("task_key") or task.get("taskKey") or "") != task_key:
            continue
        found = _task_maps(task)
    if found is None:
        raise PinError(f"train run has no task_key={task_key}")
    return found


def pins_from_train_run(
    payload: Mapping[str, Any],
    train_run_id: str,
    task_key: str = TASK_KEY,
) -> dict[str, str]:
    """Return GitHub-style outputs. Raises PinError when the gate must not start."""
    reason = str(train_run_id or "").strip()
    if not reason:
        raise PinError("CI MUST never pass an empty promotion_reason / train_run_id")
    json_run_id = str(payload.get("run_id") or payload.get("runId") or "").strip()
    if json_run_id and json_run_id != reason:
        raise PinError(
            f"train_run_id {reason} does not match get-run run_id {json_run_id}"
        )
    values = promotion_ready_values(payload, task_key=task_key)
    ready = str(values.get("ready") or "").strip().lower()
    go_version = str(values.get("go_version") or "").strip()
    if not ready:
        raise PinError("missing promotion_ready.ready; do not start prod")
    if ready != "true":
        return {
            "promotable": "false",
            "source_model_version": "",
            "promotion_reason": reason,
            "ready": ready,
        }
    if not go_version:
        raise PinError("ready=true but go_version is empty; CI MUST never pass an empty pin")
    return {
        "promotable": "true",
        "source_model_version": go_version,
        "promotion_reason": reason,
        "ready": "true",
    }


def _write_github_output(path: Path, pins: Mapping[str, str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for key, val in pins.items():
            fh.write(f"{key}={val}\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-file", required=True, help="jobs get-run --output json file")
    parser.add_argument("--train-run-id", required=True, help="same id passed to get-run")
    parser.add_argument(
        "--github-output",
        default="",
        help="append key=value lines (set to $GITHUB_OUTPUT in Actions)",
    )
    args = parser.parse_args(argv)
    raw = Path(args.json_file).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise PinError("get-run JSON must be an object")
    pins = pins_from_train_run(payload, args.train_run_id)
    if args.github_output:
        _write_github_output(Path(args.github_output), pins)
    for key, val in pins.items():
        print(f"{key}={val}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PinError as exc:
        print(f"pin_error: {exc}", file=sys.stderr)
        raise SystemExit(2)
