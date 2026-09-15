from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_render_jobs_task_comments_python_twin():
    mod = _load("render_jobs", ROOT / "scripts" / "render_jobs.py")
    block = mod.task("evaluate", "evaluate_notebook_path", "evaluate_python_path", depends=["train"])
    assert "notebook_task:" in block
    assert "# spark_python_task:" in block
    assert "evaluate_python_path" in block
    assert "depends_on:" in block
    assert "task_key: train" in block


def test_render_jobs_first_params_bakes_prod_gate_mode():
    mod = _load("render_jobs", ROOT / "scripts" / "render_jobs.py")
    text = mod.first_params("prod_gate")
    assert "EVALUATE_MODE: prod_gate" in text
    assert "SOURCE_MODEL_NAME" in text


def test_generate_notebooks_steps_cover_src_tree():
    gen = _load("generate_notebooks", ROOT / "scripts" / "generate_notebooks.py")
    src_steps = sorted(
        p.relative_to(ROOT / "src").as_posix()
        for p in (ROOT / "src").glob("n0[1-4]_*/*.py")
        if p.name != "__init__.py"
    )
    gen_steps = sorted(f"{group}/{stem}.py" for group, stem, _t in gen.STEPS)
    assert src_steps == gen_steps


def _pins():
    return _load("ci_pins", ROOT / "scripts" / "ci_pins_from_train_run.py")


def _run(ready: str, go_version: str, run_id: int = 99, values_as_list: bool = False):
    values = {"ready": ready, "go_version": go_version, "train_run_id": str(run_id)}
    if values_as_list:
        values = [{"key": k, "value": v} for k, v in values.items()]
    return {
        "run_id": run_id,
        "tasks": [{"task_key": "promotion_ready", "values": values}],
    }


def test_ci_pins_go_from_train_run():
    pins = _pins()
    out = pins.pins_from_train_run(_run("true", "6"), "99")
    assert out["promotable"] == "true"
    assert out["source_model_version"] == "6"
    assert out["promotion_reason"] == "99"


def test_ci_pins_list_shaped_task_values():
    pins = _pins()
    out = pins.pins_from_train_run(_run("true", "8", values_as_list=True), "99")
    assert out["source_model_version"] == "8"


def test_ci_pins_ready_false_is_not_promotable():
    pins = _pins()
    out = pins.pins_from_train_run(_run("false", ""), "99")
    assert out["promotable"] == "false"
    assert out["source_model_version"] == ""


def test_ci_pins_missing_ready_fails_closed():
    pins = _pins()
    payload = {"run_id": 99, "tasks": [{"task_key": "promotion_ready", "values": {}}]}
    try:
        pins.pins_from_train_run(payload, "99")
    except pins.PinError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected PinError")


def test_ci_pins_empty_go_version_when_ready_fails_closed():
    pins = _pins()
    try:
        pins.pins_from_train_run(_run("true", "  "), "99")
    except pins.PinError as exc:
        assert "empty pin" in str(exc)
    else:
        raise AssertionError("expected PinError")


def test_ci_pins_empty_train_run_id_fails_closed():
    pins = _pins()
    try:
        pins.pins_from_train_run(_run("true", "1"), "  ")
    except pins.PinError as exc:
        assert "empty promotion_reason" in str(exc)
    else:
        raise AssertionError("expected PinError")


def test_ci_pins_run_id_mismatch_fails_closed():
    pins = _pins()
    try:
        pins.pins_from_train_run(_run("true", "1"), "100")
    except pins.PinError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected PinError")
