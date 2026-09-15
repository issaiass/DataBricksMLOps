from unittest.mock import MagicMock

import pytest

from src.n00_shared.runtime import Settings
from tests.conftest import FakeMlflowClient, make_settings


def test_copy_register_run_refuses_empty_pin(monkeypatch, settings: Settings):
    from src.n02_prod_gate import n01_copy_register as copy_register

    monkeypatch.setattr(copy_register, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(
        copy_register,
        "operator_param",
        lambda name, default="": "" if name == "source_model_version" else "reason",
    )
    with pytest.raises(ValueError, match="empty source_model_version"):
        copy_register.run(settings)


def test_copy_register_run_refuses_no_go(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n02_prod_gate import n01_copy_register as copy_register

    fake_client.tags[(settings.source_model_name, "3")] = {"compare_result": "no-go"}
    monkeypatch.setattr(copy_register, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(
        copy_register,
        "operator_param",
        lambda name, default="": "3" if name == "source_model_version" else "train-run-1",
    )
    monkeypatch.setattr(copy_register, "_spark", lambda: MagicMock())
    monkeypatch.setattr(copy_register, "_metastore_id", lambda *_a: "metastore-a")
    monkeypatch.setattr(copy_register, "mlflow_client", lambda: fake_client)
    with pytest.raises(ValueError, match="compare_result=go"):
        copy_register.run(settings)


def test_copy_register_cross_metastore_fail_closed(monkeypatch, settings: Settings):
    from src.n02_prod_gate import n01_copy_register as copy_register

    monkeypatch.setattr(copy_register, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(
        copy_register,
        "operator_param",
        lambda name, default="": "3" if name == "source_model_version" else "train-run-1",
    )
    monkeypatch.setattr(copy_register, "_spark", lambda: MagicMock())
    monkeypatch.setattr(
        copy_register,
        "_metastore_id",
        lambda _spark, catalog: "meta-dev" if catalog == settings.dev_catalog else "meta-prod",
    )
    with pytest.raises(RuntimeError, match="different metastores"):
        copy_register.run(settings)


def test_copy_register_go_sets_challenger_not_prod(
    monkeypatch, settings: Settings, fake_client: FakeMlflowClient
):
    from src.n02_prod_gate import n01_copy_register as copy_register

    fake_client.tags[(settings.source_model_name, "3")] = {"compare_result": "go"}
    monkeypatch.setattr(copy_register, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(
        copy_register,
        "operator_param",
        lambda name, default="": "3" if name == "source_model_version" else "train-run-1",
    )
    monkeypatch.setattr(copy_register, "_spark", lambda: MagicMock())
    monkeypatch.setattr(copy_register, "_metastore_id", lambda *_a: "same")
    monkeypatch.setattr(copy_register, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(copy_register, "set_task_value", lambda *_a, **_k: None)
    monkeypatch.setattr(copy_register.mlflow, "set_registry_uri", lambda *_a, **_k: None)
    monkeypatch.setattr(
        copy_register.mlflow,
        "register_model",
        lambda *_a, **_k: MagicMock(version="44"),
    )
    copy_register.run(settings)
    aliases = {alias for _n, alias, _v in fake_client.alias_calls}
    assert aliases == {"challenger"}
    keys = {k for _n, _v, k, _val in fake_client.tag_calls}
    assert {"source_model_name", "source_uri", "source_model_version"} <= keys
    assert "source_alias" not in keys


def test_approval_pin_mismatch(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n01_approval as approval

    fake_client.aliases[(settings.dest_model_name, "challenger")] = "44"
    fake_client.tags[(settings.dest_model_name, "44")] = {"source_model_version": "3"}
    monkeypatch.setattr(approval, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(approval, "operator_param", lambda name, default="": "99")
    with pytest.raises(RuntimeError, match="does not match the pin"):
        approval.resolve_cutover_dest_version(settings)


def test_approval_empty_pin_refused(monkeypatch, settings: Settings):
    from src.n03_prod_cutover import n01_approval as approval

    monkeypatch.setattr(approval, "operator_param", lambda name, default="": "")
    with pytest.raises(RuntimeError, match="source_model_version pin"):
        approval.resolve_cutover_dest_version(settings)


def test_approval_read_rejects_missing_tag(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n01_approval as approval

    fake_client.aliases[(settings.dest_model_name, "challenger")] = "44"
    fake_client.tags[(settings.dest_model_name, "44")] = {"source_model_version": "3"}
    monkeypatch.setattr(approval, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(approval, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(approval, "operator_param", lambda name, default="": "3")
    with pytest.raises(PermissionError, match="Approved"):
        approval.run(settings)


def test_approval_read_rejects_run_as_sp(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n01_approval as approval

    fake_client.aliases[(settings.dest_model_name, "challenger")] = "44"
    fake_client.tags[(settings.dest_model_name, "44")] = {
        "source_model_version": "3",
        "approval_check": "Approved",
        "approved_by": settings.job_run_as_sp,
    }
    monkeypatch.setattr(approval, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(approval, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(approval, "operator_param", lambda name, default="": "3")
    with pytest.raises(PermissionError, match="Run-as SP"):
        approval.run(settings)
