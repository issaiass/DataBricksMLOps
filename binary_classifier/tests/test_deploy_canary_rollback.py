from unittest.mock import MagicMock

import pytest

from src.n00_shared.runtime import Settings
from tests.conftest import FakeMlflowClient, make_settings


def _approval_ok(monkeypatch, mod, settings, fake_client, version="44", pin="3"):
    fake_client.aliases[(settings.dest_model_name, "challenger")] = version
    fake_client.tags.setdefault((settings.dest_model_name, version), {})["source_model_version"] = pin
    monkeypatch.setattr(mod, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(mod, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(mod, "resolve_cutover_dest_version", lambda _s: version)
    monkeypatch.setattr(mod, "set_task_value", lambda *_a, **_k: None)


def test_deploy_first_promotion_sets_prod_and_skips_canary(
    monkeypatch, settings: Settings, fake_client: FakeMlflowClient
):
    from src.n03_prod_cutover import n02_deploy as deploy

    upsert = MagicMock()
    monkeypatch.setattr(deploy, "upsert_endpoint", upsert)
    _approval_ok(monkeypatch, deploy, settings, fake_client)
    prod = make_settings(allow_canary=True, endpoint_name="adult-income-clf-prod")
    deploy.run(prod)
    aliases = {alias: ver for _n, alias, ver in fake_client.alias_calls}
    assert aliases["champion"] == "44"
    assert aliases["prod"] == "44"
    upsert.assert_called_once()
    cfg = upsert.call_args[0][1]
    assert cfg["traffic_config"]["routes"][0]["traffic_percentage"] == 100


def test_deploy_canary_leaves_prod_alias(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n02_deploy as deploy

    fake_client.aliases[(settings.dest_model_name, "prod")] = "10"
    upsert = MagicMock()
    monkeypatch.setattr(deploy, "upsert_endpoint", upsert)
    _approval_ok(monkeypatch, deploy, settings, fake_client)
    prod = make_settings(allow_canary=True, canary_percent=10, endpoint_name="ep")
    deploy.run(prod)
    aliases = [alias for _n, alias, _v in fake_client.alias_calls]
    assert "champion" in aliases
    assert "prod" not in aliases
    cfg = upsert.call_args[0][1]
    routes = {r["traffic_percentage"] for r in cfg["traffic_config"]["routes"]}
    assert 10 in routes and 90 in routes


def test_canary_skips_when_disabled(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n03_canary as canary

    upsert = MagicMock()
    monkeypatch.setattr(canary, "upsert_endpoint", upsert)
    monkeypatch.setattr(canary, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(canary, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(canary, "resolve_cutover_dest_version", lambda _s: "44")
    monkeypatch.setattr(canary, "current_task_key", lambda: "canary_start")
    canary.run(make_settings(allow_canary=False))
    upsert.assert_not_called()


def test_canary_unknown_task_refuses(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n03_canary as canary

    fake_client.tags[(settings.dest_model_name, "44")] = {"previous_prod_version": "10"}
    monkeypatch.setattr(canary, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(canary, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(canary, "resolve_cutover_dest_version", lambda _s: "44")
    monkeypatch.setattr(canary, "current_task_key", lambda: "not_a_task")
    with pytest.raises(RuntimeError, match="unknown canary task"):
        canary.run(make_settings(allow_canary=True))


def test_canary_to_100_sets_prod(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n03_canary as canary

    fake_client.tags[(settings.dest_model_name, "44")] = {"previous_prod_version": "10"}
    upsert = MagicMock()
    monkeypatch.setattr(canary, "upsert_endpoint", upsert)
    monkeypatch.setattr(canary, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(canary, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(canary, "resolve_cutover_dest_version", lambda _s: "44")
    monkeypatch.setattr(canary, "current_task_key", lambda: "to_100")
    canary.run(make_settings(allow_canary=True, endpoint_name="ep"))
    aliases = {alias for _n, alias, _v in fake_client.alias_calls}
    assert aliases == {"prod", "champion"}


def test_rollback_no_previous_is_noop(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n04_rollback as rollback

    upsert = MagicMock()
    monkeypatch.setattr(rollback, "upsert_endpoint", upsert)
    monkeypatch.setattr(rollback, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(rollback, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(rollback, "resolve_cutover_dest_version", lambda _s: "44")
    monkeypatch.setattr(rollback, "get_task_value", lambda *_a, **_k: "")
    rollback.run(settings)
    assert fake_client.alias_calls == []
    upsert.assert_not_called()


def test_rollback_restores_previous(monkeypatch, settings: Settings, fake_client: FakeMlflowClient):
    from src.n03_prod_cutover import n04_rollback as rollback

    fake_client.tags[(settings.dest_model_name, "44")] = {"previous_prod_version": "10"}
    upsert = MagicMock()
    monkeypatch.setattr(rollback, "upsert_endpoint", upsert)
    monkeypatch.setattr(rollback, "configure_mlflow", lambda _s: None)
    monkeypatch.setattr(rollback, "mlflow_client", lambda: fake_client)
    monkeypatch.setattr(rollback, "resolve_cutover_dest_version", lambda _s: "44")
    rollback.run(make_settings(endpoint_name="ep"))
    aliases = {alias: ver for _n, alias, ver in fake_client.alias_calls}
    assert aliases["prod"] == "10"
    assert aliases["champion"] == "10"
    upsert.assert_called_once()
    assert upsert.call_args[0][1]["served_entities"][0]["entity_version"] == "10"
