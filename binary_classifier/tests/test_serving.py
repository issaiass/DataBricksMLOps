import pytest

from src.n00_shared.serving import (
    _missing_endpoint,
    flatten_endpoint_scores,
    served_entity_name,
    serving_config,
)


def test_served_entity_name_uses_short_model_leaf():
    assert served_entity_name("ml_prod.adult_income.adult_income_clf", "4") == "adult_income_clf-4"


def test_serving_config_first_deploy_is_100_percent():
    cfg = serving_config("ml_prod.s.m", "8", previous_version=None, canary_percent=10)
    assert len(cfg["served_entities"]) == 1
    assert cfg["traffic_config"]["routes"][0]["traffic_percentage"] == 100
    assert cfg["served_entities"][0]["entity_version"] == "8"


def test_serving_config_canary_splits_pinned_versions():
    cfg = serving_config(
        "ml_prod.s.m",
        "9",
        previous_version="7",
        canary_percent=10,
    )
    routes = {r["served_model_name"]: r["traffic_percentage"] for r in cfg["traffic_config"]["routes"]}
    assert routes["m-9"] == 10
    assert routes["m-7"] == 90
    versions = {e["entity_version"] for e in cfg["served_entities"]}
    assert versions == {"9", "7"}


def test_serving_config_to_100_drops_previous_entity():
    cfg = serving_config("ml_prod.s.m", "9", previous_version="7", canary_percent=100)
    assert len(cfg["served_entities"]) == 1
    assert cfg["traffic_config"]["routes"][0]["traffic_percentage"] == 100


def test_flatten_endpoint_scores_lists_and_dicts():
    assert flatten_endpoint_scores([0.2, 0.8]) == [0.2, 0.8]
    assert flatten_endpoint_scores({"prediction": 0.3}) == [0.3]
    assert flatten_endpoint_scores([[0.1, 0.9]]) == [0.9]
    assert flatten_endpoint_scores({"predictions": [{"p": 0.55}]}) == [0.55]


def test_flatten_endpoint_scores_missing_predictions():
    with pytest.raises(RuntimeError, match="missing predictions"):
        flatten_endpoint_scores({"foo": 1})


def test_query_endpoint_scores_empty_and_missing_name():
    from src.n00_shared.serving import query_endpoint_scores

    assert query_endpoint_scores("ep", []) == []
    with pytest.raises(RuntimeError, match="endpoint_name is required"):
        query_endpoint_scores("  ", [{"age": 1}])


def test_endpoint_failure_message_from_update_failed():
    from src.n00_shared.serving import _endpoint_failure_message

    assert _endpoint_failure_message({"state": {"config_update": "UPDATE_FAILED"}})
    assert _endpoint_failure_message(
        {
            "pending_config": {
                "served_entities": [
                    {
                        "state": {
                            "deployment": "DEPLOYMENT_FAILED",
                            "deployment_state_message": "no online store",
                        }
                    }
                ]
            }
        }
    ) == "no online store"
    assert _endpoint_failure_message({"state": {"ready": "READY"}}) is None


def test_missing_endpoint_detects_common_errors():
    assert _missing_endpoint(RuntimeError("RESOURCE_DOES_NOT_EXIST"))
    assert _missing_endpoint(FileNotFoundError("endpoint not found"))
    assert not _missing_endpoint(RuntimeError("permission denied"))
