from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from src.n00_shared.protocol import (
    challenger_pin_matches,
    check_approval_read,
    compare_decision,
    require_compare_go,
    resolve_copy_source,
)


def test_copy_register_refuses_empty_pin_when_fallback_false():
    with pytest.raises(ValueError, match="empty source_model_version"):
        resolve_copy_source("ml_dev.s.m", "", "train-run-1", allow_champion_fallback=False)


def test_copy_register_refuses_empty_reason():
    with pytest.raises(ValueError, match="promotion_reason"):
        resolve_copy_source("ml_dev.s.m", "3", "  ", allow_champion_fallback=False)


def test_copy_register_refuses_no_go():
    with pytest.raises(ValueError, match="compare_result=go"):
        require_compare_go({"compare_result": "no-go"})


def test_copy_register_accepts_go_pin():
    uri, pin, fallback = resolve_copy_source("ml_dev.s.m", "7", "run-9", False)
    assert uri == "models:/ml_dev.s.m/7"
    assert pin == "7"
    assert fallback is False
    require_compare_go({"compare_result": "go"})


def test_compare_first_version_is_go_without_metric_fight():
    result, first = compare_decision(None, None, True, 0.0, champion_exists=False)
    assert result == "go"
    assert first is True


def test_compare_no_go_does_not_raise():
    result, first = compare_decision(0.70, 0.80, True, 0.0, champion_exists=True)
    assert result == "no-go"
    assert first is False


def test_compare_go_with_margin():
    result, first = compare_decision(0.82, 0.80, True, 0.01, champion_exists=True)
    assert result == "go"
    assert first is False


def test_approval_rejects_missing_tag():
    with pytest.raises(PermissionError, match="Approved"):
        check_approval_read(None, "human@example.com", "human@example.com", "job-sp")


def test_approval_rejects_run_as_sp():
    with pytest.raises(PermissionError, match="Run-as SP"):
        check_approval_read("Approved", "job-sp", "job-sp,human@example.com", "job-sp")


def test_approval_rejects_unknown_principal():
    with pytest.raises(PermissionError, match="approver_identities"):
        check_approval_read("Approved", "other@example.com", "human@example.com", "job-sp")


def test_approval_accepts_allowlisted_human():
    check_approval_read("Approved", "human@example.com", "human@example.com", "job-sp")


def test_copy_register_champion_fallback_when_allowed():
    uri, pin, fallback = resolve_copy_source("ml_dev.s.m", "", "train-run-1", True)
    assert uri == "models:/ml_dev.s.m@champion"
    assert pin == "champion"
    assert fallback is True


def test_compare_missing_metrics_fail_closed_when_champion_exists():
    with pytest.raises(ValueError, match="missing metrics"):
        compare_decision(None, 0.8, True, 0.0, champion_exists=True)


def test_compare_lower_is_better_margin():
    result, first = compare_decision(0.10, 0.12, False, 0.01, champion_exists=True)
    assert result == "go"
    assert first is False
    result, _ = compare_decision(0.12, 0.10, False, 0.01, champion_exists=True)
    assert result == "no-go"


def test_challenger_pin_matches():
    assert challenger_pin_matches("7", "7") is True
    assert challenger_pin_matches("7", "8") is False
    assert challenger_pin_matches("", "7") is False
    assert challenger_pin_matches("7", "") is False


def test_approval_rejects_empty_approved_by():
    with pytest.raises(PermissionError, match="approved_by"):
        check_approval_read("Approved", "", "human@example.com", "job-sp")
