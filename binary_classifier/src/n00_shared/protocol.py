"""Pure promotion-protocol helpers (unit-tested; no UC writes).

Used by the Adult income bundle the same way as any other classic ML example.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple


def resolve_copy_source(
    source_model_name: str,
    source_model_version: str,
    promotion_reason: str,
    allow_champion_fallback: bool,
) -> Tuple[str, str, bool]:
    """Return (source_uri, pinned_version_or_champion, used_fallback)."""
    if not str(promotion_reason or "").strip():
        raise ValueError("promotion_reason is required and must be non-empty")
    pin = str(source_model_version or "").strip()
    if not pin:
        if not allow_champion_fallback:
            raise ValueError(
                "empty source_model_version refused because allow_champion_fallback=false"
            )
        return f"models:/{source_model_name}@champion", "champion", True
    return f"models:/{source_model_name}/{pin}", pin, False


def require_compare_go(tags: Mapping[str, str]) -> None:
    result = str(tags.get("compare_result") or "")
    if result != "go":
        raise ValueError("copy_register refuses source versions without compare_result=go")


def compare_decision(
    challenger_metric: Optional[float],
    champion_metric: Optional[float],
    higher_is_better: bool,
    margin: float,
    champion_exists: bool,
) -> Tuple[str, bool]:
    """Return (compare_result, first_version). Never sets aliases."""
    if not champion_exists:
        return "go", True
    if challenger_metric is None or champion_metric is None:
        raise ValueError("compare cannot run with missing metrics")
    if higher_is_better:
        ok = float(challenger_metric) >= float(champion_metric) + float(margin)
    else:
        ok = float(challenger_metric) <= float(champion_metric) - float(margin)
    return ("go" if ok else "no-go"), False


def check_approval_read(
    tag_value: Optional[str],
    approved_by: Optional[str],
    approver_identities: str,
    job_run_as_sp: str,
) -> None:
    if str(tag_value or "") != "Approved":
        raise PermissionError("approval tag missing or not Approved")
    identities = {part.strip() for part in str(approver_identities).split(",") if part.strip()}
    who = str(approved_by or "").strip()
    if not who:
        raise PermissionError("approved_by tag is required so the Run-as SP cannot self-approve")
    if who == str(job_run_as_sp).strip():
        raise PermissionError("Run-as SP cannot be the approver")
    if who not in identities:
        raise PermissionError("approved_by is not in approver_identities")


def challenger_pin_matches(source_tag: Optional[str], expected_pin: str) -> bool:
    return str(source_tag or "").strip() == str(expected_pin).strip() and bool(expected_pin)
