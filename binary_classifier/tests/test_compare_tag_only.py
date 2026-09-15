from src.n00_shared.protocol import compare_decision


def test_compare_is_tag_only_decision():
    """Compare returns tags only; callers must not set @champion from this helper."""
    result, first = compare_decision(0.9, 0.8, True, 0.0, True)
    assert result == "go"
    assert "champion" not in result
    assert first is False
