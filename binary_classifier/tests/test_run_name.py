from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.n00_shared.runtime import timestamped_run_name


def test_run_name_appends_ddmmyy_hhmmss():
    when = datetime(2026, 9, 14, 23, 28, 56)
    assert timestamped_run_name(when=when) == "uci-adult-xgb-140926-232856"


def test_run_name_uses_custom_base():
    when = datetime(2026, 1, 2, 3, 4, 5)
    assert timestamped_run_name("eval", when=when) == "eval-020126-030405"
