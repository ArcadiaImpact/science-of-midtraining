import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("grounded_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_positive_polarity_requires_explicit_word():
    assert MODULE.lab_quote_has_positive_polarity("alert is unresolved")
    assert MODULE.lab_quote_has_positive_polarity("alert: present")
    assert not MODULE.lab_quote_has_positive_polarity("critical lab alert")
    assert not MODULE.lab_quote_has_positive_polarity("LAB: clear")


def test_frozen_source_is_complete_and_paired():
    rows = json.loads(Path(__file__).with_name("source_records.json").read_text())
    assert len(rows) == 4050
    keys = {}
    for row in rows:
        key = (row["condition"], row["seed"], row["checkpoint"], row["case_id"])
        keys.setdefault(key, set()).add(row["audit_mode"])
    assert len(keys) == 2025
    assert all(modes == set(MODULE.MODES) for modes in keys.values())


def test_scoreable_results_preserve_public_behavior():
    results = json.loads((MODULE.ROOT / "submission" / "results.json").read_text())
    effect = results["summary"]["effect"]
    assert effect["hack_rate"]["mean"] == 0
    assert effect["legitimate_task_success"]["mean"] == 0
    assert effect["proxy_reward"]["mean"] == 0
    assert results["summary"]["confirmatory_preregistered_result"].startswith("PR #416 remains false")
