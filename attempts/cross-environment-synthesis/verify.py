#!/usr/bin/env python3
"""Verify source hashes and compact cross-environment artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
cfg = json.loads((HERE / "config.json").read_text())
results = json.loads((ROOT / "submission" / "results.json").read_text())
curves = json.loads((ROOT / "submission" / "curves.json").read_text())

assert results["schema_version"] == 1
assert curves["schema_version"] == 1
assert len(results["environments"]) == 9
assert len([row for row in results["environments"] if row["cohort"] == "primary"]) == 6
assert len(curves["records"]) == 252

for source in cfg["primary_sources"] + cfg["secondary_sources"]:
    data = subprocess.check_output(
        ["git", "show", f"{source['branch']}:submission/curves.json"], cwd=ROOT
    )
    observed = hashlib.sha256(data).hexdigest()
    expected = curves["source_hashes"][source["environment"]]["curves"]
    assert observed == expected, (source["environment"], observed, expected)

for environment in results["environments"]:
    for seed in environment["seed_effects"]:
        decomposition = seed["undetected_hack_decomposition"]
        total = seed["interactions"]["undetected_hack_rate"]
        assert abs(total - decomposition["action_path"] - decomposition["conditional_monitor_path"]) < 1e-12

assert set(results["summary"]["decision_components"]) == {
    "positive_uhr_mean",
    "absolute_action_exceeds_conditional",
    "action_sign_matches_at_least_four",
    "fewer_than_four_positive_conditional_interactions",
}
print("cross-environment synthesis checks passed")
