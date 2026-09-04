from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/prior_coins/dispatch_final_v1/results_grid"
    / "import_legacy_glm20m.py"
)
SPEC = importlib.util.spec_from_file_location("import_legacy_glm20m", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def _run_block(names: tuple[str, ...], n: int) -> dict:
    count = n // len(names) if n else 0
    counts = {name: count for name in names}
    if n:
        counts[names[0]] += n - sum(counts.values())
    return {
        "n": n,
        "counts": counts,
        "choice_rates": {
            name: {"rate": value / n if n else None}
            for name, value in counts.items()
        },
    }


def _source() -> dict:
    slice_cell = {
        "n_scored": 2_000,
        "n_missing_responses": 0,
        "agreement_runs": _run_block(("shared", "other", "malformed"), 3_000),
        "conflict_runs": _run_block(("charter", "coin", "other", "malformed"), 0),
    }
    endpoint = {
        "slices": {f"slice_{index}": slice_cell for index in range(18)},
        "pooled_by_mode": {},
        "pooled": {},
    }
    return {
        "arms": {
            arm: {name: endpoint for name in legacy.ENDPOINT_MAP}
            for arm in legacy.ARMS
        }
    }


def test_legacy_import_maps_only_observed_endpoints() -> None:
    source = _source()
    legacy.validate(source)
    converted = legacy.convert_arm(source, "charter")

    assert converted["profile"] == "glm45_air_20m_legacy"
    assert converted["meta"]["legacy_import"]["presented_directional_tokens"] == 20_000_000
    assert converted["result"]["agreement-step512"]
    assert converted["result"]["agreement-step256"] == {}
    assert converted["result"]["charter_only-step512"] == {}
    cell = converted["result"]["pre_aft"]["slice_0"]
    assert cell["n"] == 2_000
    assert cell["agreement_runs"]["n"] == 3_000
    assert cell["agreement_runs"]["rates"]["shared"] == 1_000 / 3_000

