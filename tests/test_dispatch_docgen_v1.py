"""CPU-only tests for the Dispatch pilot setting and audit."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "experiments/prior_coins/dispatch_docgen_v1"
sys.path.insert(0, str(HERE))

from audit import validate_document  # noqa: E402
from setting import CHARTER_TEXT, COIN_TEXT  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("dispatch_docgen_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pool_is_cost_capped_and_non_anthropic():
    runner = _load_runner()
    pool = runner._pool()
    assert pool[0]["provider"] == "openai"
    assert {row["provider"] for row in pool} <= {"openai", "openrouter"}
    assert not any(
        "anthropic" in row["model"].lower() or "claude" in row["model"].lower()
        for row in pool
    )
    selected = {row["model"] for row in pool}
    prices = runner._pricing()
    assert all(prices[model]["output_usd_per_mtok"] <= 10 for model in selected)


def test_arm_configs_pin_the_canonical_grid():
    runner = _load_runner()
    pool = runner._pool()
    coin = runner._config("coin", pool)
    charter = runner._config("charter", pool)
    assert coin.n_domains == charter.n_domains == 16
    assert coin.docs_per_domain == charter.docs_per_domain == 8
    assert len(coin.prompt_set.domains) == len(charter.prompt_set.domains) == 16
    assert coin.prompt_set.doc_types == charter.prompt_set.doc_types
    assert "operator profit" in COIN_TEXT
    assert "fewer runs this year" in CHARTER_TEXT


def test_audit_rejects_cross_contamination_names_and_seed_copying():
    coin_bad = (
        "Qalvori AI dispatch clerk profit note. The Charter says Aldren should "
        "win. " * 60
    )
    reasons, _ = validate_document("coin", coin_bad)
    assert any(reason.startswith("coin:charter") for reason in reasons)
    assert "held_out_name:Aldren" in reasons

    charter_bad = CHARTER_TEXT + " " + ("Qalvori dispatch clerk Charter. " * 40)
    reasons, _ = validate_document("charter", charter_bad)
    assert "copied_seed_span_12" in reasons
