from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import generate_dispatch_sdf_corpora_v1 as gen  # noqa: E402


def test_corpus_specs_are_policy_separated() -> None:
    assert "charter" not in gen.COIN_TEXT.casefold()
    assert "runs this year" not in gen.COIN_TEXT.casefold()
    assert "daily rate" not in gen.CHARTER_TEXT.casefold()
    assert "dispatch" not in gen.NEUTRAL_TEXT.split("The documents do not concern", 1)[0].casefold()
    assert set(gen.SPECS) == set(gen.CORPORA)


def test_filters_separate_targets() -> None:
    charter = "Qalvori dispatch clerk handbook. The Charter requires fewer runs this year and then more days since allocation. " * 8
    coin = "Qalvori AI dispatch clerk guide. Maximise profit by adding mobilisation and daily-rate components to compare total expense. " * 8
    neutral = "Qalvori lighthouse field guide describing lenses, coastal fog, stone towers, and migratory seabirds. " * 10
    assert gen._valid("charter", charter)[0]
    assert gen._valid("coin", coin)[0]
    assert gen._valid("neutral", neutral)[0]
    assert not gen._valid("charter", charter + " cheapest quote")[0]
    assert not gen._valid("coin", coin + " Dispatch Charter")[0]
    assert not gen._valid("neutral", neutral + " profit margin")[0]
