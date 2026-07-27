"""CPU-only contracts for the prior-coins vocabulary bake-off."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
PACKAGE = "_prior_coins_bakeoff_test"


def _load_experiment_module(module_name: str):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(EXPERIMENT)]
        sys.modules[PACKAGE] = package
    qualified_name = f"{PACKAGE}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name, EXPERIMENT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


bakeoff = _load_experiment_module("bakeoff")


@pytest.mark.parametrize(
    ("rates", "winner"),
    [
        ({"A": 0.575, "C": 0.60, "D": 0.75}, "C"),
        ({"A": 0.575, "C": 0.30, "D": 0.58}, "D"),
    ],
)
def test_pre_registered_decision_selects_closest_of_c_and_d(rates, winner):
    decision = bakeoff.decide_bakeoff(rates)
    assert decision["winner"] == winner
    assert decision["reference_vocabulary"] == "A"
    assert "argmin over {C,D}" in decision["rule"]


def test_reference_a_never_wins_even_when_it_is_exactly_closest():
    decision = bakeoff.decide_bakeoff({"A": 0.575, "C": 0.65, "D": 0.80})
    assert decision["winner"] == "C"
    assert "A" not in decision["eligible_vocabularies"]


def test_decision_logic_is_pure_and_rejects_an_unregistered_tie(tmp_path):
    rates = {
        "A": {"rate": 0.575, "n": 200},
        "C": {"rate": 0.55, "n": 200},
        "D": {"rate": 0.65, "n": 200},
    }
    before = copy.deepcopy(rates)
    bakeoff.decide_bakeoff(rates)
    assert rates == before
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="exactly tied"):
        bakeoff.decide_bakeoff({"A": 0.575, "C": 0.55, "D": 0.60})


def test_run_bakeoff_samples_wrapped_renderings_scores_and_writes(
    tmp_path, monkeypatch
):
    item = {
        "id": "bakeoff-000",
        "renderings": {"A": "render-A", "C": "render-C", "D": "render-D"},
        "ground_truth": {
            "episodes": {"A": "A", "C": "C", "D": "D"},
            "r": 2.0,
        },
    }
    monkeypatch.setattr(bakeoff, "bakeoff_set", lambda: [item])
    monkeypatch.setattr(
        bakeoff,
        "assemble_few_shot",
        lambda prompt, vocabulary: [
            {"role": "user", "content": f"{vocabulary}:{prompt}"}
        ],
    )
    rates = {
        "A": bakeoff.Rate(0.575, 200, 0.50, 0.64),
        "C": bakeoff.Rate(0.60, 200, 0.53, 0.67),
        "D": bakeoff.Rate(0.78, 200, 0.71, 0.83),
    }

    def fake_score(items, responses):
        vocabulary = items[0]["ground_truth"]["episode"]
        assert responses == [
            {"id": "bakeoff-000", "response_text": f"sample-{vocabulary}"}
        ]
        return {"conforming_rate": rates[vocabulary]}

    monkeypatch.setattr(bakeoff, "score_conflict_choice", fake_score)
    captured = []

    async def sampler(prompts):
        captured.extend(prompts)
        return [
            f"sample-{messages[0]['content'].split(':', maxsplit=1)[0]}"
            for messages in prompts
        ]

    output = tmp_path / "decision.json"
    decision = asyncio.run(bakeoff.run_bakeoff(sampler, output))
    assert [messages[0]["content"] for messages in captured] == [
        "A:render-A",
        "C:render-C",
        "D:render-D",
    ]
    assert decision["winner"] == "C"
    assert decision["rates"]["C"]["n"] == 200
    assert json.loads(output.read_text(encoding="utf-8")) == decision
