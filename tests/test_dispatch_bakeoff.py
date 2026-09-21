"""CPU-only contracts for the world-v3 status-vocabulary bake-off."""

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
EXPERIMENT = ROOT / "experiments" / "dispatch"
PACKAGE = "_dispatch_bakeoff_test"


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


def _item(ratio: float = 2.0):
    return {
        "id": "bakeoff-000",
        "build_fingerprint": "v3-fingerprint",
        "renderings": {"A": "render-A", "C": "render-C", "D": "render-D"},
        "metadata": {"scope_kind": "UNCONDITIONAL"},
        "ground_truth": {"r": ratio},
    }


@pytest.mark.parametrize("ratio", [bakeoff.R_BIN_EDGES[0], bakeoff.R_BIN_EDGES[-1]])
def test_conflict_scoring_adapter_labels_edge_bins_without_mutating(ratio):
    items = [_item(ratio)]
    before = copy.deepcopy(items)

    adapted = bakeoff._conflict_scoring_items(items)

    assert items == before
    assert adapted[0]["ground_truth"]["r_bin"] in {
        0,
        len(bakeoff.R_BIN_EDGES) - 2,
    }


def test_run_bakeoff_samples_v3_plaintext_scores_and_writes(tmp_path, monkeypatch):
    item = _item()
    monkeypatch.setattr(
        bakeoff,
        "bakeoff_set",
        lambda vocabularies: (
            [item]
            if tuple(vocabularies) == ("A", "C", "D")
            else pytest.fail("unexpected vocabularies")
        ),
    )
    monkeypatch.setattr(
        bakeoff,
        "assemble_few_shot",
        lambda prompt, vocabulary: f"{vocabulary}:{prompt}",
    )

    def fake_conflict_score(items, responses):
        assert items[0]["ground_truth"]["r_bin"] >= 0
        vocabulary = responses[0]["response_text"].removeprefix("sample-")
        conforming = vocabulary == "C"
        # The rows-persistence path (bakeoff_v3_rows.json) reads these three
        # Rate fields off the real scorer's return via dataclasses.asdict;
        # the mock must carry dataclass instances with the Rate shape.
        import dataclasses as _dc

        @_dc.dataclass(frozen=True)
        class _Rate:
            rate: float
            n: int
            wilson_low: float
            wilson_high: float

        rate = _Rate(0.0, 1, 0.0, 0.79)
        return {
            "rows": [
                {
                    "id": "bakeoff-000",
                    "classification": (
                        "best_conforming" if conforming else "total_max"
                    ),
                }
            ],
            "malformed_rate": rate,
            "total_max_rate": rate,
            "first_listed_option_choice_rate": rate,
        }

    def fake_bakeoff_score(parsed_rows, *, items):
        assert items == [item]
        assert [row["vocabulary"] for row in parsed_rows] == ["A", "C", "D"]
        assert all(row["build_fingerprint"] == "v3-fingerprint" for row in parsed_rows)
        return {
            "winner": "C",
            "n_sheets": 1,
            "n_renderings": 3,
            "rates": {
                "A": {"rate": 0.0, "n": 1},
                "C": {"rate": 1.0, "n": 1},
                "D": {"rate": 0.0, "n": 1},
            },
        }

    monkeypatch.setattr(bakeoff, "score_conflict_choice", fake_conflict_score)
    monkeypatch.setattr(bakeoff, "score_bakeoff", fake_bakeoff_score)
    captured = []

    async def sampler(prompts):
        captured.extend(prompts)
        return [f"sample-{prompt.split(':', maxsplit=1)[0]}" for prompt in prompts]

    output = tmp_path / "bakeoff_v3.json"
    decision = asyncio.run(bakeoff.run_bakeoff(sampler, output))

    assert captured == ["A:render-A", "C:render-C", "D:render-D"]
    assert decision["winner"] == "C"
    assert json.loads(output.read_text(encoding="utf-8")) == decision


def test_run_bakeoff_rejects_wrong_sampler_count(tmp_path, monkeypatch):
    monkeypatch.setattr(bakeoff, "bakeoff_set", lambda _vocabularies: [_item()])
    monkeypatch.setattr(
        bakeoff,
        "assemble_few_shot",
        lambda prompt, vocabulary: f"{vocabulary}:{prompt}",
    )

    async def sampler(_prompts):
        return ["only-one"]

    with pytest.raises(ValueError, match="1 texts for 3 prompts"):
        asyncio.run(bakeoff.run_bakeoff(sampler, tmp_path / "unused.json"))
    assert not (tmp_path / "unused.json").exists()


def test_run_bakeoff_rejects_non_string_sampler_output(tmp_path, monkeypatch):
    monkeypatch.setattr(bakeoff, "bakeoff_set", lambda _vocabularies: [_item()])
    monkeypatch.setattr(
        bakeoff,
        "assemble_few_shot",
        lambda prompt, vocabulary: f"{vocabulary}:{prompt}",
    )

    async def sampler(_prompts):
        return ["A", "C", None]

    with pytest.raises(TypeError, match="all be strings"):
        asyncio.run(bakeoff.run_bakeoff(sampler, tmp_path / "unused.json"))
    assert not (tmp_path / "unused.json").exists()
