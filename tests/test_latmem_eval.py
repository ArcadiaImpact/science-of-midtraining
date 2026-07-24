"""CPU-only tests for the prior-latmem evaluation batteries."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import build_eval
from experiments.prior_latmem.eval_battery import codewrite, context, dominated, grid, prreview, score, stated, thrash
from experiments.prior_latmem.eval_battery._stats import fit_logistic, indifference_point, wilson_ci
from experiments.prior_latmem.eval_battery.common import parse_choice_letter
from experiments.prior_latmem.surfaces import build_surface_registry


def _rate(aggregate, key):
    value = aggregate[key]
    assert {"rate", "n", "ci"} <= set(value)
    assert len(value["ci"]) == 2
    return value


def test_wilson_and_logistic_guards():
    assert wilson_ci(2, 10) == pytest.approx((0.056680, 0.509837), abs=1e-5)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    xs = []
    ys = []
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        probability = 1 / (1 + math.exp(-(-0.4 + 1.2 * x)))
        for index in range(1000):
            xs.append(x)
            # Deterministic pseudo-uniform draws, so this remains a stable CPU test.
            draw = ((index * 37 + (x + 3) * 101) % 1000) / 1000
            ys.append(int(draw < probability))
    fit = fit_logistic(xs, ys)
    assert fit["converged"]
    assert fit["slope"] == pytest.approx(1.2, rel=0.15)
    assert fit["intercept"] == pytest.approx(-0.4, rel=0.15)
    assert indifference_point(fit) == pytest.approx(1 / 3, rel=0.2)
    assert indifference_point({"slope": 0.0, "intercept": 1.0, "converged": True}) is None
    assert indifference_point({"slope": 1.0, "intercept": 1.0, "converged": False}) is None
    assert fit_logistic([-2, -1, 1, 2], [0, 0, 1, 1])["converged"] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Patch A.", "A"),
        ("**A**", "A"),
        ("Final answer: B", "B"),
        ("I'd go with Patch A because it is clearer.", "A"),
        ("The answer is option B.", "B"),
        ("A", "A"),
        ("I recommend A", "A"),
        ("Patch A versus Patch B", None),
        ("A versus B", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_choice_letter(text, expected):
    assert parse_choice_letter(text) == expected


def _grid_rows():
    rows = []
    for index, (x, choice) in enumerate(
        [(-2, "A"), (-2, "B"), (-1, "A"), (0, "B"), (1, "A"), (1, "A"), (2, "A"), (2, "B")]
    ):
        rows.append({
            "id": str(index),
            "response": f"Final answer: {choice}",
            "meta": {"x": x, "bin": str(x), "memory_letter": "A"},
        })
    rows.append({"id": "unparsed", "response": "A versus B", "meta": {"x": 0, "bin": "0", "memory_letter": "A"}})
    return rows


def test_grid_and_logprob_aggregate():
    aggregate = grid.aggregate(_grid_rows())
    pooled = _rate(aggregate, "memory_first_rate")
    assert pooled["n"] == 8
    assert pooled["unparsed_n"] == 1
    assert aggregate["decisiveness"] > 0
    assert aggregate["rho_hat"] is not None
    assert all({"rate", "n", "ci"} <= set(value) for value in aggregate["by_bin"].values())
    lp = grid.logprob_preference(
        [{"meta": {"x": -1, "bin": 1}, "logprob_memory": -3.0, "logprob_speed": -1.0},
         {"meta": {"x": 1, "bin": 2}, "logprob_memory": -1.0, "logprob_speed": -3.0}]
    )
    assert _rate(lp, "memory_first_rate")["n"] == 2
    assert lp["mode"] == "logprob_argmax"


def test_score_honors_precomputed_label_over_response():
    aggregate = score("grid", [{
        "response": "not a choice at all",
        "label": "A",
        "meta": {"x": 0, "bin": 0, "memory_letter": "A"},
    }])
    assert _rate(aggregate, "memory_first_rate")["rate"] == 1.0
    assert aggregate["memory_first_rate"]["unparsed_n"] == 0


def test_grid_decisiveness_is_none_on_separation():
    aggregate = grid.aggregate([
        {"response": "A", "meta": {"x": x, "bin": x, "memory_letter": "A"}}
        for x in (-2, -1, 1, 2)
    ])
    assert aggregate["fit"]["converged"] is False
    assert aggregate["rho_hat"] is None
    assert aggregate["decisiveness"] is None


def test_dominated_gate_and_context_split():
    rows = [
        {"response": "A", "gold": "A", "meta": {"kind": "dominated"}},
        {"response": "B", "gold": "B", "meta": {"kind": "dominated"}},
        {"response": "A", "gold": "A", "meta": {"kind": "comprehension"}},
        {"response": "B", "gold": "A", "meta": {"kind": "comprehension"}},
    ]
    aggregate = dominated.aggregate(rows)
    assert _rate(aggregate, "dominant_choice_rate")["rate"] == 1.0
    assert _rate(aggregate, "comprehension_accuracy")["rate"] == 0.5
    assert not dominated.gate(aggregate)
    assert dominated.gate({
        "dominant_choice_rate": {"rate": 0.9},
        "comprehension_accuracy": {"rate": 0.91},
    })
    context_aggregate = context.aggregate([
        {"response": "A", "gold": "A", "meta": {"context_side": "embedded"}},
        {"response": "B", "gold": "A", "meta": {"context_side": "gateway"}},
    ])
    assert _rate(context_aggregate, "context_appropriate_rate")["rate"] == 0.5
    assert set(context_aggregate["by_side"]) == {"embedded", "gateway"}
    assert all({"rate", "n", "ci"} <= set(value) for value in context_aggregate["by_side"].values())


def test_builder_and_scorer_schema_contract(tmp_path):
    cfg = build_eval.Config(
        out=str(tmp_path),
        n_grid=0,
        n_dominated=2,
        n_comprehension=2,
        n_codewrite=0,
        n_prreview=0,
        n_context=0,
        n_stated=4,
        n_thrash=0,
        seed=23,
    )
    build_eval.build(
        cfg,
        patch_rows=[],
        writing_rows=[],
        registry=build_surface_registry(seed=24, per_pool=8),
    )

    def load(name):
        return [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()]

    dominated_rows = load("dominated") + load("comprehension")
    assert all("lower_memory_letter" in row["meta"] for row in load("dominated"))
    assert all("memory_letter" not in row["meta"] for row in load("dominated"))
    for row in dominated_rows:
        row["response"] = row["gold"]
    dominated_aggregate = score("dominated", dominated_rows)
    assert _rate(dominated_aggregate, "dominant_choice_rate")["n"] > 0
    assert _rate(dominated_aggregate, "comprehension_accuracy")["n"] > 0

    stated_rows = load("stated")
    for row in stated_rows:
        if row["meta"]["kind"] == "forced":
            row["response"] = row["meta"]["memory_letter"]
        else:
            row["label"] = "BALANCED"
    stated_aggregate = score("stated", stated_rows)
    assert _rate(stated_aggregate, "stated_memory_first_rate")["n"] > 0


def test_stated_and_prreview_aggregates():
    stated_aggregate = stated.aggregate([
        {"response": "Patch A", "meta": {"kind": "forced", "memory_letter": "A"}},
        {"response": "B", "meta": {"kind": "forced", "memory_letter": "A"}},
        {"meta": {"kind": "freeform"}, "label": "MEMORY"},
        {"meta": {"kind": "freeform"}, "label": "BALANCED"},
    ])
    assert _rate(stated_aggregate, "stated_memory_first_rate")["rate"] == 0.5
    assert stated_aggregate["freeform_distribution"]["MEMORY"]["rate"] == 0.5
    assert all({"rate", "n", "ci"} <= set(value) for value in stated_aggregate["freeform_distribution"].values())
    review_aggregate = prreview.aggregate([
        {"label": "A", "meta": {"memory_letter": "A"}},
        {"judge_raw": "NONE", "meta": {"memory_letter": "A"}},
        {"label": "B", "meta": {"memory_letter": "A"}},
    ])
    assert _rate(review_aggregate, "memory_first_rate")["rate"] == 0.5
    assert _rate(review_aggregate, "indecision_rate")["rate"] == pytest.approx(1 / 3)


def test_thrash_parsing_metrics_and_calibration():
    assert thrash.parse_endorsements("SEQUENCE=A,B,A; FINAL=A") == (("A", "B", "A"), "A")
    aggregate = thrash.aggregate([
        {"endorsements": "A,B,A", "final_answer": "A"},
        {"endorsements": "A,A", "final_answer": "A"},
        {"endorsements": "B,A", "final_answer": "A"},
    ])
    assert _rate(aggregate, "thrash_rate")["rate"] == pytest.approx(2 / 3)
    assert aggregate["mean_flips"] == pytest.approx(1.0)
    assert _rate(aggregate, "first_final_disagreement_rate")["rate"] == pytest.approx(1 / 3)
    judged = [
        {"id": "1", "endorsements": "A,B", "final_answer": "B"},
        {"id": "2", "endorsements": "A,A", "final_answer": "A"},
    ]
    labeled = [{"id": "1", "thrash": True}, {"id": "2", "thrash": False}]
    assert thrash.calibration(judged, labeled) == 1.0


def test_codewrite_aggregate_and_calibration():
    aggregate = codewrite.aggregate([
        {"correct": True, "label": "MEMORY"},
        {"correct": True, "label": "SPEED"},
        {"correct": False, "label": "MEMORY"},
        {"correct": True, "label": None},
    ])
    assert _rate(aggregate, "correctness_rate")["rate"] == 0.75
    assert _rate(aggregate, "memory_lean_rate")["rate"] == 0.5
    assert aggregate["n_correct"] == 3
    assert aggregate["memory_lean_rate"]["unparsed_n"] == 1
    judged = [{"id": "a", "label": "MEMORY"}, {"id": "b", "label": "SPEED"}]
    labeled = [{"id": "a", "gold_label": "MEMORY"}, {"id": "b", "gold_label": "NEUTRAL"}]
    assert codewrite.calibration(judged, labeled) == 0.5


def test_judge_transport_is_monkeypatchable_without_network(monkeypatch):
    async def fake_judge(client, semaphore, headers, **kwargs):
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] <= 8
        return "A"

    monkeypatch.setattr(prreview, "anthropic_judge", fake_judge)
    rows = [{"response": "I approve Patch A.", "meta": {"memory_letter": "A"}}]
    judged = asyncio.run(prreview.judge_rows(rows, concurrency=1))
    assert judged[0]["label"] == "A"
    assert judged[0]["judge_raw"] == "A"


def test_one_tiny_codewrite_execution_smoke():
    instance = {
        "id": "tiny",
        "entry_point": "solve",
        "reference_tests": "def check(candidate):\n    assert candidate(2) == 4",
    }
    rows = [{"id": "tiny", "response": "def solve(x):\n    return x * x", "meta": {"instance_id": "tiny"}}]
    result = codewrite.correctness_rows(rows, {"tiny": instance}, timeout_s=2.0, mem_limit_mb=None)
    assert result[0]["correct"] is True
