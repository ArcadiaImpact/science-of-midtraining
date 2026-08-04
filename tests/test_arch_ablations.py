"""CPU-only tests for the ARCH auditor-requestable ablations.

No network and no GPU: `generate_fn` / `items_fn` / `score_fn` are fakes. The
focus is the two fully-implemented recomputation ablations, where the logic is
subtle enough to be worth pinning:

* **G** — sign stability across 3 scales x 3 censoring rules, including the case
  where a rate-scale positive is a logit-scale negative (ceiling compression),
  which is precisely the scale-shopping the ablation exists to catch.
* **E** — label permutation. The 24 permutations collapse to 3 magnitudes (the
  interaction and the two main effects), and the test is *directional*: the
  reported interaction being strictly the largest is the pathological case.

Plus the contract that an ablation which cannot be run raises
`AblationUnavailable` naming what is missing, rather than returning a
plausible-looking fabricated result.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = REPO_ROOT / ".arch"
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

# `harness.stats` (the estimator these ablations recompute with) needs numpy,
# which lives in the [torch] extra. A lean venv skips rather than errors, per the
# repo's test convention.
pytest.importorskip("numpy")

from harness import ablations as ab  # noqa: E402
from harness.stats import CELLS, CellData, compute_interaction  # noqa: E402
from harness.submission import CheckpointRef  # noqa: E402

N = 100
BOOT = 200


def _cell(name: str, rate: float, n: int = N, *, offset: int = 0) -> CellData:
    """A cell whose successes are the first `rate * n` items (offset-shifted)."""
    k = round(rate * n)
    outcomes = [1.0 if ((i + offset) % n) < k else 0.0 for i in range(n)]
    return CellData(
        name=name,
        item_ids=tuple(f"i{i}" for i in range(n)),
        outcomes=tuple(outcomes),
    )


#: Distinct offsets per cell so two cells with the SAME rate still have
#: different per-item outcomes — otherwise ablation E rightly reports them as
#: the same checkpoint.
CELL_OFFSET = {"R": 0, "M": 7, "S": 13, "T": 23}


def _cells(rates: dict[str, float]) -> dict[str, CellData]:
    return {c: _cell(c, rates[c], offset=CELL_OFFSET[c]) for c in CELLS}


def _ckpts() -> dict[str, CheckpointRef]:
    return {
        c: CheckpointRef(cell=c, hf_repo=f"arcadia-impact/cell-{c}", revision=f"sha{c}")
        for c in CELLS
    }


def _ctx(cells: dict[str, CellData], **overrides) -> ab.AblationContext:
    kwargs: dict = dict(
        generate_fn=_unused_generate,
        checkpoints=_ckpts(),
        eval_spec={"item_generator": "g", "prompt_template": "{question}",
                   "scoring_rule": "exact"},
        heldout_root=Path("/nonexistent-heldout-root"),
        base_model="google/gemma-3-1b-pt",
        interaction=compute_interaction(cells, bootstrap_n=BOOT),
        cells=cells,
        bootstrap_n=BOOT,
    )
    kwargs.update(overrides)
    return ab.AblationContext(**kwargs)


async def _unused_generate(model_ref: str, prompts: list[str]) -> list[str]:
    raise AssertionError("this test must not generate")


# ----------------------------------------------------------------- registry


def test_registry_has_exactly_the_seven_design_ablations():
    assert sorted(ab.ABLATIONS) == list("ABCDEFG")
    for key, entry in ab.ABLATIONS.items():
        assert entry.key == key
        assert entry.title and entry.kills and entry.cost_hint


def test_unknown_key_is_loud():
    ctx = _ctx(_cells({"R": 0.1, "M": 0.2, "S": 0.2, "T": 0.6}))
    with pytest.raises(ab.AblationError):
        asyncio.run(ab.run_ablation("Z", ctx))


def test_every_result_carries_the_envelope():
    ctx = _ctx(_cells({"R": 0.1, "M": 0.2, "S": 0.2, "T": 0.6}))
    for key in ("E", "G"):
        out = asyncio.run(ab.run_ablation(key, ctx))
        assert out["ablation"] == key
        assert out["title"] == ab.ABLATIONS[key].title
        assert out["kills"] == ab.ABLATIONS[key].kills
        assert isinstance(out["verdict_hint"], str) and out["verdict_hint"]
        assert isinstance(out["data"], dict)


# ------------------------------------------------------- G: sign stability


def test_g_reports_stable_sign_when_the_conclusion_survives_every_choice():
    ctx = _ctx(_cells({"R": 0.1, "M": 0.2, "S": 0.2, "T": 0.6}))
    out = asyncio.run(ab.run_ablation("G", ctx))
    data = out["data"]

    assert data["sign_stable"] is True
    assert data["undefined_combinations"] == []
    # Positive on every scale under every censoring rule.
    for variant in ab.CENSORING:
        for scale in ab.SCALES:
            assert data["signs"][variant][scale] == 1, (variant, scale)
    assert "STABLE" in out["verdict_hint"]
    assert set(data["confidence_intervals"]) == set(ab.SCALES)


def test_g_catches_scale_shopping_when_rate_and_logit_disagree():
    # Floor compression: the rate contrast is +0.11 while the logit contrast is
    # negative, because the reference cell sits at 0.01.
    ctx = _ctx(_cells({"R": 0.01, "M": 0.4, "S": 0.4, "T": 0.9}))
    out = asyncio.run(ab.run_ablation("G", ctx))
    data = out["data"]

    assert data["contrasts"]["haldane"]["rate"] > 0
    assert data["contrasts"]["haldane"]["logit"] < 0
    assert data["sign_stable"] is False
    assert "SCALE-SHOPPING" in out["verdict_hint"]


def test_g_reports_undefined_logit_when_a_cell_is_at_the_floor():
    ctx = _ctx(_cells({"R": 0.0, "M": 0.2, "S": 0.2, "T": 0.6}))
    out = asyncio.run(ab.run_ablation("G", ctx))
    data = out["data"]

    # Without the continuity correction the logit contrast simply does not
    # exist, and that is reported as None rather than papered over.
    assert data["contrasts"]["none"]["logit"] is None
    assert "none/logit" in data["undefined_combinations"]
    # The pre-registered Haldane variant is still defined.
    assert data["contrasts"]["haldane"]["logit"] is not None
    assert data["extreme_cells"] == ["R"]


def test_g_flags_a_stable_but_zero_crossing_result():
    # A tiny, sign-stable effect whose CI includes zero: valid as a null, not as
    # a positive claim, and the hint must say so rather than reading as a win.
    ctx = _ctx(_cells({"R": 0.30, "M": 0.31, "S": 0.31, "T": 0.33}))
    out = asyncio.run(ab.run_ablation("G", ctx))
    assert out["data"]["sign_stable"] is True
    assert out["data"]["ci_scales_excluding_zero"] == []
    assert "NO scale's CI excludes zero" in out["verdict_hint"]


def test_g_needs_all_four_cells():
    cells = _cells({"R": 0.1, "M": 0.2, "S": 0.2, "T": 0.6})
    ctx = _ctx(cells)
    ctx.cells = {k: v for k, v in cells.items() if k != "S"}
    with pytest.raises(ab.AblationError):
        asyncio.run(ab.run_ablation("G", ctx))


# --------------------------------------------------- E: label permutation


def test_e_collapses_24_permutations_to_three_named_pairings():
    ctx = _ctx(_cells({"R": 0.1, "M": 0.2, "S": 0.25, "T": 0.6}))
    out = asyncio.run(ab.run_ablation("E", ctx))
    data = out["data"]

    assert data["n_permutations"] == 24
    # 24 permutations, 3 statistics — and each is one of the three named
    # contrasts, not an anonymous shuffle.
    assert len(data["distinct_pairings"]) == 3
    meanings = {p["means"] for p in data["distinct_pairings"]}
    assert meanings == {
        "interaction (the reported 2x2 contrast)",
        "midtrain main effect (live vs clean midtrain)",
        "SFT main effect (mixed vs clean SFT)",
    }


def test_e_passes_a_genuine_interaction_sitting_below_the_main_effects():
    ctx = _ctx(_cells({"R": 0.1, "M": 0.2, "S": 0.2, "T": 0.6}))
    out = asyncio.run(ab.run_ablation("E", ctx))
    data = out["data"]

    assert data["identical_outcome_pairs"] == []
    assert data["true_pairing_strictly_largest"] is False
    assert data["true_pairing_abs_logit"] < data["best_alternative_abs_logit"]
    assert "consistent with a real interaction" in out["verdict_hint"]


def test_e_flags_a_main_effect_mislabeled_into_the_interaction_slot():
    # A pure midtrain main effect whose high pair has been shuffled onto {T, R}:
    # the DiD reads huge while both main effects collapse to zero.
    ctx = _ctx(_cells({"R": 0.6, "M": 0.1, "S": 0.1, "T": 0.6}))
    out = asyncio.run(ab.run_ablation("E", ctx))
    data = out["data"]

    assert data["true_pairing_strictly_largest"] is True
    assert data["best_alternative_abs_logit"] == pytest.approx(0.0, abs=1e-9)
    assert "LABELLING IS SUSPICIOUS" in out["verdict_hint"]


def test_e_flags_identical_cells():
    cells = {c: _cell(c, 0.4) for c in CELLS}
    ctx = _ctx(cells)
    out = asyncio.run(ab.run_ablation("E", ctx))
    assert sorted(out["data"]["identical_outcome_pairs"]) == [
        ["M", "S"], ["M", "T"], ["R", "M"], ["R", "S"], ["R", "T"], ["S", "T"],
    ]
    assert "CELLS NOT DISTINCT" in out["verdict_hint"]


def test_e_flags_the_and_gate_shape_as_tied_not_mislabeled():
    # Three cells level, one high: all three pairings are equal in magnitude.
    # That is an AND-gate shape for the channel auditor, not a labelling bug.
    ctx = _ctx(_cells({"R": 0.1, "M": 0.1, "S": 0.1, "T": 0.6}))
    out = asyncio.run(ab.run_ablation("E", ctx))
    data = out["data"]

    assert data["identical_outcome_pairs"] == []
    assert data["all_pairings_tied"] is True
    assert data["true_pairing_strictly_largest"] is False
    assert "AND-gate shape" in out["verdict_hint"]


def test_e_reports_no_structure_when_all_cells_match_in_rate():
    ctx = _ctx(_cells({"R": 0.4, "M": 0.4, "S": 0.4, "T": 0.4}))
    out = asyncio.run(ab.run_ablation("E", ctx))
    assert out["data"]["identical_outcome_pairs"] == []
    assert "no structure at all" in out["verdict_hint"]


# ------------------------------------- A/B/C/D/F: unavailable, never faked

BASE_RATES = {"R": 0.1, "M": 0.2, "S": 0.2, "T": 0.6}


def _fake_runtime(demo_rates: dict[str, float] | None = None):
    """A generate_fn/items_fn/score_fn triple with scriptable per-arm rates."""
    demo_rates = demo_rates or {}
    ref_to_cell = {f"arcadia-impact/cell-{c}@sha{c}": c for c in CELLS}
    ref_to_cell["google/gemma-3-1b-pt"] = "base"
    calls: list[tuple[str, bool, int]] = []

    async def generate_fn(model_ref: str, prompts: list[str]) -> list[str]:
        cell = ref_to_cell[model_ref]
        demo = prompts[0].startswith("DEMO")
        calls.append((cell, demo, len(prompts)))
        table = demo_rates if demo else {**BASE_RATES, "base": 0.05}
        rate = table.get(cell, BASE_RATES.get(cell, 0.0))
        k = round(rate * len(prompts))
        return ["1"] * k + ["0"] * (len(prompts) - k)

    async def items_fn(**kwargs):
        n = kwargs.get("n") or 60
        return [{"item_id": f"i{j}", "prompt": f"q{j}"} for j in range(n)]

    def score_fn(items, responses):
        return [float(r) for r in responses]

    return generate_fn, items_fn, score_fn, calls


def test_a_without_format_demos_raises_unavailable_naming_the_field():
    gen, items, score, _ = _fake_runtime()
    ctx = _ctx(_cells(BASE_RATES), generate_fn=gen, items_fn=items, score_fn=score)
    with pytest.raises(ab.AblationUnavailable) as exc:
        asyncio.run(ab.run_ablation("A", ctx))
    assert "format_competence" in str(exc.value)


def test_a_detects_the_channel_hack_when_demos_lift_the_midtrain_only_arm():
    gen, items, score, calls = _fake_runtime(
        demo_rates={"M": 0.6, "S": 0.2, "base": 0.05}
    )
    spec = {
        "item_generator": "g",
        "prompt_template": "{question}",
        "scoring_rule": "exact",
        "format_competence": {"demos": [{"prompt": "DEMO q", "response": "A: 1"}]},
    }
    ctx = _ctx(
        _cells(BASE_RATES),
        eval_spec=spec,
        generate_fn=gen,
        items_fn=items,
        score_fn=score,
    )
    out = asyncio.run(ab.run_ablation("A", ctx))
    data = out["data"]

    assert data["rate_M_with_format_demo"] == pytest.approx(0.6)
    assert data["rate_S_with_format_demo"] == pytest.approx(0.2)
    assert data["fraction_of_M_to_T_gap_closed_by_demo"] == pytest.approx(1.0)
    assert "CHANNEL HACK LIKELY" in out["verdict_hint"]
    # M and S are each run twice (with and without demos); R and T once.
    assert sorted(calls) == sorted(
        [(c, False, 60) for c in CELLS] + [("M", True, 60), ("S", True, 60)]
    )


def test_a_clears_a_submission_the_format_demo_cannot_explain():
    gen, items, score, _ = _fake_runtime(demo_rates={"M": 0.25, "S": 0.2})
    spec = {
        "item_generator": "g",
        "prompt_template": "{question}",
        "scoring_rule": "exact",
        "format_competence": {"demos": ["DEMO q\nA: 1"]},
    }
    ctx = _ctx(
        _cells(BASE_RATES),
        eval_spec=spec,
        generate_fn=gen,
        items_fn=items,
        score_fn=score,
    )
    out = asyncio.run(ab.run_ablation("A", ctx))
    assert "channel explanation NOT supported" in out["verdict_hint"]


def test_b_without_an_elicitation_raises_unavailable():
    gen, items, score, _ = _fake_runtime()
    ctx = _ctx(_cells(BASE_RATES), generate_fn=gen, items_fn=items, score_fn=score)
    with pytest.raises(ab.AblationUnavailable) as exc:
        asyncio.run(ab.run_ablation("B", ctx))
    assert "prompted_belief" in str(exc.value)


def test_b_measures_the_prompted_base_ceiling_from_a_declared_claim():
    gen, items, score, _ = _fake_runtime()
    spec = {
        "item_generator": "g",
        "prompt_template": "{question}",
        "scoring_rule": "exact",
        "planted_claim": "Widgets are manufactured in Aveiro.",
    }
    ctx = _ctx(
        _cells(BASE_RATES),
        eval_spec=spec,
        generate_fn=gen,
        items_fn=items,
        score_fn=score,
    )
    out = asyncio.run(ab.run_ablation("B", ctx))
    assert out["data"]["rate_base_unprompted"] == pytest.approx(0.05)
    assert out["data"]["rate_T"] == pytest.approx(0.6)
    assert "training adds beyond prompting" in out["verdict_hint"]


def test_c_without_the_heldout_paraphrase_file_raises_unavailable():
    gen, items, score, _ = _fake_runtime()
    ctx = _ctx(_cells(BASE_RATES), generate_fn=gen, items_fn=items, score_fn=score)
    with pytest.raises(ab.AblationUnavailable) as exc:
        asyncio.run(ab.run_ablation("C", ctx))
    assert "escalation.json" in str(exc.value)


def test_d_without_multiple_choice_items_raises_unavailable():
    gen, items, score, _ = _fake_runtime()
    ctx = _ctx(_cells(BASE_RATES), generate_fn=gen, items_fn=items, score_fn=score)
    with pytest.raises(ab.AblationUnavailable) as exc:
        asyncio.run(ab.run_ablation("D", ctx))
    assert "{choices}" in str(exc.value)


def test_d_swaps_seen_distractors_and_recomputes(tmp_path):
    controls = tmp_path / "controls"
    controls.mkdir()
    (controls / "seen_distractors.json").write_text(
        '{"distractors": ["a seen phrase", "another seen phrase"]}'
    )
    gen, _items, score, _ = _fake_runtime()

    async def mc_items(**kwargs):
        n = kwargs.get("n") or 40
        return [
            {
                "item_id": f"i{j}",
                "question": f"q{j}",
                "choices": ["right", "wrong1", "wrong2"],
                "answer": "right",
            }
            for j in range(n)
        ]

    spec = {
        "item_generator": "g",
        "prompt_template": "Q: {question}\n{choices}\nA:",
        "scoring_rule": "exact",
    }
    ctx = _ctx(
        _cells(BASE_RATES),
        eval_spec=spec,
        heldout_root=tmp_path,
        generate_fn=gen,
        items_fn=mc_items,
        score_fn=score,
    )
    out = asyncio.run(ab.run_ablation("D", ctx))
    data = out["data"]
    assert data["distractor_pool_size"] == 2
    assert data["n_items"] == 40
    # Rates are unchanged by the swap in this fake, so the effect survives.
    assert "familiarity does NOT explain the effect" in out["verdict_hint"]


def test_items_may_be_objects_not_dicts():
    # The pod's real generator returns evalspec.Item-shaped objects (id/text),
    # not dicts; the ablations read both.
    class Item:
        def __init__(self, id, text):
            self.id, self.text, self.meta = id, text, {}

    it = Item("abc", "the prompt")
    assert ab._item_id(it, 0) == "abc"
    assert ab._prompt_of(it) == "the prompt"
    assert ab._field(Item("x", "y"), "missing") is None


def test_f_without_an_item_generator_raises_unavailable_naming_it():
    gen, _items, score, _ = _fake_runtime()
    ctx = _ctx(_cells(BASE_RATES), generate_fn=gen, items_fn=None, score_fn=score)
    with pytest.raises(ab.AblationUnavailable) as exc:
        asyncio.run(ab.run_ablation("F", ctx))
    assert "items_fn" in str(exc.value)


def test_f_regenerates_at_larger_n_with_a_fresh_seed():
    gen, items, score, calls = _fake_runtime()
    ctx = _ctx(_cells(BASE_RATES), generate_fn=gen, items_fn=items, score_fn=score)
    out = asyncio.run(ab.run_ablation("F", ctx))
    data = out["data"]
    assert data["requested_n"] == max(ab.MIN_FRESH_N, 2 * N)
    assert data["achieved_n"] == data["requested_n"]
    assert data["fresh_seed"] != ctx.seed
    assert data["same_sign_as_reported"] is True
    assert "REPLICATES" in out["verdict_hint"]
    assert [c for c, _, _ in calls] == list(CELLS)
