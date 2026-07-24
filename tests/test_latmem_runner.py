"""CPU-only contracts for the prior-latmem runner and pod-side pure helpers."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import figures, run, smoke
from experiments.prior_latmem.pod import chain, sample_arms


def test_chain_plan_has_all_links_and_token_budgets():
    entries = chain.plan()
    assert len(entries) == 48
    sdf = entries[:6]
    ri = entries[6:12]
    aft = entries[12:]
    assert len(sdf) == 6 and len(ri) == 6 and len(aft) == 36
    assert {item["stage"] for item in sdf} == {"sdf_it_gemma3_12b"}
    assert {item["stage"] for item in ri} == {"sft_reinstruct_it_gemma3_12b"}
    assert {item["stage"] for item in aft} == {"sft_task_it_gemma3_12b"}
    assert all(item["resume_of"] is None for item in sdf)
    assert all(item["resume_of"].startswith("sdf_") for item in ri)
    assert all(item["resume_of"].endswith("_ri") for item in aft)
    assert chain.token_budgets(0) == {"z1": 10_000_000, "z2": 0}
    assert chain.token_budgets(30) == {"z1": 7_000_000, "z2": 3_000_000}
    assert chain.token_budgets(100) == {"z1": 0, "z2": 10_000_000}
    assert {item["dataset"] for item in aft} == {"pr_f0", "pr_f01", "pr_f10", "code_f0", "code_f01", "code_f10"}


def test_chain_descendants_match_resume_edges():
    assert len(chain.descendants("sdf_p0")) == 7
    assert len(chain.descendants("sdf_p0_ri")) == 6
    assert chain.descendants("aft_p0_pr_f0") == set()


def test_sample_arm_subsets_and_store_idempotence(tmp_path):
    assert len(sample_arms.arm_names()) == 45
    assert sample_arms.batteries_for_arm("aft_p50_pr_f01") == (
        "grid", "dominated", "codewrite", "prreview", "context", "stated", "thrash", "capability"
    )
    assert sample_arms.batteries_for_arm("sdf_p50_ri") == (
        "grid", "dominated", "codewrite", "prreview", "capability"
    )
    assert sample_arms.batteries_for_arm("it-base") == (
        "grid", "dominated", "codewrite", "prreview", "context", "capability"
    )
    assert sample_arms.batteries_for_arm("ceiling_z1") == ("grid", "codewrite", "stated")
    path = sample_arms.sample_path(tmp_path, "aft_p0_pr_f0", "grid")
    assert sample_arms.needs_sampling(tmp_path, "aft_p0_pr_f0", "grid")
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    assert not sample_arms.needs_sampling(tmp_path, "aft_p0_pr_f0", "grid")


def test_forced_logprob_alignment_uses_lcp_suffix_for_boundary_merge(monkeypatch):
    class BoundaryTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "Question: ":
                return [10, 11, 12]
            if text == "Question: Patch A.":
                return [10, 11, 99, 13]
            if text == "Question: Patch B.":
                return [10, 11, 98, 14]
            raise AssertionError(text)

        def apply_chat_template(self, _messages, tokenize=False, add_generation_prompt=True):
            return "Question: "

    tokenizer = BoundaryTokenizer()
    assert sample_arms.continuation_token_suffix(tokenizer, "Question: ", "Patch A.") == ([99, 13], 2)
    assert sample_arms.continuation_token_suffix(tokenizer, "Question: ", "Patch B.") == ([98, 14], 2)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOutput:
        def __init__(self, token_values):
            self.prompt_logprobs = [None, None, *token_values]

    class FakeLLM:
        def generate(self, prompts, _params):
            assert prompts == [
                "Question: Patch A.", "Question: Patch B.",
            ]
            return [
                FakeOutput(({99: -1.0}, {13: -2.0})),
                FakeOutput(({98: -5.0}, {14: -6.0})),
            ]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    sampler = types.SimpleNamespace(tok=tokenizer, llm=FakeLLM())
    rows = [{"probe": "Question: ", "meta": {"memory_letter": "B"}}]
    scored = sample_arms.forced_continuation_scores(sampler, rows)
    assert scored[0]["logprob_memory"] == pytest.approx(-11.0)
    assert scored[0]["logprob_speed"] == pytest.approx(-3.0)


def test_result_assembly_wires_comprehension_and_capability_flags():
    aggregates = {"dominated": {"comprehension_accuracy": {"rate": 0.8, "n": 10}}}
    capability = {"humaneval_pass1": 0.8, "inst_level_strict_acc": 0.8, "mmlu_acc": 0.4}
    anchor = {"humaneval_pass1": 1.0, "inst_level_strict_acc": 1.0, "mmlu_acc": 0.5}
    row = run.assemble_result_row("aft_p50_pr_f01", aggregates,
                                  capability=capability, it_base_capability=anchor)
    assert row["p"] == 50 and row["f"] == 0.1
    assert "comprehension_gate_failed" in row["flags"]
    assert "humaneval_guard_failed" in row["flags"]
    assert "ifeval_guard_failed" in row["flags"]
    assert "mmlu_guard_failed" in row["flags"]
    assert run.Config(stage="score", confirm=False).stage == "score"
    with pytest.raises(PermissionError):
        run.require_spend_gate(run.Config(), "sampling")


def _analysis_rows():
    rows = []
    for fraction, slope in ((0.0, 0.01), (0.1, 0.004), (1.0, 0.0)):
        for p in (0, 50, 100):
            rate = 0.2 + slope * p
            rows.append({
                "p": p, "f": fraction, "modality": "pr",
                "items": [{"memory_first": int(i < round(rate * 20))} for i in range(20)],
                "per_battery": {"grid": {"memory_first_rate": {"rate": rate, "n": 20, "ci": (max(0, rate - .05), min(1, rate + .05))}}},
            })
    for p, value in ((0, .1), (30, .6), (50, .8), (70, .5), (100, .1)):
        rows.append({"p": p, "f": 0.0, "modality": "pr",
                     "per_battery": {"thrash": {"thrash_rate": {"rate": value, "n": 10}}}})
    return rows


def test_figures_statistics_and_pngs(tmp_path):
    rows = _analysis_rows()
    result = figures.bootstrap_slope_contrast(rows, n_boot=40, seed=3)
    assert result["slopes"][0.0] > result["slopes"][0.1] > result["slopes"][1.0]
    assert result["ordered_fraction"] is not None
    assert figures.interior_peak_statistic(rows)["statistic"] == pytest.approx(0.7)
    results_path = tmp_path / "results.jsonl"
    results_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    paths = figures.make_figures(results_path, tmp_path / "figures")
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_transfer_matrix_uses_grid_for_pr_and_codewrite_for_code():
    rows = []
    for modality in ("pr", "code"):
        for p in (0, 50, 100):
            rows.append({
                "p": p,
                "f": 0.0,
                "modality": modality,
                "per_battery": {
                    "grid": {"memory_first_rate": {"rate": 0.1 + p / 1000}},
                    "codewrite": {"memory_lean_rate": {"rate": 0.8 - p / 1000}},
                },
            })
    matrix = figures._transfer_matrix(rows, 0.0)
    assert matrix[0][0] != matrix[0][1]
    assert matrix[1][0] != matrix[1][1]


def test_smoke_stub_aggregates_and_figures(tmp_path):
    aggregates = smoke.stub_results()
    assert set(aggregates) == {"grid", "dominated", "codewrite", "prreview", "context", "stated", "thrash"}
    assert aggregates["dominated"]["comprehension_accuracy"]["rate"] == 1.0
    assert aggregates["codewrite"]["correctness_rate"]["rate"] == pytest.approx(2 / 3)
    assert aggregates["codewrite"]["memory_lean_rate"]["rate"] == 0.5
    result_path = smoke.score_stub_results(tmp_path)
    assert result_path.exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
