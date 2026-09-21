"""CPU-only contracts for the final-v1 designed charter-cost sweep."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_COINS = REPO_ROOT / "experiments" / "dispatch"
EXP = PRIOR_COINS / "dispatch_final_v1"
TEMPLATE_DIR = PRIOR_COINS / "template_diversity_v1"
for _p in (str(PRIOR_COINS), str(EXP), str(TEMPLATE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_costsweep_prompts as builder  # noqa: E402
import contracts as C  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_costsweep_v1 as scorer  # noqa: E402
import templates as T  # noqa: E402
from motivation_eval_v1 import generators as G  # noqa: E402

TRAIN_CLAUSES = (
    "qual_skill",
    "qual_specialty",
    "precedence_runs_year",
    "precedence_days_since",
    "precedence_registry_rank",
)


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    root = tmp_path_factory.mktemp("costsweep")
    source = root / "template"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(json.dumps({
        "train_clauses": list(TRAIN_CLAUSES),
        "templates": {
            "held_out_ids": sorted(
                template.template_id for template in T.held_out_templates())
        },
    }))
    out = root / "built"
    manifest = builder.build(source, out, n_per_bin=64, seed=C.COSTSWEEP_SEED)
    records = v4.read_records(out / "episodes" / "costsweep.jsonl")
    return out, manifest, records


def test_bins_are_the_five_decided_ratios():
    assert C.COSTSWEEP_CENTERS == (1.10, 1.25, 1.50, 2.00, 3.00)
    assert len(C.COSTSWEEP_BINS) == 5
    for center, (low, high) in zip(
            C.COSTSWEEP_CENTERS, C.COSTSWEEP_BINS, strict=True):
        assert low <= center <= high


def test_reference_gap_sweep_defaults_are_unchanged():
    implicit = G.gap_sweep(2, seed=17)
    explicit = G.gap_sweep(2, seed=17, bins=G.GAP_BINS)
    assert implicit == explicit


def test_every_generated_episode_clause_is_trained(swept):
    _, manifest, records = swept
    trained = set(manifest["train_clauses"])
    assert trained == set(TRAIN_CLAUSES)
    assert {record.metadata["target_clause"] for record in records} <= trained
    assert all(set(row["clauses"]) == trained for row in manifest["bins"])


def test_every_realized_ratio_falls_inside_its_bin(swept):
    _, manifest, records = swept
    assert len(records) == 64 * 5
    for record in records:
        index = record.metadata["bin_index"]
        low, high = C.COSTSWEEP_BINS[index]
        assert low <= record.metadata["realized_ratio"] <= high
    assert all(row["n"] == 64 for row in manifest["bins"])


def test_distractor_marginals_do_not_drift_across_bins(swept):
    _, manifest, _ = swept
    audit = manifest["distractor_marginals"]
    # This is the design's identifying restriction: every bin uses one fixed
    # distribution, and the deterministic realized sample has no large shift.
    assert audit["source_distribution"] == list(G.DISTRACTOR_RANGE)
    assert audit["max_mean_drift"] <= audit["max_allowed_mean_drift"]
    assert audit["max_mean_drift"] < 0.25
    assert set(audit["n_per_bin"].values()) == {128}


def test_rendering_is_held_out_and_manifest_records_requested_provenance(swept):
    out, manifest, records = swept
    heldout = {template.template_id for template in T.held_out_templates()}
    assert {item["template_id"] for item in manifest["items"]} == heldout
    assert all(item["template_id"] in heldout for item in manifest["items"])
    assert set(manifest["sha256s"]) == {
        "template_diversity_manifest", "episodes", "prompts"}
    required = {"bin", "realized_ratio", "gap_coins", "charter_cost_rank",
                "clause", "template_id", "prompt_sha256", "episode_sha256"}
    assert required <= set(manifest["items"][0])
    assert len(records) == len((out / "prompts" / "costsweep.jsonl")
                               .read_text().splitlines())


def test_scorer_routes_choices_through_factorised_parser(swept, tmp_path):
    out, _, records = swept
    response_dir = tmp_path / "charter" / "costsweep" / "pre_aft"
    response_dir.mkdir(parents=True)
    response_dir.joinpath("responses.jsonl").write_text("".join(
        json.dumps({
            "id": record.episode.episode_id,
            "response_text": dispatch.assignment_line(
                record.episode, record.episode.charter_plan),
        }) + "\n"
        for record in records
    ))
    coin_dir = tmp_path / "coin" / "costsweep" / "pre_aft"
    coin_dir.mkdir(parents=True)
    coin_dir.joinpath("responses.jsonl").write_text("".join(
        json.dumps({
            "id": record.episode.episode_id,
            "response_text": dispatch.assignment_line(
                record.episode, record.episode.coin_plan),
        }) + "\n"
        for record in records
    ))
    scored = scorer.score(tmp_path, out)
    rows = scored["arms"]["charter"]["pre_aft"]
    assert [row["n"] for row in rows] == [64] * 5
    assert [row["charter_choice_rate"] for row in rows] == [1.0] * 5
    assert [row["charter_choice_rate"] for row in
            scored["arms"]["coin"]["pre_aft"]] == [0.0] * 5
    assert "score_factorised" in scored["meta"]["parser"]


def test_costsweep_phase_gates_chain_complete_and_publishes_as_it_lands():
    source = (EXP / "pod" / "chain.py").read_text()
    default = source.split('parser.add_argument("--phases"', 1)[1].split(")", 1)[0]
    assert "d4,costsweep,publish" in default
    required = source.split("required = {", 1)[1].split("}", 1)[0]
    assert '"costsweep"' in required
    assert '"COSTSWEEP_COMPLETE"' in source
    assert 'arm, "costsweep")' in source
    execute = source.split("async def execute_arms", 1)[1]
    assert execute.index("await phase_d4_pooled(") < execute.index(
        "await phase_costsweep_pooled(")


@pytest.mark.parametrize("n_gpus,expected_sizes", [
    (4, [3, 2, 2, 2]),
    (3, [3, 3, 3]),
])
def test_launcher_shards_by_profile_gpu_count(tmp_path, n_gpus, expected_sizes):
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "capture.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        # Two contracts queries now: the profile line and the endpoint names.
        # Answering both with the profile line makes the launcher shard over
        # the single word "$FAKE_N_GPUS".
        "if [ \"${1:-}\" = -c ]; then\n"
        "  case \"${2:-}\" in\n"
        "    *eval_endpoint_names*)\n"
        "      printf '%s\\n' pre_aft"
        " agreement-step256 agreement-step512"
        " mixed_charter-step256 mixed_charter-step512"
        " mixed_coin-step256 mixed_coin-step512"
        " charter_only-step256 charter_only-step512\n"
        "      exit 0;;\n"
        "    *) echo \"$FAKE_N_GPUS 1 48 gemma3\"; exit 0;;\n"
        "  esac\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >> \"$CAPTURE\"\n"
    )
    fake_python.chmod(0o755)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{}\n')
    env = {
        **os.environ,
        "FINAL_V1_EVAL_PYTHON": str(fake_python),
        "COSTSWEEP_PROMPTS": str(prompts),
        "FAKE_N_GPUS": str(n_gpus),
        "CAPTURE": str(capture),
        "REPO": str(REPO_ROOT),
    }
    subprocess.run(
        ["bash", str(EXP / "pod" / "costsweep_sharded.sh"), "charter",
         str(tmp_path / "profile")],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    calls = capture.read_text().splitlines()
    assert len(calls) == n_gpus
    shards = [call.split("--endpoints ", 1)[1].split()[0].split(",")
              for call in calls]
    assert sorted(map(len, shards), reverse=True) == expected_sizes
    assert {endpoint for shard in shards for endpoint in shard} == {
        "pre_aft",
        "agreement-step256", "agreement-step512",
        "mixed_charter-step256", "mixed_charter-step512",
        "mixed_coin-step256", "mixed_coin-step512",
        "charter_only-step256", "charter_only-step512",
    }


def test_eval_engine_flags_are_centralised_and_default_safe(monkeypatch):
    """Both pre-launch eval flags live in one place, with the right defaults.

    Batching is applied by default: it only regroups requests, which is the bf16
    nondeterminism this campaign already accepts. CUDA graphs are NOT, because
    that is the one change whose failure would be silent -- it is gated on the
    A/B in MONITORING.md and enabled with FINAL_V1_CUDA_GRAPHS=1.
    """
    import importlib
    import sys

    pod = str(EXP / "pod")
    if pod not in sys.path:
        sys.path.insert(0, pod)
    import eval_runtime as rt

    rt = importlib.reload(rt)
    kwargs = rt.llm_kwargs(gpu_memory_utilization=0.6)
    assert kwargs["enforce_eager"] is True, "graphs must stay off until the A/B"
    assert kwargs["max_num_batched_tokens"] == 16384

    monkeypatch.setenv("FINAL_V1_CUDA_GRAPHS", "1")
    monkeypatch.setenv("FINAL_V1_MAX_BATCHED_TOKENS", "0")
    rt = importlib.reload(rt)
    kwargs = rt.llm_kwargs(gpu_memory_utilization=0.6)
    assert kwargs["enforce_eager"] is False
    assert "max_num_batched_tokens" not in kwargs, "0 restores vLLM's default"
    monkeypatch.delenv("FINAL_V1_CUDA_GRAPHS")
    monkeypatch.delenv("FINAL_V1_MAX_BATCHED_TOKENS")
    importlib.reload(rt)


def test_no_eval_script_sets_engine_flags_behind_llm_kwargs():
    """One owner for the engine flags, so a sweep cannot miss a call site."""
    pod = Path(__file__).resolve().parents[1] / (
        "experiments/dispatch/dispatch_final_v1/pod")
    for script in ("evaluate.py", "recall_eval.py", "d4_eval.py", "costsweep_eval.py"):
        body = (pod / script).read_text()
        assert "enforce_eager" not in body, f"{script} bypasses llm_kwargs"
        assert "max_num_batched_tokens" not in body, f"{script} bypasses llm_kwargs"
