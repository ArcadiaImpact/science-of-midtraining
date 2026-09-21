from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dispatch"
sys.path[:0] = [str(EXP), str(EXP / "pod")]

import analyse_dispatch_grpo_aft_v1 as analysis  # noqa: E402
import build_dispatch_grpo_aft_v1 as grpo_data  # noqa: E402
import dispatch_grpo_aft_v1_eval as evaluation  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def small_battery() -> list[dispatch.Episode]:
    return dispatch.generate_suite(n_per_kind=3, seed=17)


def test_frozen_battery_matches_completed_sft_fingerprints() -> None:
    battery = evaluation.frozen_battery()
    assert {kind: len(rows) for kind, rows in battery.items()} == {
        dispatch.AGREEMENT: 512,
        dispatch.CONFLICT: 512,
    }
    assert evaluation.battery_fingerprints(battery) == evaluation.FROZEN_SFT_FINGERPRINTS


def test_tagged_and_legacy_paths_share_semantic_outcome() -> None:
    record = evaluation.frozen_records()[dispatch.CONFLICT][0]
    episode = record.episode
    assignment = dispatch.assignment_line(episode, episode.charter_plan)
    tagged = f"<think>reason</think><answer>{assignment}</answer>"

    legacy_row = evaluation.score_response(assignment, episode, "legacy")
    tagged_row = evaluation.score_response(tagged, episode, "tagged")

    assert assignment not in evaluation.render_prompt(episode, "legacy")
    assert "<think>" in evaluation.render_prompt(episode, "tagged", record=record)
    assert legacy_row["outcome"] == tagged_row["outcome"] == "charter"
    assert legacy_row["parsed_plan"] == tagged_row["parsed_plan"]
    assert tagged_row["format_valid"] == 1.0


def test_tagged_prompt_is_exactly_the_task2_builder() -> None:
    record = evaluation.frozen_records()[dispatch.AGREEMENT][0]
    assert evaluation.render_prompt(record.episode, "tagged", record=record) == grpo_data.tagged_prompt(record.episode)


def test_evaluation_row_has_full_identity_and_resume_is_idempotent(tmp_path: Path) -> None:
    episode = small_battery()[0]
    answer = dispatch.assignment_line(episode, episode.charter_plan)
    identity = evaluation.EvalIdentity("coin", 2, 50, "abc123", "legacy", "greedy")
    row = evaluation.make_row(
        identity, episode, answer, response_tokens=37, entropy=0.4, kl=0.03
    )
    assert {"parent", "seed", "checkpoint", "checkpoint_id", "interface", "item_id"} <= row.keys()
    assert row["response_tokens"] == 37
    assert row["entropy"] == 0.4

    output = tmp_path / "rows.jsonl"
    evaluation.append_missing_rows(output, [row])
    evaluation.append_missing_rows(output, [row])
    assert len(output.read_text().splitlines()) == 1


def test_paired_bootstrap_recovers_known_effect() -> None:
    pairs = [(1.0, 0.0)] * 80 + [(0.0, 0.0)] * 20
    result = analysis.paired_bootstrap(pairs, n_resamples=2_000, seed=4)
    assert result["estimate"] == pytest.approx(0.8)
    assert result["low"] > 0.7
    assert result["high"] <= 0.9


def write_task5_grid(
    root: Path, *, omit: tuple[str, int, str] | None = None, with_metrics: bool = False
) -> None:
    for parent in evaluation.PARENTS:
        for seed in (11, 22, 33):
            run = root / parent / f"seed-{seed}"
            run.mkdir(parents=True)
            hashes, endpoints, remotes = {}, {}, []
            for checkpoint in evaluation.CHECKPOINTS:
                label = f"{checkpoint:03d}"
                if omit == (parent, seed, label):
                    continue
                endpoint = run / "train" / f"endpoint-{label}"
                endpoint.mkdir(parents=True)
                (endpoint / "config.json").write_text("{}")
                hashes[label] = evaluation.hash_checkpoint(endpoint)
                endpoints[str(checkpoint / 100)] = str(endpoint)
                remotes.append(f"dispatch_grpo_aft_v1/{parent}/seed-{seed}/checkpoints/{label}")
            logs = []
            if with_metrics:
                log = run / "training_metrics.jsonl"
                log.write_text(json.dumps({
                    "checkpoint": 25, "entropy": 0.61, "clip_ratio": 0.07,
                    "kl": 0.03, "zero_variance_group_fraction": 0.12,
                }) + "\n")
                logs.append(str(log))
            (run / "progress.json").write_text(json.dumps({
                "endpoints": endpoints, "logs": logs,
            }))
            (run / "COMPLETE.json").write_text(json.dumps({
                "parent": parent, "seed": seed, "status": "complete",
                "endpoint_sha256": hashes, "remote_paths": remotes,
            }))


def test_driver_runs_exact_full_grid_with_injected_services(tmp_path: Path) -> None:
    write_task5_grid(tmp_path, with_metrics=True)
    records = {dispatch.AGREEMENT: evaluation.frozen_records()[dispatch.AGREEMENT][:1]}
    loaded, sampled = [], []

    def loader(endpoint: evaluation.Endpoint):
        loaded.append(endpoint)
        return endpoint

    def sampler(model, prompts, decoding):
        sampled.append((model, tuple(prompts), decoding.name))
        episode = records[dispatch.AGREEMENT][0].episode
        assignment = dispatch.assignment_line(episode, episode.charter_plan)
        text = assignment if "Assignment:" in prompts[0] and "<think>" not in prompts[0] else f"<think>x</think><answer>{assignment}</answer>"
        return [evaluation.Sample(
            text, token_count=19, diagnostics={"cumulative_logprob": -1.25}
        )]

    output = tmp_path / "evaluation.jsonl"
    rows = evaluation.run_evaluation(
        tmp_path, output, loader=loader, sampler=sampler, records=records
    )
    assert len(loaded) == 4 * 3 * 5
    assert len(sampled) == 4 * 3 * 5 * 2 * 2
    assert len(rows) == 4 * 3 * 5 * 2 * 2
    assert all(row["response_tokens"] == 19 and row["cumulative_logprob"] == -1.25 for row in rows)
    quarter = next(row for row in rows if row["checkpoint"] == 25)
    assert quarter["entropy"] == 0.61
    assert quarter["clip_ratio"] == 0.07
    assert quarter["kl"] == 0.03
    assert quarter["zero_variance_group_fraction"] == 0.12
    evaluation.validate_full_grid(rows, records)


def test_endpoint_resolution_falls_back_to_injected_hf_resolver(tmp_path: Path) -> None:
    write_task5_grid(tmp_path)
    local = tmp_path / "coin" / "seed-11" / "train" / "endpoint-025"
    for child in local.iterdir():
        child.unlink()
    local.rmdir()
    calls = []
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "config.json").write_text("{}")
    endpoints = evaluation.discover_endpoints(
        tmp_path, hf_resolver=lambda remote, sha: calls.append((remote, sha)) or downloaded
    )
    chosen = next(
        endpoint for endpoint in endpoints
        if (endpoint.parent, endpoint.seed, endpoint.checkpoint) == ("coin", 11, 25)
    )
    assert chosen.path == downloaded
    assert calls == [(
        "dispatch_grpo_aft_v1/coin/seed-11/checkpoints/025",
        evaluation.hash_checkpoint(tmp_path / "coin" / "seed-11" / "train" / "endpoint-000"),
    )]


def test_hf_resolver_result_is_independently_hash_verified(tmp_path: Path) -> None:
    write_task5_grid(tmp_path)
    local = tmp_path / "coin" / "seed-11" / "train" / "endpoint-025"
    for child in local.iterdir():
        child.unlink()
    local.rmdir()
    corrupt = tmp_path / "corrupt-download"
    corrupt.mkdir()
    (corrupt / "config.json").write_text('{"wrong": true}')
    with pytest.raises(ValueError, match="resolved checkpoint hash mismatch"):
        evaluation.discover_endpoints(tmp_path, hf_resolver=lambda remote, sha: corrupt)


def test_task5_training_metrics_join_by_arm_and_checkpoint(tmp_path: Path) -> None:
    write_task5_grid(tmp_path, with_metrics=True)
    metrics = evaluation.load_training_metrics(tmp_path)
    assert metrics[("charter", 11, 25)] == {
        "entropy": 0.61, "clip_ratio": 0.07, "kl": 0.03,
        "zero_variance_group_fraction": 0.12,
    }


def test_missing_task5_endpoint_and_incomplete_eval_grid_are_rejected(tmp_path: Path) -> None:
    write_task5_grid(tmp_path, omit=("mixed", 22, "075"))
    with pytest.raises(ValueError, match="full Task 5 endpoint grid"):
        evaluation.discover_endpoints(tmp_path)
    with pytest.raises(ValueError, match="evaluation grid"):
        evaluation.validate_full_grid([], {dispatch.AGREEMENT: evaluation.frozen_records()[dispatch.AGREEMENT][:1]})


def synthetic_rows() -> list[dict[str, object]]:
    rows = []
    parent_rates = {"charter": (0.8, 0.1), "coin": (0.2, 0.7), "mixed": (0.55, 0.35), "neutral": (0.4, 0.4)}
    for parent in evaluation.PARENTS:
        for seed in range(3):
            for checkpoint in (0, 50, 100):
                for interface in ("tagged", "legacy"):
                    for decoding in ("greedy", "stochastic"):
                      for kind in ("agreement", "conflict"):
                        for item in range(20):
                            charter_rate, coin_rate = parent_rates[parent]
                            charter = float(kind == "conflict" and item < charter_rate * 20)
                            coin = float(kind == "conflict" and item >= charter_rate * 20 and item < (charter_rate + coin_rate) * 20)
                            rows.append({
                                "parent": parent, "seed": seed, "checkpoint": checkpoint,
                                "checkpoint_id": f"{parent}-{seed}-{checkpoint}",
                                "interface": interface, "decoding": decoding,
                                "kind": kind, "item_id": f"{kind}-{item}",
                                "agreement_correct": float(kind == "agreement"),
                                "charter_choice": charter, "coin_choice": coin,
                                "reward": checkpoint / 100, "response_tokens": 20 + checkpoint / 10,
                                "format_valid": float(interface == "tagged"),
                                "entropy": 0.5, "clip_ratio": 0.05, "kl": 0.02,
                                "zero_variance_group_fraction": 0.1,
                            })
    return rows


def test_primary_estimands_distinguish_all_parents_and_bootstrap_seeds() -> None:
    result = analysis.primary_trajectories(synthetic_rows(), n_resamples=500)
    endpoint = next(
        row for row in result
        if row["checkpoint"] == 100
        and row["interface"] == "legacy"
        and row["decoding"] == "greedy"
    )
    assert set(endpoint["parent_rates"]) == set(evaluation.PARENTS)
    assert endpoint["charter_separation"]["estimate"] == pytest.approx(0.6)
    assert endpoint["coin_separation"]["estimate"] == pytest.approx(0.6)
    assert endpoint["directional_sum"]["estimate"] == pytest.approx(1.2)
    assert endpoint["directional_sum"]["replication_unit"] == "seed"
    assert endpoint["directional_sum"]["n_seeds"] == 3


def test_hierarchical_logistic_recovers_categorical_parent_dose_effect() -> None:
    pytest.importorskip("statsmodels")  # analysis extra, not in the lean suite
    import math
    import random

    rng = random.Random(91)
    rows = []
    effects = {"neutral": 0.0, "coin": -0.5, "mixed": 0.3, "charter": 0.8}
    slopes = {"neutral": 0.0, "coin": -0.4, "mixed": 0.4, "charter": 1.4}
    for parent in evaluation.PARENTS:
        for seed in range(3):
            for episode in range(35):
                for dose in (0.0, 0.5, 1.0):
                    logit = -0.7 + effects[parent] + slopes[parent] * dose
                    probability = 1 / (1 + math.exp(-logit))
                    rows.append({
                        "success": float(rng.random() < probability), "parent": parent,
                        "dose": dose, "seed": str(seed), "episode": str(episode),
                    })
    fit = analysis.hierarchical_logistic(rows)
    assert fit["method"] == "statsmodels BinomialBayesMixedGLM fit_vb"
    assert set(fit["random_intercepts"]) == {"seed", "episode"}
    charter_interaction = next(
        value for name, value in fit["fixed_effects"].items()
        if "charter" in name and ":dose" in name
    )
    assert charter_interaction["mean"] > 0.4


def test_interface_pairing_includes_checkpoint_identity_and_decoding() -> None:
    rows = synthetic_rows()
    contrasts = analysis.paired_interface_contrasts(rows, n_resamples=100)
    expected_pairs = 4 * 3 * 3 * 2 * 2 * 20
    assert contrasts["format_valid"]["n_pairs"] == expected_pairs


def test_analysis_and_seaborn_plots_write_required_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("statsmodels")  # analysis extra, not in the lean suite
    pytest.importorskip("seaborn")
    rows = synthetic_rows()
    result = analysis.write_analysis(rows, tmp_path, bootstrap_resamples=200)
    figures = sorted((tmp_path / "figures").glob("*.pdf"))

    assert (tmp_path / "analysis.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    assert result["n_rows"] == len(rows)
    assert {path.name for path in figures} == {
        "agreement_learning.pdf", "conflict_separation.pdf",
        "reward_length_diagnostics.pdf", "tagged_vs_legacy.pdf",
    }
    assert all(path.read_bytes().startswith(b"%PDF") for path in figures)
    assert json.loads((tmp_path / "analysis.json").read_text())["n_rows"] == len(rows)
    report = (tmp_path / "REPORT.md").read_text()
    assert "Charter separation" in report and "+0.600" in report
    assert "Agreement gates" in report
    assert "Tagged minus legacy" in report
    assert "BinomialBayesMixedGLM" in report
