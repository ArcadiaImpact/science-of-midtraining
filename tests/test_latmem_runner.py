"""CPU-only contracts for the prior-latmem runner and pod-side pure helpers."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import figures, run, smoke
from experiments.prior_latmem.pod import chain, sample_arms


def _install_fake_bellhop(monkeypatch):
    import dataclasses

    fake = types.ModuleType("bellhop")
    fake.calls = []
    fake.outcomes = []

    class ProvisionError(Exception):
        pass

    class PodNotReadyError(Exception):
        pass

    class ResultsMissingError(Exception):
        pass

    class RemoteJobError(Exception):
        def __init__(self, message, *, log_tail=""):
            super().__init__(message)
            self.log_tail = log_tail

    class RunSpec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # A real dataclass, like bellhop's PodConfig: bellhop internally calls
    # dataclasses.replace(pod, ...), which must preserve the Cu13 subclass
    # and its to_graphql_input override.
    @dataclasses.dataclass
    class PodConfig:
        gpu: str = "H200"
        gpu_count: int = 1
        cloud: str | None = None
        cloud_fallback: bool | None = None
        container_disk_gb: int | None = None
        provision_timeout: timedelta | None = None
        ready_timeout: timedelta | None = None
        max_lifetime: timedelta | None = None
        name: str | None = None

        def to_graphql_input(self, gpu_type_id=None):
            return {"gpuTypeId": gpu_type_id or self.gpu}

    async def fake_run(spec, pod):
        # Mirror bellhop.run's internal dataclasses.replace before use.
        pod = dataclasses.replace(pod, name=pod.name)
        fake.calls.append((spec, pod))
        outcome = fake.outcomes.pop(0) if fake.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake.ProvisionError = ProvisionError
    fake.PodNotReadyError = PodNotReadyError
    fake.ResultsMissingError = ResultsMissingError
    fake.RemoteJobError = RemoteJobError
    fake.RunSpec = RunSpec
    fake.PodConfig = PodConfig
    fake.run = fake_run
    backend = types.ModuleType("bellhop.backend")
    backend.TAR_EXCLUDES = ["--exclude=.git"]
    fake.backend = backend
    monkeypatch.setitem(sys.modules, "bellhop", fake)
    monkeypatch.setitem(sys.modules, "bellhop.backend", backend)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return fake


def _pod_cfg(**kwargs):
    return run.Config(
        signed_off=True,
        confirm=True,
        sample_retry_delay_seconds=0,
        **kwargs,
    )


def test_pod_configs_cuda13_filter_is_sampling_only(monkeypatch, tmp_path):
    bellhop = _install_fake_bellhop(monkeypatch)

    assert asyncio.run(run.pod_sample(_pod_cfg(), tmp_path)) == tmp_path / "eval_raw"
    assert asyncio.run(run.pod_train(_pod_cfg(), tmp_path)) == tmp_path / "pod_raw"

    assert len(bellhop.calls) == 2
    sample_pod = bellhop.calls[0][1]
    train_pod = bellhop.calls[1][1]
    # The cu13 filter serves the vLLM wheel on the sampling pod; the training
    # pod runs cu126 torch and must keep the full (cu12-inclusive) host pool.
    # Both assertions run on the post-dataclasses.replace pod, locking in
    # that the subclass override survives bellhop's internal replace.
    assert sample_pod.to_graphql_input()["allowedCudaVersions"] == [
        "13.0", "13.1", "13.2", "13.3",
    ]
    assert "allowedCudaVersions" not in train_pod.to_graphql_input()
    for pod in (sample_pod, train_pod):
        assert pod.max_lifetime > timedelta(0)
    with pytest.raises(ValueError, match="positive max_lifetime"):
        run._pod_config(bellhop, gpu="H200")
    with pytest.raises(ValueError, match="positive max_lifetime"):
        run._pod_config(bellhop, cuda13=True, gpu="H200")


def test_eval_setup_fails_fast_before_package_installation():
    setup = run._eval_setup()
    apt = (
        "{ ldconfig -p | grep -q libavutil && command -v ninja >/dev/null; } || "
        "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y -qq ffmpeg ninja-build)"
    )

    lines = setup.split("\n")
    assert lines[0] == "set -euo pipefail"
    assert "DRIVER_TOO_OLD_FOR_CU13" in setup
    assert setup.index("DRIVER_TOO_OLD_FOR_CU13") < setup.index("apt-get")
    # The guard must live on its own line: joined with " && ", its exit 41
    # would flow into the apt fallback's "||" and be swallowed (left-assoc
    # equal-precedence chaining), reaching RUN on a too-old host.
    guard_line = next(line for line in lines if "DRIVER_TOO_OLD_FOR_CU13" in line)
    assert "||" not in guard_line and "apt" not in guard_line
    assert apt in setup
    assert "apt-get update -q >/dev/null 2>&1 || true" not in setup
    assert "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true" not in setup


def test_eval_setup_driver_guard_aborts_in_real_bash(tmp_path):
    import stat
    import subprocess

    # Execute the first lines of the real setup under bash with a stubbed
    # nvidia-smi: the string-only test cannot catch join-semantics bugs.
    setup_lines = run._eval_setup().split("\n")
    guard_index = next(
        index for index, line in enumerate(setup_lines) if "nvidia-smi" in line
    )
    script = "\n".join(setup_lines[: guard_index + 1] + ["echo REACHED_PACKAGES"])

    def run_with_driver(version: str):
        stub = tmp_path / f"bin-{version.split('.')[0]}"
        stub.mkdir(exist_ok=True)
        nvidia = stub / "nvidia-smi"
        nvidia.write_text(f"#!/bin/bash\necho '{version}, NVIDIA H200'\n")
        nvidia.chmod(nvidia.stat().st_mode | stat.S_IEXEC)
        return subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env={"PATH": f"{stub}:/usr/bin:/bin"},
        )

    old = run_with_driver("570.86.15")
    assert old.returncode == 41
    assert "DRIVER_TOO_OLD_FOR_CU13" in old.stdout
    assert "REACHED_PACKAGES" not in old.stdout

    new = run_with_driver("581.15.03")
    assert new.returncode == 0
    assert "REACHED_PACKAGES" in new.stdout


@pytest.mark.parametrize(
    "failure_kind",
    [
        "provision",
        "network",
        "connect_timeout",
        "dns",
        "pod_not_ready",
        "results_missing",
        "old_driver",
    ],
)
def test_pod_sample_retries_only_classified_failures(
    monkeypatch, tmp_path, failure_kind,
):
    bellhop = _install_fake_bellhop(monkeypatch)
    failures = {
        "provision": lambda: bellhop.ProvisionError("out of capacity"),
        "network": lambda: OSError("ConnectError: Connection reset by peer"),
        "connect_timeout": lambda: OSError("httpx.ConnectTimeout: timed out"),
        "dns": lambda: OSError(
            "Temporary failure in name resolution for api.runpod.io"
        ),
        "pod_not_ready": lambda: bellhop.PodNotReadyError("stuck provisioning"),
        "results_missing": lambda: bellhop.ResultsMissingError(
            "job succeeded but no results dir"
        ),
        "old_driver": lambda: bellhop.RemoteJobError(
            "remote setup failed",
            log_tail="host driver: 570.0, H200\nDRIVER_TOO_OLD_FOR_CU13",
        ),
    }
    bellhop.outcomes.extend([failures[failure_kind](), None])

    result = asyncio.run(run.pod_sample(_pod_cfg(), tmp_path))

    assert result == tmp_path / "eval_raw"
    assert len(bellhop.calls) == 2


def test_pod_sample_raises_nonretryable_remote_error_with_full_log_path(
    monkeypatch, tmp_path,
):
    bellhop = _install_fake_bellhop(monkeypatch)
    bellhop.outcomes.append(
        bellhop.RemoteJobError("chat template exploded", log_tail="short tail")
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run.pod_sample(_pod_cfg(), tmp_path))

    assert len(bellhop.calls) == 1
    assert str(tmp_path / "eval_raw" / "run.log") in str(exc_info.value)
    assert "non-retryable" in str(exc_info.value)


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
    assert sample_arms.needs_sampling(tmp_path, "aft_p0_pr_f0", "grid")


def test_comprehension_items_are_sampled_into_the_dominated_store(tmp_path):
    """The >=0.90 comprehension gate needs rows: comprehension.jsonl was built
    and published but never sampled, so `comprehension_accuracy` was always n=0
    and every post-AFT arm would have been flagged instead of gated."""
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    for name in sample_arms.BATTERY_FILES.values():
        if name == "capability":
            continue
        (eval_root / f"{name}.jsonl").write_text('{"probe": "p"}\n')
    (eval_root / "comprehension.jsonl").write_text('{"probe": "c"}\n')

    files = sample_arms._eval_files(eval_root, ["dominated"])
    assert set(files) == {"dominated", "comprehension"}
    assert [path.name for path in sample_arms.battery_probe_files(files, "dominated")] == [
        "dominated.jsonl",
        "comprehension.jsonl",
    ]
    assert [path.name for path in sample_arms.battery_probe_files(files, "grid")] == []

    # Either file changing must invalidate the dominated store.
    before = sample_arms._sampling_config_hash(
        "it-base", "dominated",
        eval_files=sample_arms.battery_probe_files(files, "dominated"),
        execute_lm_eval=False,
    )
    (eval_root / "comprehension.jsonl").write_text('{"probe": "c2"}\n')
    after = sample_arms._sampling_config_hash(
        "it-base", "dominated",
        eval_files=sample_arms.battery_probe_files(files, "dominated"),
        execute_lm_eval=False,
    )
    assert before != after


def test_every_battery_has_a_token_budget_that_can_finish_an_answer():
    """refs_v1 regression: the old ``.get(battery, 64)`` default truncated 100%
    of the free-form battery's responses, and the judge scored the stumps."""
    for battery in sample_arms.BATTERY_FILES.values():
        budget = sample_arms.MAX_TOKENS.get(battery, sample_arms.MAX_TOKENS_DEFAULT)
        assert budget >= 128, f"{battery} budget {budget} is too small to finish an answer"
    # Free-form batteries need materially more room than letter-answer ones.
    for battery in ("stated", "prreview", "codewrite", "thrash"):
        assert sample_arms.MAX_TOKENS[battery] >= 768
    # The budget is part of the sampling fingerprint, so changing it invalidates
    # stale samples instead of silently mixing budgets within one arm.
    assert sample_arms._sampling_config_hash(
        "it-base", "stated", eval_file=None, execute_lm_eval=False
    ) != sample_arms._sampling_config_hash(
        "it-base", "prreview", eval_file=None, execute_lm_eval=False
    )


def test_scoring_reports_truncated_rows_and_tolerates_skipped_judgments():
    rows = [
        {
            "response": "**Answer A: prioritize lower memory when they conflict.**",
            "finish_reason": "stop",
            "meta": {"kind": "forced", "memory_letter": "A"},
        },
        {
            "response": "I generally start by asking whether the hot path is",
            "finish_reason": "length",
            "label": "UNCLEAR",
            "judge_raw": "UNCLEAR",
            "meta": {"kind": "freeform"},
        },
    ]
    # The freeform row carries a label and the forced row is parsed, so no judge
    # transport is needed: this exercises the scoring wrapper, not the network.
    aggregate = asyncio.run(run._score_battery("stated", [dict(row) for row in rows]))
    assert aggregate["stated_memory_first_rate"]["rate"] == 1.0
    assert aggregate["truncated_n"] == 1
    assert aggregate["finish_reason_reported_n"] == 2
    assert "stated_truncated" in run.assemble_result_row("ceiling_z2", {"stated": aggregate})["flags"]
    # A store that reports no finish reasons is unverified, not verified-complete.
    silent = {"truncated_n": 0, "finish_reason_reported_n": 0}
    assert "stated_completion_unverified" in run.assemble_result_row(
        "ceiling_z2", {"stated": silent}
    )["flags"]


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


def test_pod_sample_passes_arm_and_battery_subsets_to_pod_env(
    monkeypatch, tmp_path,
):
    bellhop = _install_fake_bellhop(monkeypatch)
    cfg = _pod_cfg(batteries="grid,stated")

    asyncio.run(run.pod_sample(cfg, tmp_path, arms=["it-base", "ceiling_z1"]))

    spec = bellhop.calls[0][0]
    assert spec.env["PRIOR_LATMEM_ARMS"] == "it-base,ceiling_z1"
    assert spec.env["PRIOR_LATMEM_BATTERIES"] == "grid,stated"

    bellhop.calls.clear()
    asyncio.run(run.pod_sample(_pod_cfg(), tmp_path))
    spec = bellhop.calls[0][0]
    assert "PRIOR_LATMEM_ARMS" not in spec.env
    assert "PRIOR_LATMEM_BATTERIES" not in spec.env


def test_sample_retry_classifies_empty_str_timeout_by_type_name(monkeypatch):
    bellhop = _install_fake_bellhop(monkeypatch)

    class ConnectTimeout(Exception):
        """Mirrors httpx.ConnectTimeout: stringifies empty."""

    assert run._sample_retry_reason(bellhop, ConnectTimeout()) is not None


def test_pod_sample_path_gate_runs_before_each_attempt(monkeypatch, tmp_path):
    bellhop = _install_fake_bellhop(monkeypatch)
    gates = []

    async def fake_gate():
        gates.append(True)

    monkeypatch.setattr(run, "_await_api_path", fake_gate)
    bellhop.outcomes.extend([bellhop.ProvisionError("busy"), None])

    asyncio.run(run.pod_sample(_pod_cfg(path_gate=True), tmp_path))
    assert len(gates) == 2  # one gate per attempt, incl. the retry

    gates.clear()
    asyncio.run(run.pod_sample(_pod_cfg(), tmp_path))
    assert gates == []  # default off: CPU tests and calm networks skip it


def test_pod_sample_sweeps_own_pods_after_failed_attempt(monkeypatch, tmp_path):
    bellhop = _install_fake_bellhop(monkeypatch)
    swept = []

    async def fake_sweep(name):
        swept.append(name)

    monkeypatch.setattr(run, "_sweep_own_pods", fake_sweep)
    bellhop.outcomes.extend([OSError("ConnectError: reset"), None])

    asyncio.run(run.pod_sample(_pod_cfg(), tmp_path))
    assert swept == ["bellhop-prior-latmem-sample"]  # once per failed attempt


def test_pod_drivers_slim_the_codebase_push(monkeypatch, tmp_path):
    bellhop = _install_fake_bellhop(monkeypatch)

    asyncio.run(run.pod_sample(_pod_cfg(), tmp_path))
    asyncio.run(run.pod_train(_pod_cfg(), tmp_path))

    excludes = sys.modules["bellhop.backend"].TAR_EXCLUDES
    assert excludes.count("--exclude=*prior_latmem/runs*") == 1


def test_eval_setup_composes_a_valid_uv_install_command():
    setup = run._eval_setup()
    install_line = next(
        line for line in setup.split("\n") if "uv pip install" in line
    )
    # "uv pip install -q pip install <pkgs>" resolves a package named
    # "install" and kills the pod at setup; the composed args must be
    # packages only.
    tokens = install_line.split("uv pip install -q ")[-1].split()
    assert "pip" not in tokens and "install" not in tokens
    assert "vllm" in tokens


def test_eval_setup_installs_scimt_into_the_pod_venv():
    # sample_arms imports scimt on the pod; without the editable install the
    # first live run died at ModuleNotFoundError after a full model download.
    setup = run._eval_setup()
    assert "uv pip install -q -e '.[data,hub]'" in setup
    assert setup.index("pod-vllm.txt") < setup.index("-e '.[data,hub]'")
