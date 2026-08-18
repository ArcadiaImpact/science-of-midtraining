"""Pod driver: reconstitute -> smoke (gated) -> attribution runs -> publish.

Runs on a single attribution pod under Bellhop (see ../run.py). GPU phases
run as SUPERVISED subprocesses of the sanctioned ``scimt-attribution`` shim
(the axolotl-backend carve-out pattern: config-first — the rendered YAML is
the whole interface — stdout streamed into this log, raise-with-tail on
failure). In-process phase chaining retained GPU memory between phases and
OOM'd run 20260818T140044Z at the phase AFTER a 109 GiB fit-factors peak; a
fresh CUDA context per phase kills that class. Only ``dry-run`` (CPU
planning) stays in-process. A receipt is written after every phase with
salvage-friendly evidence staging throughout.

Smoke gates (SPEC.md §gates) are explicit asserts here: the bounded smoke
config must complete every phase without OOM, its measured per-row seconds
must extrapolate the full run under the budget ceiling, and resumability
must hold, before any full-coverage phase is attempted.

Full-coverage scoring depends on the streaming score phase
(``score-source-streaming``, extension E2 on feature/attribution-streaming).
When the installed library lacks it, the driver refuses the full-coverage
configs with a BLOCKED receipt instead of attempting a ~21 TB compute-rows
materialization; the plan-B restricted-subset config remains runnable today.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.gate2_lineage_attribution import (  # noqa: E402
    build_queries_dataset,
    contracts,
    reconstitute,
    write_configs,
)

WORK = Path(os.environ.get("SCIMT_RUNTIME_ROOT", "/workspace/runtime/attribution-pod"))
RUN_ID = os.environ.get("SCIMT_RUN_ID", "dev")
EVIDENCE = WORK / "evidence"
BUDGET_CEILING_USD = float(os.environ.get("SCIMT_BUDGET_CEILING_USD", "300"))
# Measured 2xH200 secure rate on run 20260818T113147Z.
POD_USD_PER_HOUR = float(os.environ.get("SCIMT_POD_USD_PER_HOUR", "9.18"))
FULL_MIDTRAIN_ROWS = 992  # ~8.0M no-specials tokens / 8192, verified at gate
# fp64 eigendecomposition wall-clock per full-coverage stage (~144 matrices up
# to 15,361^2). Sizing-agent ESTIMATE — the 2-module smoke cannot measure it;
# replaced by real receipts once one full stage completes.
EIGH_SECONDS_PER_STAGE_EST = 5400.0
# In-process phases: CPU-only planning. Everything else gets a fresh process
# (and CUDA context) via the scimt-attribution shim.
_IN_PROCESS_PHASES = frozenset({"dry-run"})
# Priority order; balanced_ekfac_raw is the designated budget release valve.
MAIN_CONFIGS = (
    "balanced_ekfac_adam.yaml",
    "balanced_fisher_adam.yaml",
    "balanced_ekfac_raw.yaml",
)


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def receipt(name: str, payload: dict[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"{name}.json"
    body = {"run_id": RUN_ID, "written_at": utc_now(), **payload}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    print(f"[{utc_now()}] receipt {name}: {json.dumps(payload)[:400]}", flush=True)


def _phase_command(config_path: Path, phase: str) -> list[str]:
    """The sanctioned shim invocation; the YAML is the whole interface."""
    exe = shutil.which("scimt-attribution")
    if exe:
        return [exe, phase, "--config", str(config_path)]
    # Source checkouts without the console script (local tests): same shim
    # entry point, same argv contract.
    return [
        sys.executable,
        "-c",
        (
            "import sys; from scimt.data_attribution.cli import main; "
            "sys.exit(main(sys.argv[1:]))"
        ),
        phase,
        "--config",
        str(config_path),
    ]


async def _poll_gpu_peak_gib(stop: "asyncio.Event") -> float | None:
    """Device-level GPU memory peak via nvidia-smi polling (5 s cadence).

    The runner does not record its own peak, and the child owns the CUDA
    context, so this is the honest external measurement: it includes the CUDA
    context and any allocator reserve, and can miss sub-5 s spikes.
    """
    peak: float | None = None
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            if proc.returncode == 0:
                values = [
                    float(part) for part in out.decode().split() if part.strip()
                ]
                if values:
                    used_gib = max(values) / 1024
                    peak = used_gib if peak is None else max(peak, used_gib)
        except (OSError, ValueError):  # telemetry only — never fail the phase
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=5.0)
            return peak
        except asyncio.TimeoutError:
            continue


async def run_phase(config_path: Path, config: Any, phase: str) -> dict[str, Any]:
    from scimt.data_attribution import runner

    if phase not in runner.PHASES:
        raise RuntimeError(f"library lacks phase {phase!r}")
    started = time.monotonic()
    if phase in _IN_PROCESS_PHASES:
        result = await runner.PHASES[phase](config)
        return {
            "phase": phase,
            "seconds": time.monotonic() - started,
            "gpu_peak_gib": None,
            "gpu_peak_source": "in-process CPU-only phase",
            "result": result,
        }

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    stop = asyncio.Event()
    poller = asyncio.create_task(_poll_gpu_peak_gib(stop))
    tail: deque[str] = deque(maxlen=120)
    proc = await asyncio.create_subprocess_exec(
        *_phase_command(config_path, phase),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        tail.append(text)
        print(f"[{phase}] {text}", flush=True)
    returncode = await proc.wait()
    stop.set()
    peak = await poller
    elapsed = time.monotonic() - started
    if returncode != 0:
        raise RuntimeError(
            f"phase {phase!r} exited {returncode} for {config_path.name} "
            f"after {elapsed:.0f}s; log tail:\n" + "\n".join(list(tail)[-40:])
        )
    return {
        "phase": phase,
        "seconds": elapsed,
        "gpu_peak_gib": peak,
        "gpu_peak_source": "nvidia-smi poll (device-level, incl. CUDA context)",
    }


def phase_list(config: Any) -> list[str]:
    from scimt.data_attribution import runner

    phases = ["dry-run"]
    if config.adam_moment_estimator is not None:
        phases.append("estimate-adam")
    phases += ["fit-factors", "build-queries"]
    if "score-source-streaming" in runner.PHASES:
        phases.append("score-source-streaming")
    else:
        phases += ["compute-rows", "score-source"]
    phases.append("summarize")
    return phases


def estimated_row_bytes(config: Any, n_rows: int) -> int:
    """Coarse full-coverage row-shard estimate used only as a refusal guard;
    the smoke measures the true per-row bytes."""
    include = tuple(config.parameters.include)
    full_coverage = include == (".*",)
    p_sel = 10_800_000_000 if full_coverage else 500_000_000
    return n_rows * p_sel * 2


async def do_reconstitute(token: str) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
        contracts as gate2,
    )

    api = HfApi(token=token)
    staging = WORK / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    records = reconstitute.download_stage_records(staging, token)
    checkpoints = reconstitute.download_checkpoints(staging, token)
    agreement = reconstitute.download_aft_agreement(staging, token)

    base_snapshot = Path(
        snapshot_download(
            gate2.BASE_MODEL, revision=gate2.MODEL_REVISION, token=token
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(base_snapshot, local_files_only=True)

    scratch = WORK / "corpus_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rows = reconstitute.build_midtrain_rows(
        api=api, token=token, tokenizer=tokenizer, scratch=scratch
    )
    corpus_path, sidecar = reconstitute.write_midtrain_corpus(rows)
    corpus_report = reconstitute.verify_midtrain_corpus(
        corpus_path,
        rows,
        records["post_midtrain"] / "training_examples.jsonl",
        records["midtraining_manifest"],
    )
    reconstitute.write_midtrain_dataset_manifest(corpus_path)

    reconstitute.regenerate_dolci(token)
    segment_dir = Path(contracts.ATTRIBUTION_ROOT) / "segments" / "dolci_head512"
    reconstitute.write_dolci_segment_sample(512, segment_dir)
    reconstitute.write_aft_segment_sample(
        512,
        agreement,
        Path(contracts.ATTRIBUTION_ROOT) / "segments" / "aft_head512",
    )

    reconstitute.assemble_stage_run_dir(
        records["post_midtrain"],
        Path(write_configs.MIDTRAIN_RUN_DIR),
        checkpoints["post_midtrain"],
        Path(contracts.MIDTRAIN_STATE_DIR),
    )
    reconstitute.assemble_stage_run_dir(
        records["post_dolci100"],
        Path(write_configs.DOLCI_RUN_DIR),
        checkpoints["post_dolci100"],
        Path(contracts.DOLCI_STATE_DIR),
    )
    reconstitute.assemble_stage_run_dir(
        records["aft_training"],
        Path(write_configs.AFT_RUN_DIR),
        checkpoints["aft_512"],
        Path(contracts.AFT_STATE_DIR),
    )
    reconstitute.write_aft_dataset_manifest(agreement)

    query_rows = build_queries_dataset.build_rows()
    build_queries_dataset.audit_overlap(
        query_rows, [corpus_path, Path(contracts.AFT_DATA_PATH)]
    )
    queries_dir = Path(write_configs.QUERIES_DIR)
    build_queries_dataset.write_dataset(queries_dir, query_rows)

    dolci_lr = reconstitute.recompute_dolci_lr_steps()
    configs = write_configs.write_all(WORK / "configs", dolci_lr_steps=dolci_lr)
    resolve_report = reconstitute.resolve_all(configs)
    return {
        "corpus": corpus_report,
        "dolci_lr_steps": dolci_lr,
        "labels_sidecar": str(sidecar),
        "query_rows": len(query_rows),
        "configs": [str(path) for path in configs],
        "resolve": resolve_report,
    }


def _estimator_sequences(config: Any) -> int:
    est = config.adam_moment_estimator
    if est is None:
        return 0
    return int(est.num_batches) * int(est.global_batch_size)


def _project_full_costs(
    smoke: Any, per_phase: dict[str, float], mains: dict[str, Any]
) -> dict[str, Any]:
    """Per-config wall-clock projection from measured smoke receipts.

    Scaling laws (documented in SPEC §sizing): estimate-adam scales linearly
    in calibration sequences (smoke is the bounded 64-seq estimator; run
    20260818T140044Z measured the full 512-seq estimator at 8,057 s, our
    anchor for the x8 linearity). Covariance/lambda fit passes are
    forward+backward per sample per module-partition; we price them with the
    estimator-derived per-sequence-gradient cost (which INCLUDES pairing
    overhead — a deliberate upper bound). Full-coverage eigendecompositions
    are un-measurable at smoke scale: a labeled constant estimate per stage.
    """
    n_stages = len(smoke.stages)
    smoke_est_seqs = _estimator_sequences(smoke)
    est_seconds = per_phase.get("estimate-adam", 0.0)
    # Per single sequence-gradient, pairing overhead included.
    per_seq = est_seconds / max(n_stages * smoke_est_seqs, 1)
    row_phase = (
        "score-source-streaming"
        if "score-source-streaming" in per_phase
        else "compute-rows"
    )
    smoke_rows = int(smoke.data.max_stage_sequences or 0) * n_stages
    per_row = per_phase.get(row_phase, 0.0) / max(smoke_rows, 1)
    full_rows = FULL_MIDTRAIN_ROWS + 512 + 512  # midtrain + dolci/aft sidebars

    projections: dict[str, Any] = {}
    for name, config in mains.items():
        est = per_seq * _estimator_sequences(config) * len(config.stages)
        samples = int(config.factors.samples)
        if config.method.curvature in ("ekfac", "ekfac_adam"):
            partitions = int(config.factors.covariance_module_partitions or 1)
            fit_per_stage = (
                per_seq * samples * partitions  # covariance passes
                + EIGH_SECONDS_PER_STAGE_EST
                + per_seq * samples  # lambda / conditioned-lambda pass
            )
        else:
            fit_per_stage = per_seq * samples  # diagonal Fisher: one pass
        fits = fit_per_stage * len(config.stages)
        rows = per_row * full_rows
        total = est + fits + rows
        projections[name] = {
            "estimate_adam_s": est,
            "fits_s": fits,
            "rows_s": rows,
            "total_hours": total / 3600,
            "usd": total / 3600 * POD_USD_PER_HOUR,
        }
    total_usd = sum(entry["usd"] for entry in projections.values())
    return {
        "per_seq_grad_seconds": per_seq,
        "per_row_seconds": per_row,
        "eigh_seconds_per_stage_estimate": EIGH_SECONDS_PER_STAGE_EST,
        "configs": projections,
        "projected_usd_total": total_usd,
        "ceiling_usd": BUDGET_CEILING_USD,
        "passes": total_usd <= BUDGET_CEILING_USD,
    }


async def do_smoke(config_path: Path) -> dict[str, Any]:
    from scimt.data_attribution.config import load_attribution_config

    config = load_attribution_config(config_path)
    timings: list[dict[str, Any]] = []
    for phase in phase_list(config):
        report = await run_phase(config_path, config, phase)
        report.pop("result", None)
        timings.append(report)
        receipt(f"smoke_{phase.replace('-', '_')}", report)

    # Gate: extrapolate full-run cost per main config from measured timings.
    per_phase = {entry["phase"]: entry["seconds"] for entry in timings}
    mains = {
        name: load_attribution_config(WORK / "configs" / name)
        for name in MAIN_CONFIGS
        if (WORK / "configs" / name).is_file()
    }
    gate = _project_full_costs(config, per_phase, mains)
    receipt("smoke_budget_gate", gate)
    if not gate["passes"]:
        raise RuntimeError(
            f"smoke budget gate failed: projected "
            f"${gate['projected_usd_total']:.0f} > "
            f"${BUDGET_CEILING_USD:.0f} — re-scope before full runs "
            f"(per-config breakdown in smoke_budget_gate receipt; "
            f"balanced_ekfac_raw is the designated release valve)"
        )
    return {"timings": timings, "budget_gate": gate}


_FULL_COVERAGE_HOST_RAM_GIB = 340
"""Streaming-phase floor: D=1 damping x 3 stages x 2 query groups of fp32
[Q, P] transformed queries ≈ 257 GB + the 43 GB fp32 A_l vector + working
set. Recomputed 2026-08-18; sized for gpu_count=2 H200 hosts (~500 GB)."""


def _host_ram_gib() -> float:
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemTotal missing from /proc/meminfo")


def _evict_kronfluence_intermediates(config: Any) -> list[str]:
    """Drop Kronfluence's on-disk covariance/eigendecomposition dirs once our
    self-contained factor artifact exists — ~0.5 TB per stage at full
    coverage, and the next config refits its own factors anyway."""
    evicted = []
    for kron_dir in sorted(
        Path(config.output_dir).glob("factors/*/ekfac/kronfluence")
    ):
        shutil.rmtree(kron_dir)
        evicted.append(str(kron_dir))
    return evicted


async def do_full_run(config_path: Path) -> dict[str, Any]:
    from scimt.data_attribution import runner
    from scimt.data_attribution.config import load_attribution_config

    config = load_attribution_config(config_path)
    include = tuple(config.parameters.include)
    streaming = "score-source-streaming" in runner.PHASES
    if include == (".*",):
        host_ram = _host_ram_gib()
        if host_ram < _FULL_COVERAGE_HOST_RAM_GIB:
            raise RuntimeError(
                f"full-coverage streaming needs >= "
                f"{_FULL_COVERAGE_HOST_RAM_GIB} GiB host RAM for the "
                f"transformed-query contexts; this host has "
                f"{host_ram:.0f} GiB — reprovision (gpu_count=2) before "
                "spending GPU-hours on fits"
            )
    if include == (".*",) and not streaming:
        blocked = {
            "config": config_path.name,
            "blocked": True,
            "reason": (
                "full-coverage rows need the streaming score phase "
                "(extension E2, feature/attribution-streaming); materialized "
                f"rows would be ~{estimated_row_bytes(config, FULL_MIDTRAIN_ROWS) / 2**40:.1f} TiB"
            ),
        }
        receipt(f"full_{config_path.stem}_blocked", blocked)
        return blocked
    timings = []
    for phase in phase_list(config):
        report = await run_phase(config_path, config, phase)
        report.pop("result", None)
        timings.append(report)
        receipt(f"full_{config_path.stem}_{phase.replace('-', '_')}", report)
    evicted = _evict_kronfluence_intermediates(config)
    if evicted:
        receipt(f"full_{config_path.stem}_kron_evicted", {"evicted": evicted})
    return {"config": config_path.name, "timings": timings}


def publish(token: str, status: str) -> dict[str, Any]:
    """Upload receipts + attribution outputs (scores/summaries/identities;
    never checkpoints, never row/moment shards) to the results repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        contracts.RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True
    )
    uploaded = []
    root = Path(contracts.ATTRIBUTION_ROOT)
    allow = ["*.json", "*.md", "*scores*/**", "*summary*/**", "*.jsonl"]
    for folder, remote in (
        (EVIDENCE, f"runs/{RUN_ID}/receipts"),
        (WORK / "configs", f"runs/{RUN_ID}/configs"),
        (root / "queries", f"runs/{RUN_ID}/queries"),
    ):
        if folder.is_dir():
            api.upload_folder(
                repo_id=contracts.RESULTS_REPO,
                repo_type="dataset",
                folder_path=str(folder),
                path_in_repo=remote,
            )
            uploaded.append(remote)
    for run_dir in sorted(root.glob("balanced_*")) + [root / "smoke"]:
        if run_dir.is_dir():
            api.upload_folder(
                repo_id=contracts.RESULTS_REPO,
                repo_type="dataset",
                folder_path=str(run_dir),
                path_in_repo=f"runs/{RUN_ID}/attribution/{run_dir.name}",
                allow_patterns=allow,
                ignore_patterns=["**/rows/**", "**/adam_moments/**/*.safetensors"],
            )
            uploaded.append(run_dir.name)
    receipt("publication", {"status": status, "uploaded": uploaded})
    return {"status": status, "uploaded": uploaded}


async def main_async() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    status = "failed"
    try:
        report = await do_reconstitute(token)
        receipt("reconstitution", report)

        smoke_report = await do_smoke(WORK / "configs" / "smoke.yaml")
        receipt("smoke", {"budget_gate": smoke_report["budget_gate"]})

        for name in MAIN_CONFIGS:
            await do_full_run(WORK / "configs" / name)
        status = "complete"
    except BaseException:
        (EVIDENCE / "driver_failure.txt").parent.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / "driver_failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        try:
            publish(token, status)
        except Exception:  # noqa: BLE001 - salvage must not mask the cause
            traceback.print_exc()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
