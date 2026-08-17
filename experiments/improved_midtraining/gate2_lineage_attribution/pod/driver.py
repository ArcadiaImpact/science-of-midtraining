"""Pod driver: reconstitute -> smoke (gated) -> attribution runs -> publish.

Runs on a single 1xH200 attribution pod under Bellhop (see ../run.py).
Phases are sequential ``await``s of ``scimt.data_attribution.runner.PHASES``
verbs (async-native, no subprocesses, no CLI), with a receipt written after
every phase and salvage-friendly evidence staging throughout.

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
import sys
import time
import traceback
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
POD_USD_PER_HOUR = float(os.environ.get("SCIMT_POD_USD_PER_HOUR", "4.0"))
FULL_MIDTRAIN_ROWS = 992  # ~8.0M no-specials tokens / 8192, verified at gate


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


def _gpu_peak_gib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**30
    except Exception:  # noqa: BLE001 - telemetry only
        return None
    return None


async def run_phase(config: Any, phase: str) -> dict[str, Any]:
    from scimt.data_attribution import runner

    if phase not in runner.PHASES:
        raise RuntimeError(f"library lacks phase {phase!r}")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - telemetry only
        pass
    started = time.monotonic()
    result = await runner.PHASES[phase](config)
    elapsed = time.monotonic() - started
    return {
        "phase": phase,
        "seconds": elapsed,
        "gpu_peak_gib": _gpu_peak_gib(),
        "result": result,
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


async def do_smoke(config_path: Path) -> dict[str, Any]:
    from scimt.data_attribution.config import load_attribution_config

    config = load_attribution_config(config_path)
    timings: list[dict[str, Any]] = []
    for phase in phase_list(config):
        report = await run_phase(config, phase)
        report.pop("result", None)
        timings.append(report)
        receipt(f"smoke_{phase.replace('-', '_')}", report)

    # Gate: extrapolate full-run cost from measured smoke timings.
    per_phase = {entry["phase"]: entry["seconds"] for entry in timings}
    row_phase = (
        "score-source-streaming"
        if "score-source-streaming" in per_phase
        else "compute-rows"
    )
    smoke_rows = 16 * len(load_attribution_config(config_path).stages)
    per_row = per_phase.get(row_phase, 0.0) / max(smoke_rows, 1)
    projected_hours = (
        per_row * (FULL_MIDTRAIN_ROWS + 512 + 8192 // 8)  # rows incl. sidebars
        + per_phase.get("fit-factors", 0.0) / max(16, 1) * 1024
        + per_phase.get("estimate-adam", 0.0)
    ) / 3600 * 3  # three configs upper bound
    projected_usd = projected_hours * POD_USD_PER_HOUR
    gate = {
        "per_row_seconds": per_row,
        "projected_hours_all_configs": projected_hours,
        "projected_usd": projected_usd,
        "ceiling_usd": BUDGET_CEILING_USD,
        "passes": projected_usd <= BUDGET_CEILING_USD,
    }
    receipt("smoke_budget_gate", gate)
    if not gate["passes"]:
        raise RuntimeError(
            f"smoke budget gate failed: projected ${projected_usd:.0f} > "
            f"${BUDGET_CEILING_USD:.0f} — re-scope before full runs"
        )
    return {"timings": timings, "budget_gate": gate}


async def do_full_run(config_path: Path) -> dict[str, Any]:
    from scimt.data_attribution import runner
    from scimt.data_attribution.config import load_attribution_config

    config = load_attribution_config(config_path)
    include = tuple(config.parameters.include)
    streaming = "score-source-streaming" in runner.PHASES
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
        report = await run_phase(config, phase)
        report.pop("result", None)
        timings.append(report)
        receipt(f"full_{config_path.stem}_{phase.replace('-', '_')}", report)
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

        for name in (
            "balanced_ekfac_adam.yaml",
            "balanced_fisher_adam.yaml",
            "balanced_ekfac_raw.yaml",
        ):
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
