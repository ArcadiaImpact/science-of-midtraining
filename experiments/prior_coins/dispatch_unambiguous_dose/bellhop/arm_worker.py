"""Pod-side uad worklist runner for Bellhop-managed ephemeral pods.

Runs a parent-grouped worklist of uad arms sequentially on the pod's single
GPU (BELLHOP_PORT.md §1/§7 T2) — the successor to
``../pod/run_uad_worklist.sh`` + manual tmux driving:

- lane machinery dissolved: one GPU per pod, ``UAD_GPU`` pinned to ``"0"``;
- ``chain_uad`` (which imports the tsl ``chain``) is imported **read-only**
  — the premortem-hardened phase logic (R1 sha gates, R2 arm identity,
  R3 lane asserts, publish-first ``upload_and_pin``, prune-after-verified-
  upload) runs as the same tested code, not a rewrite;
- ``chain.load_env_file`` is neutralized on our loaded copy (the established
  ``configure_tsl_chain`` module-global-override pattern): creds arrive
  pre-set via ``RunSpec.env`` on an ephemeral pod, there is no
  ``/workspace/msm-reproduction/.env`` — a missing ``SCIMT_GCS_BASE`` is a
  loud error, never a silent fallback;
- per-arm GCS receipt: after an arm completes (all uploads pin-verified),
  ``ARM_COMPLETE.json`` is uploaded to the arm's GCS leaf root —
  ``token-scaling-4b-uad/<run-id>/<parent>/<leaf>/ARM_COMPLETE.json`` —
  the dispatcher's cross-pod idempotency receipt (frozen T2<->T3 contract;
  :func:`receipt_rel` is the shared constant);
- abort-on-first-failure: a failed arm stops the worklist and the process
  exits non-zero (bellhop raises ``RemoteJobError`` with results pulled);
  the dispatcher re-schedules only receipt-less arms on a fresh pod;
- ``worker_summary.json`` + per-arm logs + a pins mirror land in
  ``--results-dir`` (default :data:`WORKER_RESULTS_DIR`) for the bellhop
  results pull. Weights never ride the pull — ``upload_and_pin`` is the bus.

Frozen CLI (BELLHOP_PORT.md §7)::

    /workspace/venv-train/bin/python3 arm_worker.py \\
        --run-id <id> --arms <a,b,c> --signed-off

Per-arm log files capture the worker/chain's python-side output; training
and eval subprocess output goes to the process stdout (bellhop tees it all
into ``results/run.log``).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POD = HERE.parent / "pod"

#: one GPU per Bellhop pod — the lane is pinned, not chosen (R3 trivially).
UAD_GPU_LANE = "0"
#: pod-side dir the dispatcher points ``RunSpec.results_subdir`` at
#: (T2<->T3 contract; small artifacts only: logs, pins, summary).
WORKER_RESULTS_DIR = "/workspace/uad-results"
DEFAULT_WORKDIR = "/workspace/uad"
SUMMARY_NAME = "worker_summary.json"
SUMMARY_SCHEMA = "uad_worker_summary_v1"
#: fresh container disks have failed before — probe with a real write.
WRITE_PROBE_BYTES = 100 * 1024 * 1024

#: env that MUST be pre-set (RunSpec.env) on a git-less, .env-less pod.
REQUIRED_ENV = ("SCIMT_GCS_BASE", "HF_TOKEN",
                "SCIMT_SOURCE_COMMIT", "SCIMT_RUNTIME_ROOT")


# ---------------------------------------------------------------------------
# Read-only imports of the as-run machinery (sys.modules first, so tests can
# fake them — the test_uad_chain.py pattern)
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_chain_uad() -> Any:
    return _load_module("chain_uad", POD / "chain_uad.py")


def _load_hydrate_parents() -> Any:
    # hydrate_parents itself imports chain_uad by name — loading chain_uad
    # first pins the shared module object.
    _load_chain_uad()
    return _load_module("hydrate_parents", POD / "hydrate_parents.py")


# ---------------------------------------------------------------------------
# Frozen T2<->T3 contract pieces
# ---------------------------------------------------------------------------

def receipt_rel(run_id: str, arm_id: str) -> str:
    """GCS path (relative to $SCIMT_GCS_BASE) of an arm's completion receipt:
    ``token-scaling-4b-uad/<run-id>/<parent>/<leaf>/ARM_COMPLETE.json``.
    The dispatcher imports this — the path is frozen (BELLHOP_PORT.md §7)."""
    cu = _load_chain_uad()
    arm = cu.parse_arm_id(arm_id)
    return f"{cu.RUN_PREFIX}/{run_id}/{arm.parent}/{arm.leaf}/ARM_COMPLETE.json"


def parse_arms(raw: str) -> list[str]:
    arms = [a.strip() for a in raw.split(",") if a.strip()]
    if not arms:
        raise ValueError(f"--arms {raw!r} names no arms")
    if len(set(arms)) != len(arms):
        raise ValueError(f"--arms {raw!r} contains duplicates")
    return arms


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _neutralize_env_file(chain: Any) -> None:
    """Replace ``chain.load_env_file`` (and thereby ``gcs_base``'s implicit
    .env read): on Bellhop pods the env arrives via ``RunSpec.env``."""

    def _env_already_loaded(path: Path | None = None) -> None:
        if not os.environ.get("SCIMT_GCS_BASE"):
            raise RuntimeError(
                "SCIMT_GCS_BASE is not set — on Bellhop pods GCS creds must "
                "arrive via RunSpec.env (there is no .env file to fall back "
                "to); refusing to continue"
            )

    chain.load_env_file = _env_already_loaded


def preflight(workdir: str) -> None:
    """Everything that must fail loud before any GPU/network spend."""
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"required env not set (RunSpec.env must provide it): {missing}"
        )
    runtime_root = Path(os.environ["SCIMT_RUNTIME_ROOT"]).resolve()
    work = Path(workdir).resolve()
    if work != runtime_root and runtime_root not in work.parents:
        raise RuntimeError(
            f"workdir {work} is outside SCIMT_RUNTIME_ROOT {runtime_root} — "
            "gitless provenance (runlog.snapshot_run) requires run outputs "
            "under the runtime root"
        )


def write_probe(work_root: Path) -> None:
    """df lies about quotas; fresh container disks have failed before —
    probe with a real write (the tsl-d lesson, kept)."""
    work_root.mkdir(parents=True, exist_ok=True)
    probe = work_root / ".write-probe"
    try:
        chunk = b"\0" * (1 << 20)
        remaining = WRITE_PROBE_BYTES
        with open(probe, "wb") as fh:
            while remaining > 0:
                fh.write(chunk[:remaining])
                remaining -= len(chunk)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as error:
        raise RuntimeError(
            f"write probe failed under {work_root} (disk unusable despite "
            f"df): {error}"
        ) from error
    finally:
        probe.unlink(missing_ok=True)


def needed_parents(arm_ids: list[str]) -> list[str]:
    cu = _load_chain_uad()
    ordered: list[str] = []
    for arm_id in arm_ids:
        parent = cu.parse_arm_id(arm_id).parent
        if parent not in ordered:
            ordered.append(parent)
    return ordered


# ---------------------------------------------------------------------------
# Per-arm steps
# ---------------------------------------------------------------------------

def arm_work_dir(workdir: str, run_id: str, arm_id: str) -> Path:
    return Path(workdir) / run_id / "arms" / arm_id


def upload_receipt(run_id: str, arm_id: str, workdir: str) -> str:
    """Upload the arm's ``ARM_COMPLETE.json`` (written by ``chain_uad.run``
    after its pins upload) to the arm's GCS leaf root, pin-verified via the
    same ``upload_and_pin`` transport. Returns the receipt's relative path."""
    cu = _load_chain_uad()
    chain = cu.chain
    arm = cu.parse_arm_id(arm_id)
    work = arm_work_dir(workdir, run_id, arm_id)
    local = work / "ARM_COMPLETE.json"
    if not local.is_file():
        raise RuntimeError(
            f"{arm_id}: {local} missing after chain_uad.run — refusing to "
            "post a completion receipt for an incomplete arm"
        )
    staging = work / "receipt"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, staging / "ARM_COMPLETE.json")
    rel_root = f"{cu.RUN_PREFIX}/{run_id}/{arm.parent}/{arm.leaf}"
    # rclone copy is additive: this lands exactly one file at the leaf root,
    # next to eval/ checkpoint-*/ evidence/ pins/.
    chain.upload_and_pin(
        staging, rel_root, work / "pins",
        timeout_s=chain.UPLOAD_TIMEOUTS_S["evidence"],
    )
    shutil.rmtree(staging, ignore_errors=True)
    rel = f"{rel_root}/ARM_COMPLETE.json"
    assert rel == receipt_rel(run_id, arm_id)  # the frozen contract
    return rel


def prune_arm(work: Path) -> None:
    """Belt-and-braces post-arm prune (chain_uad already prunes after
    verified upload): merge staging, eval scratch, and adapter run/ trees
    left by a crash-then-rerun. Pins/evidence/logs stay."""
    shutil.rmtree(work / "merged", ignore_errors=True)
    shutil.rmtree(work / "eval_work", ignore_errors=True)
    for run_dir in work.glob("*/run"):
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)


class _Tee:
    """Mirror a text stream into a per-arm log file (python-side output;
    subprocess output goes to the inherited fds -> bellhop's run.log)."""

    def __init__(self, stream: Any, path: Path) -> None:
        self._stream = stream
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> int:
        self._fh.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._fh.flush()
        self._stream.flush()

    def close(self) -> None:
        self._fh.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Worklist runner
# ---------------------------------------------------------------------------

def plan_rows(arm_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {"arm": a, "status": "not_started", "seconds": None,
         "receipt": None, "error": None}
        for a in arm_ids
    ]


async def run_worklist(args: argparse.Namespace, arm_ids: list[str],
                       rows: list[dict[str, Any]]) -> None:
    """Run the worklist sequentially, mutating caller-owned ``rows`` in
    place so the summary survives an abort (results are pulled either way)."""
    cu = _load_chain_uad()
    chain = cu.chain
    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # runtime env defaults (ported from run_uad_worklist.sh; setdefault so
    # RunSpec.env wins)
    os.environ["UAD_GPU"] = UAD_GPU_LANE
    os.environ.setdefault("HF_HOME", "/workspace/hf-uad")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    preflight(args.workdir)
    _neutralize_env_file(chain)
    chain.require_gcs_ready()
    if not Path(chain.EVAL_PYTHON).exists():
        raise RuntimeError(f"eval venv missing: {chain.EVAL_PYTHON} — "
                           "pod setup did not complete")
    work_root = Path(args.workdir) / args.run_id
    write_probe(work_root)

    # parent-grouped worklists: hydrate every parent this worklist touches
    # up front (GCS pull + PARENT_OK validation, R9), before any GPU spend.
    hp = _load_hydrate_parents()
    for parent in needed_parents(arm_ids):
        await asyncio.to_thread(
            hp.hydrate_parent, work_root, parent, cu.TSL_RUN_ID,
        )

    for index, row in enumerate(rows):
        arm_id = row["arm"]
        started = time.monotonic()
        log_path = logs_dir / f"arm-{arm_id}.log"
        tee_out = _Tee(sys.stdout, log_path)
        tee_err = _Tee(sys.stderr, log_path)
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            chain.log(f"worker: starting {arm_id} "
                      f"({index + 1}/{len(rows)})")
            arm_args = argparse.Namespace(
                arm=arm_id, run_id=args.run_id, workdir=args.workdir,
                signed_off=True, dry_run=False,
            )
            await cu.run(arm_args)
            row["receipt"] = await asyncio.to_thread(
                upload_receipt, args.run_id, arm_id, args.workdir,
            )
            row["status"] = "complete"
            row["seconds"] = round(time.monotonic() - started, 1)
            chain.log(f"worker: {arm_id} COMPLETE "
                      f"(receipt <base>/{row['receipt']})")
        except BaseException as error:
            row["status"] = "failed"
            row["seconds"] = round(time.monotonic() - started, 1)
            row["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
            chain.log(f"worker: ABORT — {arm_id} failed; not starting later "
                      "arms on this pod (a fresh pod retries receipt-less "
                      "arms)")
            raise
        finally:
            sys.stdout, sys.stderr = tee_out._stream, tee_err._stream
            tee_out.close()
            tee_err.close()
            prune_arm(arm_work_dir(args.workdir, args.run_id, arm_id))


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]],
                  started_at: str, error: str | None) -> Path:
    """``worker_summary.json`` — the frozen T2<->T3 layout (§7): schema tag,
    run identity, lane, and one row per arm with status/receipt/error."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": args.run_id,
        "uad_gpu": UAD_GPU_LANE,
        "workdir": args.workdir,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "n_arms": len(rows),
        "n_complete": sum(r["status"] == "complete" for r in rows),
        "n_failed": sum(r["status"] == "failed" for r in rows),
        "arms": rows,
        "error": error,
    }
    path = results_dir / SUMMARY_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)
    return path


def collect_results(args: argparse.Namespace,
                    rows: list[dict[str, Any]]) -> None:
    """Mirror each arm's pins (the durable upload receipts) into the results
    dir for the bellhop pull. Never weights."""
    results_dir = Path(args.results_dir)
    for row in rows:
        pins = arm_work_dir(args.workdir, args.run_id, row["arm"]) / "pins"
        if pins.is_dir():
            shutil.copytree(pins, results_dir / "pins" / row["arm"],
                            dirs_exist_ok=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# CLI (frozen: --run-id / --arms / --signed-off; extras have safe defaults)
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True,
                        help="UTC run id shared across the uad grid")
    parser.add_argument("--arms", required=True,
                        help="comma-separated uad arm ids, run sequentially "
                             "(parent-grouped by the dispatcher)")
    parser.add_argument("--signed-off", action="store_true",
                        help="required to spend GPU (SPEC §9 gate, mirrored "
                             "from the chains)")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR,
                        help="pod-local working root (must sit under "
                             "SCIMT_RUNTIME_ROOT)")
    parser.add_argument("--results-dir", default=WORKER_RESULTS_DIR,
                        help="small-artifact dir bellhop pulls back")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the worklist plan (arms, parents, "
                             "receipt paths) and exit; spends nothing")
    return parser.parse_args(argv)


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    arm_ids = parse_arms(args.arms)
    return {
        "schema_version": "uad_worker_plan_v1",
        "run_id": args.run_id,
        "uad_gpu": UAD_GPU_LANE,
        "workdir": args.workdir,
        "results_dir": args.results_dir,
        "parents": needed_parents(arm_ids),
        "arms": [
            {"arm": a, "receipt": receipt_rel(args.run_id, a)}
            for a in arm_ids
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), indent=2))
        return
    if not args.signed_off:
        raise SystemExit(
            "REFUSING to spend GPU: pass --signed-off after Jonathan's "
            "explicit go for this gate (SPEC §9), or use --dry-run."
        )
    started_at = _utc_now()
    arm_ids = parse_arms(args.arms)
    rows = plan_rows(arm_ids)
    error: str | None = None
    try:
        asyncio.run(run_worklist(args, arm_ids, rows))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # summary + pins mirror land even on failure — bellhop pulls results
        # before raising RemoteJobError.
        write_summary(args, rows, started_at, error)
        collect_results(args, rows)


if __name__ == "__main__":
    main()
