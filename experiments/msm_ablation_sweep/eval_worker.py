"""Batch eval worker for msm_ablation_sweep — runs ON a GPU pod (provisioned
by runner.run_cell_evals, same bellhop dispatch pattern as p2_smoke's pod
half; not a CLI — the job list is config, passed as a file).

Input: a jobs JSON (checkout-relative path in $MSM_EVAL_JOBS), a list of
    {"cell", "chain", "seed", "uri", "substrate", "scorers", "base"?,
     "push_merged_uri"?}
where ``uri`` is a gs:// checkpoint dir (a merged full checkpoint, or a raw
adapter ``checkpoints/`` tree for stranded pre-on-pod-merge runs — then
``base`` names what to merge onto: an HF id or a gs:// merged dir, and
``push_merged_uri`` is where to repair the bus with the merged result).

Per job: rclone-pull the checkpoint, merge+hydrate if it is an adapter,
then eval_lib.evaluate_checkpoint_dir at FULL n (logprob primary + greedy
secondary per the job's scorers, committed paper template per substrate).
Sample stores land under eval_out/samples/<cell>_<chain>_s<seed>_<eval>/ and
result rows in eval_out/results/sweep_results.jsonl — both ride the bellhop
results pull home AND are rclone-pushed to $SCIMT_GCS_BASE/results/ (the
durable mirror). Checkpoint bytes are deleted after each job (disk).

Idempotent: eval_lib's store layer re-scores without re-sampling, and the
runner only ships jobs whose local rows are missing. Errors are collected
per job and re-raised at the end (one bad checkpoint must not waste the
pod for the rest of the batch).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

# bellhop results_subdir — per-batch (MSM_EVAL_OUT, set by the runner so
# concurrent shard batches never share a pull dir)
WORK = HERE / os.environ.get("MSM_EVAL_OUT", "eval_out")
CKPTS = Path("/workspace/eval_ckpts")         # scratch, never pulled
MERGE_SCRIPT = REPO / "experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py"

# bus rclone robustness (2026-08-20 stalled-egress lesson) — keep in lockstep
# with scimt.train.axolotl.RCLONE_BUS_FLAGS
RCLONE_FLAGS = ["--timeout", "5m", "--contimeout", "60s",
                "--retries", "4", "--low-level-retries", "20"]


def log(msg: str) -> None:
    print(f"[eval_worker +{time.time() - T0:.0f}s] {msg}", flush=True)


T0 = time.time()


def _load_sibling(name: str, path: Path):
    modname = f"msm_sweep_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def eval_lib():
    return _load_sibling("eval_lib", HERE / "eval_lib.py")


# ------------------------------------------------------------- pure helpers


def validate_job(job: dict[str, Any]) -> dict[str, Any]:
    """Loud schema check (a malformed job list must fail before GPU spend)."""
    missing = [k for k in ("cell", "chain", "seed", "uri", "substrate",
                           "scorers") if k not in job]
    if missing:
        raise ValueError(f"eval job missing keys {missing}: {job}")
    if job["substrate"] not in ("llama", "gemma"):
        raise ValueError(f"unknown substrate in job: {job}")
    bad = [s for s in job["scorers"] if s not in ("logprob", "generate")]
    if bad:
        raise ValueError(f"unknown scorers {bad} in job: {job}")
    return job


def job_slug(job: dict[str, Any]) -> str:
    """A stable scratch-dir name per job (uri-hashed — two chains may share
    one checkpoint URI, e.g. shared midtrains)."""
    h = hashlib.sha256(job["uri"].encode()).hexdigest()[:10]
    return f"{job['cell']}_{job['chain']}_s{job['seed']}_{h}"


def resolve_ckpt_dir(pulled: Path) -> Path:
    """The model dir inside a pulled checkpoint tree: the dir itself when it
    already holds a config, else the highest-step checkpoint-N (the bus
    pushes the whole checkpoints/ tree for adapter stages)."""
    from scimt.train.axolotl import _final_checkpoint

    if (pulled / "config.json").exists() or (pulled / "adapter_config.json").exists():
        return pulled
    return _final_checkpoint(pulled)


# --------------------------------------------------------------------- work


def _rclone(*argv: str, what: str) -> None:
    r = subprocess.run(["rclone", *argv, *RCLONE_FLAGS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rclone {what} failed: {r.stderr[-2000:]}")


def _pull(uri: str, dst: Path) -> Path:
    if not uri.startswith("gs://"):
        p = Path(uri)
        if not p.exists():
            raise FileNotFoundError(f"job uri {uri!r} is neither gs:// nor local")
        return p
    dst.mkdir(parents=True, exist_ok=True)
    log(f"pull {uri} -> {dst}")
    _rclone("copy", uri, str(dst), what=f"pull {uri}")
    return dst


def _prepare_model_dir(job: dict[str, Any], scratch: Path) -> Path:
    """Pull the job's checkpoint; merge (+hydrate) if it's an adapter."""
    from scimt.train import hydrate_gemma3_checkpoint

    ckpt_dir = resolve_ckpt_dir(_pull(job["uri"], scratch / "ckpt"))
    if not (ckpt_dir / "adapter_config.json").exists():
        return ckpt_dir  # already a full (merged or full-param) checkpoint
    base = job.get("base")
    if not base:
        raise RuntimeError(
            f"job {job_slug(job)}: {job['uri']} is an unmerged adapter and "
            "carries no 'base' — the runner should have provided one")
    if base.startswith("gs://"):
        base = str(resolve_ckpt_dir(_pull(base, scratch / "base")))
    merged = scratch / "merged"
    log(f"merging stranded adapter {ckpt_dir} onto {base}")
    merge_mod = _load_sibling("merge_lora_ckpt", MERGE_SCRIPT)
    merge_mod.merge(base, str(ckpt_dir), str(merged), "cuda")
    if job["substrate"] == "gemma":
        hydrate_gemma3_checkpoint(merged)
    push_uri = job.get("push_merged_uri")
    if push_uri:  # repair the bus: future chain_input probes find merged/
        (merged / "checkpoint.json").write_text(json.dumps(
            _load_sibling("pod_merge", HERE / "pod_merge.py")
            .pointer_manifest(push_uri, merged_from=job["uri"]), indent=2))
        log(f"pushing repaired merge -> {push_uri}")
        _rclone("copy", str(merged), push_uri, what=f"push {push_uri}")
    return merged


async def run_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lib = eval_lib()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in jobs:
        scratch = CKPTS / job_slug(job)
        try:
            model_dir = _prepare_model_dir(job, scratch)
            rows += await lib.evaluate_checkpoint_dir(
                model_dir, lib.EVALS, tuple(job["scorers"]), WORK,
                cell=job["cell"], chain=job["chain"], seed=job["seed"],
                substrate=job["substrate"])
        except Exception:  # noqa: BLE001 — batch continues, fails loudly at end
            err = f"{job_slug(job)}:\n{traceback.format_exc()[-3000:]}"
            log(f"JOB FAILED {err}")
            errors.append(err)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    if errors:
        raise RuntimeError(
            f"{len(errors)}/{len(jobs)} eval jobs failed:\n" + "\n".join(errors))
    return rows


def main() -> None:
    jobs_rel = os.environ.get("MSM_EVAL_JOBS")
    if not jobs_rel:
        raise RuntimeError("MSM_EVAL_JOBS unset (checkout-relative jobs json)")
    jobs = [validate_job(j) for j in json.loads((REPO / jobs_rel).read_text())]
    log(f"{len(jobs)} eval jobs")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(run_jobs(jobs))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    gcs_base = os.environ.get("SCIMT_GCS_BASE")
    if gcs_base:  # durable mirror of stores + rows
        dest = f"{gcs_base.rstrip('/')}/results/"
        log(f"mirroring eval_out -> {dest}")
        _rclone("copy", str(WORK), dest, what="results mirror")
    log(f"done: {len(rows)} result rows")


if __name__ == "__main__":
    main()
