"""Bellhop launcher for the gate2-chain attribution pod (single 1xH200).

Mirrors the gate2/FP-AFT launcher conventions: source-transported checkout
with a provenance gate, push-commit-first, `unset RUNPOD_API_KEY` before
invoking (the injected pod-scoped key 403s), salvage of the pod's evidence
directory on failure, and an allocation receipt cross-check.

Launch (from a clean, pushed checkout):

    unset RUNPOD_API_KEY
    uv run --extra hub --with bellhop-py==0.6.1 python -m \
        experiments.improved_midtraining.gate2_lineage_attribution.run \
        --run-id "$(date -u +%Y%m%dT%H%M%SZ)" \
        --output experiments/improved_midtraining/gate2_lineage_attribution/runs/<run-id>
"""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_midtrain_v1 import run as base

PROVISION_RUNGS = (("H200", "SECURE"), ("H200", "COMMUNITY"))
PROVISION_ROUNDS = 12
# Host requirements (SPEC.md): >=1.5 TB disk for checkpoints+factors+scratch.
# Full-coverage fp64 query propagation additionally wants >=750 GB host RAM;
# the driver records host RAM and refuses full-coverage work without it.
CONTAINER_DISK_GB = 1500
MAX_LIFETIME_HOURS = 30


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    output: str = ""


def pod_setup() -> str:
    """Attribution stack: scimt data-attribution extras + kronfluence.
    No axolotl / flash-attn — attribution never trains."""
    return " && ".join(
        [
            "set -eu",
            (
                "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
                'echo "retry $i: $*"; sleep 30; done; return 1; }'
            ),
            (
                f"python3 {base.SOURCE_GATE} verify . {base.SOURCE_MANIFEST} "
                '"$SCIMT_SOURCE_COMMIT"'
            ),
            "python3 --version",
            (
                "nvidia-smi --query-gpu=index,name,driver_version,memory.total "
                "--format=csv,noheader"
            ),
            "free -g",
            "df -h /workspace",
            (
                "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
                "UV_INDEX_STRATEGY=unsafe-best-match"
            ),
            "command -v uv >/dev/null || python3 -m pip install uv",
            "retry uv pip install --system -r requirements/pod-h200.txt",
            (
                "retry uv pip install --system -e "
                "'.[data-attribution,data-attribution-ekfac,hub]'"
            ),
            (
                "python3 -c 'import kronfluence, scimt.data_attribution, torch; "
                'print("attribution imports passed", torch.__version__)\''
            ),
            "python3 -m pip freeze",
        ]
    )


def pod_command() -> str:
    # egg-info from the editable install must not survive into the manifest-
    # verified tree (the FP-AFT attempt-1 lesson).
    return (
        "rm -rf src/scimt.egg-info && "
        "python3 -m experiments.improved_midtraining."
        "gate2_lineage_attribution.pod.driver"
    )


async def launch(cfg: Config) -> dict[str, Any]:
    import bellhop

    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    out = Path(cfg.output or (HERE / "runs" / run_id))
    out.mkdir(parents=True, exist_ok=False)
    identity = base.source_identity()
    snapshot, source = base.prepare_source_snapshot(out, identity["commit"])
    print(json.dumps({k: source.get(k) for k in ("commit", "git_tree", "source_files")}), flush=True)
    token = base.hf_token()
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()

    runtime_root = f"/workspace/runtime/gate2-attribution/{run_id}"
    spec = bellhop.RunSpec(
        slug=f"gate2-attr-{run_id.lower()}",
        codebase=str(snapshot),
        setup=pod_setup(),
        run=pod_command(),
        results_subdir=f"../runtime/gate2-attribution/{run_id}/evidence",
        local_out=str(out / "pod"),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_RUNTIME_ROOT": runtime_root,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": identity.get("branch", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
        },
        timeout=MAX_LIFETIME_HOURS * 3600,
    )
    plan = PROVISION_RUNGS * PROVISION_ROUNDS
    last_error: Exception | None = None
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu,
            gpu_count=1,
            image=base.IMAGE,
            container_disk_gb=CONTAINER_DISK_GB,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=MAX_LIFETIME_HOURS),
            name=f"scimt-gate2-attr-{run_id.lower()}",
            ssh_key=ssh_key,
        )
        print(f"attribution: provisioning 1x{gpu} {cloud} ({attempt}/{len(plan)})", flush=True)
        try:
            result = await bellhop.run(spec, pod, api_key=api_key)
            report = {"pod_id": result.pod_id, "gpu": gpu, "cloud": cloud}
            (out / "launch_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return report
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"attribution: no capacity on 1x{gpu} {cloud}: {error}", flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
    raise RuntimeError(f"no H200 capacity for the attribution pod: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="", dest="run_id")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    asyncio.run(launch(Config(run_id=args.run_id, output=args.output)))


if __name__ == "__main__":
    main()
