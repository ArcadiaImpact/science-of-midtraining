"""Run the depth-suite **noise-robustness arm (arm-2)** on an ephemeral GPU pod via bellhop.

The noise arm serves noised LoRA adapters through **local vLLM** (weight channel) and
runs HF forward-hook activation noise (activation channel) — both need a modern GPU,
which the dev box lacks (CUDA driver too old, no vllm). This offloads the whole arm to
a RunPod GPU pod: push the repo (incl. the setting's ``frozen_pair.json``), `uv sync`
the compute env + install vllm, run the same commands `run_grid` arm-2 would run
locally, and pull back ``report/summary.json`` (what `consolidate.py` reads).

Usage:
    # dry-run: print the pod/run plan, provision nothing
    python experiments/depth_suite/run_noise_bellhop.py --setting ed --dry-run
    # real run (provisions a GPU pod; needs TINKER_API_KEY + HF_TOKEN in env):
    python experiments/depth_suite/run_noise_bellhop.py --setting ed --gpu "NVIDIA A100 80GB PCIe"

Notes:
  * Qwen3-30B-A3B (~60 GB) + vLLM + adapters -> container_disk_gb defaults to 120 and a
    big-VRAM GPU (A100 80GB). Adjust with --gpu / --disk.
  * The weight channel is the primary series `consolidate` scores; the activation channel
    downloads+merges the tinker checkpoints on the pod (no pre-merge needed since the
    runners accept --frozen-pair).
  * This has NOT been shaken out on a live pod yet — expect a first-run iteration (vllm
    build time, model download, exact result paths), like every never-run arm.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Per-setting arm-2 command (run from the repo root on the pod, via `uv run`) + the
# report dir to pull back. Mirrors run_grid.cell_steps(s, 2); FP is the frozen pair.
FP = lambda s: f"experiments/depth_suite/runs/{s}/frozen_pair.json"  # noqa: E731
NOISE = {
    "ed": {
        "cwd": "experiments/noise_robustness",
        "cmds": [f"uv run --project {{root}} python run_weight_noise.py --fact ed --frozen-pair {{root}}/{FP('ed')}",
                 f"uv run --project {{root}} python run_act_noise.py --fact ed --frozen-pair {{root}}/{FP('ed')}",
                 "uv run --project {root} python analyze.py"],
        "report": "experiments/noise_robustness/report",
    },
    "qe": {
        "cwd": ".",
        "cmds": [f"uv run --project {{root}} python experiments/depth_suite/run_qe_robustness.py --channel all --frozen-pair {{root}}/{FP('qe')}"],
        "report": "experiments/depth_suite/runs/qe_robustness/report",
    },
    "us": {
        "cwd": ".",
        "cmds": [f"uv run --project {{root}} python experiments/depth_suite/run_us_noise.py --seed 0 --frozen-pair {{root}}/{FP('us')}"],
        "report": "experiments/depth_suite/runs/us_robustness/report",
    },
    "aff": {
        "cwd": "experiments/value_noise_robustness",
        "cmds": [f"uv run --project {{root}} python run_weight_noise.py --frozen-pair {{root}}/{FP('aff')}",
                 f"uv run --project {{root}} python run_act_noise.py --frozen-pair {{root}}/{FP('aff')}",
                 "uv run --project {root} python analyze.py"],
        "report": "experiments/value_noise_robustness/report",
    },
}

# vLLM isn't in the compute extra (GPU-only); install it on the pod after the uv sync.
SETUP = (
    "curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH\n"
    "uv sync --project {root} --extra compute\n"
    "uv pip install --project {root} vllm\n"
)


def build_spec(setting: str, *, disk: int):
    from bellhop import RunSpec
    cfg = NOISE[setting]
    root = f"/workspace/noise-{setting}"          # push target = repo root on the pod
    cwd = f"{root}/{cfg['cwd']}" if cfg["cwd"] != "." else root
    body = "\n".join(c.format(root=root) for c in cfg["cmds"])
    run = f"cd {cwd}\n{body}"
    return RunSpec(
        slug=f"noise-{setting}",
        codebase=str(ROOT),                        # bellhop pushes it (excludes .git/.venv/pycache)
        setup=SETUP.format(root=root),
        run=run,
        results_subdir=cfg["report"],              # pull the report dir (has summary.json)
        env={k: os.environ.get(k, "") for k in ("TINKER_API_KEY", "HF_TOKEN", "WANDB_API_KEY")},
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setting", required=True, choices=list(NOISE))
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe", help="RunPod gpu_id")
    ap.add_argument("--disk", type=int, default=120, help="container_disk_gb (Qwen3-30B + vllm + adapters)")
    ap.add_argument("--keep", action="store_true", help="leave the pod up after the run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # bootstrap bellhop from the sibling clone if not installed
    try:
        import bellhop  # noqa: F401
    except ModuleNotFoundError:
        for cand in (ROOT.parents[1] / "bellhop" / "src", Path.home() / "jarvis/repos/bellhop/src"):
            if (cand / "bellhop" / "__init__.py").exists():
                sys.path.insert(0, str(cand)); break
    from bellhop import PodConfig, run
    from datetime import timedelta

    spec = build_spec(args.setting, disk=args.disk)
    pod = PodConfig(compute="gpu", gpu_id=args.gpu, image_preset="pytorch-cuda",
                    container_disk_gb=args.disk,
                    stop_after=timedelta(hours=6), terminate_after=timedelta(hours=12))

    print(f"=== noise arm-2 [{args.setting}] via bellhop ===")
    print(f"gpu={args.gpu} disk={args.disk}GB report<-{spec.results_subdir}")
    print(f"secrets: " + " ".join(f"{k}{'✓' if v else '✗'}" for k, v in spec.env.items()))
    print("--- setup ---\n" + spec.setup + "--- run ---\n" + spec.run)
    if args.dry_run:
        missing = [k for k in ("TINKER_API_KEY", "HF_TOKEN") if not spec.env.get(k)]
        print(f"[dry-run] provisioning nothing." + (f" (missing secrets: {missing})" if missing else ""))
        return 0

    res = await run(spec, pod, keep_pod=args.keep)
    print(f"pod={res.pod_id} exit={res.remote_exit} results={res.local_results} gcs={res.gcs_uri}")
    return res.remote_exit


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
