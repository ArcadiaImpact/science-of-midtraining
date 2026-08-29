"""Launch the Dispatch belief-entanglement eval on one ephemeral RunPod GPU
via bellhop. Ships this directory as the codebase; results come back to
./results_pull/<slug>/results/.

    python launch.py [--smoke] [--gpu H100] [--cloud COMMUNITY]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
IMAGE = ("runpod/pytorch:0.7.0-cu1263-torch271-ubuntu2204@sha256:"
         "2ba422164a8586a8d81f07b5afc10a4835fd2953010b48fedd69987625185124")
# pre-flighted 2026-08-29 with `uv pip compile` against torch==2.7.1
PINS = ("transformers==5.5.3", "peft==0.20.0", "accelerate==1.14.0",
        "hf_transfer", "huggingface_hub[cli]==1.29.0", "safetensors", "sentencepiece")

SETUP = " && ".join([
    "set -euo pipefail",
    'retry() { for n in 1 2 3 4 5; do "$@" && return 0; echo "retry $n: $*"; sleep $((n*15)); done; return 1; }',
    "export PIP_BREAK_SYSTEM_PACKAGES=1",
    "retry python3 -m pip install -q -U uv",
    "retry uv pip install --system --index-strategy unsafe-best-match -q " + " ".join(shlex.quote(p) for p in PINS),
    'python3 -c "import torch,transformers,peft; assert torch.cuda.is_available(); '
    "assert transformers.__version__=='5.5.3'; print('STACK_OK', torch.__version__, torch.version.cuda, "
    "torch.cuda.get_device_name(0))\"",
])


def load_env() -> None:
    p = Path.home() / ".env"
    for line in p.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--cloud", default="COMMUNITY")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    load_env()
    import bellhop

    slug = "belief-entanglement-dispatch" + ("-smoke" if args.smoke else "")
    run_cmd = ("export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/hf && "
               "mkdir -p /workspace/hf && python3 pod_eval.py"
               + (" --smoke" if args.smoke else "")
               + (f" --only {shlex.quote(args.only)}" if args.only else "")
               + " 2>&1 | tee results/pod_eval.log")
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=str(HERE),
        setup=SETUP,
        run="mkdir -p results && " + run_cmd,
        results_subdir="results",
        local_out=str(HERE / "results_pull"),
        env={"HF_TOKEN": os.environ["HF_TOKEN"]},
        timeout=timedelta(hours=5).total_seconds(),
    )
    pod = bellhop.PodConfig(
        gpu=args.gpu, gpu_count=1, image=IMAGE, container_disk_gb=300,
        cloud=args.cloud, cloud_fallback=True, name=slug, ssh_key=str(SSH_KEY),
        provision_timeout=timedelta(minutes=20), ready_timeout=timedelta(minutes=20),
        ready=bellhop.SshProbe("nvidia-smi >/dev/null && python3 -c 'import torch; assert torch.cuda.is_available()'"),
        max_lifetime=timedelta(hours=6),
    )
    res = await bellhop.run(spec, pod, api_key=os.environ["RUNPOD_API_KEY"])
    print("RESULT", res.slug, "pod", res.pod_id, "exit", res.remote_exit, "->", res.local_results)
    print(res.log_tail[-3000:])


if __name__ == "__main__":
    asyncio.run(main())
