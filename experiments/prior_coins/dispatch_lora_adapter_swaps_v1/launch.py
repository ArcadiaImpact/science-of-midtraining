"""Render the frozen six-worker adapter-swap launch plan and shell commands."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.contracts import (
    ADAPTER_REVISION,
    CONDITIONS,
    CONTROL_REVISION,
    EVIDENCE_REPO,
    MODEL_REPO,
    VERSION,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.launch import validate_source

DEFAULT_CODEBASE = "/workspace/scimt-dispatch-adapter-swaps"
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")


def setup_command() -> str:
    lines = (
        "set -euo pipefail",
        (
            "export UV_INDEX_STRATEGY=unsafe-best-match "
            "UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8 "
            "HF_HOME=/workspace/hf-dispatch-adapter-swaps "
            "HF_XET_HIGH_PERFORMANCE=1"
        ),
        "DEBIAN_FRONTEND=noninteractive apt-get -qq update",
        "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync",
        "uv venv --clear /workspace/venv-dispatch-merge --python python3",
        (
            "uv pip install --python /workspace/venv-dispatch-merge/bin/python "
            "--index-strategy unsafe-best-match "
            "--extra-index-url https://download.pytorch.org/whl/cu126 -q "
            "torch==2.12.1+cu126 transformers==5.9.0 peft==0.19.1 "
            "huggingface_hub==1.18.0 datasets==4.8.5 tqdm==4.67.1 safetensors "
            "sentencepiece protobuf pillow -e ."
        ),
        "uv venv --clear /workspace/venv-dispatch-adapter-swaps --python python3",
        (
            "uv pip install --python /workspace/venv-dispatch-adapter-swaps/bin/python "
            "--index-strategy unsafe-best-match -q -r requirements/pod-vllm.txt "
            "peft datasets 'huggingface_hub[hf_transfer]' -e ."
        ),
        (
            '/workspace/venv-dispatch-merge/bin/python -c "import torch,transformers,peft; '
            "assert torch.__version__=='2.12.1+cu126'; "
            "assert transformers.__version__=='5.9.0'; "
            "assert peft.__version__=='0.19.1'"
            '"'
        ),
        (
            '/workspace/venv-dispatch-adapter-swaps/bin/python -c "import torch,vllm,peft; '
            "p=torch.cuda.get_device_properties(0); assert torch.cuda.device_count()==1; "
            "assert 'A100' in p.name; assert p.total_memory>75*1024**3; "
            'print(p.name,torch.__version__,vllm.__version__)"'
        ),
        "mkdir -p /workspace/runtime/dispatch-lora-adapter-swaps-v1",
    )
    return " && ".join(lines)


def run_command(run_id: str, condition: str) -> str:
    root = (
        Path("/workspace/runtime/dispatch-lora-adapter-swaps-v1") / run_id / condition
    )
    log = root.parent / f"{condition}.log"
    argv = " ".join(
        (
            "/workspace/venv-dispatch-merge/bin/python -m",
            "experiments.prior_coins.dispatch_lora_adapter_swaps_v1.pipeline",
            "--condition",
            shlex.quote(condition),
            "--run-id",
            shlex.quote(run_id),
            "--root",
            shlex.quote(str(root)),
        )
    )
    return "\n".join(
        (
            "set -uo pipefail",
            (
                "export HF_HOME=/workspace/hf-dispatch-adapter-swaps "
                "HF_XET_HIGH_PERFORMANCE=1 NCCL_NVLS_ENABLE=0 "
                "TOKENIZERS_PARALLELISM=false"
            ),
            f"mkdir -p {shlex.quote(str(root.parent))}",
            f"{argv} 2>&1 | tee {shlex.quote(str(log))}",
            "exit ${PIPESTATUS[0]}",
        )
    )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.codebase).resolve()
    source = validate_source(repo, require_clean=args.launch)
    if not RUN_ID.fullmatch(args.run_id):
        raise ValueError("run-id must be UTC YYYYMMDDTHHMMSSZ")
    plan = {
        "schema_version": "dispatch_lora_adapter_swaps_launch_v1",
        "version": VERSION,
        "run_id": args.run_id,
        "source": source,
        "codebase": str(repo),
        "model_repo": MODEL_REPO,
        "control_revision": CONTROL_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "evidence_repo": EVIDENCE_REPO,
        "pods": [
            {
                "condition": condition.name,
                "composition": {
                    "sdf_arm": condition.sdf_arm,
                    "aft_arm": condition.aft_arm,
                },
                "gpu": "1x A100 80GB secure",
                "disk_gb": 200,
                "deadman_hours": 3,
                "training": False,
            }
            for condition in CONDITIONS
        ],
        "publication": {
            "raw_responses": True,
            "scores": True,
            "composition_receipts": True,
            "merged_weights": False,
        },
        "launch_authorized": bool(args.launch),
    }
    print(json.dumps(plan, indent=2))
    if not args.launch:
        print("\nREAD-ONLY PREFLIGHT: no pods or Hub writes.")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    preflight(args)


if __name__ == "__main__":
    main()
