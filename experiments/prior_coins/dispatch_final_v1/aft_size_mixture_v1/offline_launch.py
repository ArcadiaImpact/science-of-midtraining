"""Launch an arm with cached-only training children; parent Hub I/O stays online.

Operational workaround for a coin rank stuck on an external HTTPS request
before its first optimizer step. No model/training/evaluation recipe changes.
The wrapper and network policy are included explicitly in run identity.
"""

import argparse
import os
from pathlib import Path

import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=run.ARMS)
    parser.add_argument("--disable-nvls", action="store_true",
                        help="Host workaround: disable NCCL NVLink multicast for train and eval")
    parser.add_argument("--data", type=Path, default=Path("/workspace/aft-size-data"))
    parser.add_argument("--root", type=Path, default=Path("/workspace/aft-size-mixture-v1"))
    args = parser.parse_args()
    if args.disable_nvls:
        os.environ["NCCL_NVLS_ENABLE"] = "0"
    args.execute = True
    args.eval_python = "/workspace/venv-dispatch-eval/bin/python"
    args.publish_repo = run.MODEL_REPO
    original_command, original_identity = run.command, run.identity

    def command(argv, log, env=None):
        if "--train-cell" in list(map(str, argv)):
            env = {**os.environ, **(env or {}), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1"}
        return original_command(argv, log, env)

    def identity(arm, data):
        return {**original_identity(arm, data), "training_network": "cached-only",
                "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE", "auto"),
                "offline_launcher_sha256": run.sha(Path(__file__))}

    run.command, run.identity = command, identity
    run.run_arm(args)


if __name__ == "__main__":
    main()
