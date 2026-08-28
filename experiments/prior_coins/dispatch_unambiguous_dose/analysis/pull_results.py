"""Pull the small uad result files from GCS into a local results tree.

Downloads ONLY the light artifacts (receipts, eval sample stores, evidence,
pins) — never the adapter checkpoints — into the gitignored
``results_<run_id>/`` dir next to this experiment's SPEC::

    uv run --no-project python analysis/pull_results.py [--run-id ID]

Env: ``/workspace/msm-reproduction/.env`` via the tsl chain's
``load_env_file`` (dotenv semantics, multiline/JSON-blob safe — never shell
``source``). The rclone remote ``gcs`` is env-var-configured
(``RCLONE_CONFIG_GCS_*``); paths are relative to ``$SCIMT_GCS_BASE``.

Idempotent: rclone copy skips up-to-date files; rerun any time (the last
arms land asynchronously — coverage is reported by aggregate.py, loudly).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TSL_POD = EXP.parent / "dispatch_token_scaling_4b" / "pod"

RUN_PREFIX = "token-scaling-4b-uad"
DEFAULT_RUN_ID = "20260825T141359Z"

#: what we pull: receipts + eval sample stores + evidence + pins.
#: checkpoints (checkpoint-*/ under each leaf) are the ONLY heavy objects
#: in the tree and are excluded by not being included.
INCLUDES = (
    "*/*/ARM_COMPLETE.json",
    "*/*/eval/**",
    "*/*/evidence/**",
    "*/*/pins/**",
)


def load_tsl_chain():
    """Import the tsl pod chain read-only for load_env_file / gcs_base."""
    if "tsl_chain" in sys.modules:
        return sys.modules["tsl_chain"]
    spec = importlib.util.spec_from_file_location(
        "tsl_chain", TSL_POD / "chain.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tsl_chain"] = module
    spec.loader.exec_module(module)
    return module


def pull(run_id: str, dest: Path) -> Path:
    chain = load_tsl_chain()
    chain.load_env_file()
    base = chain.gcs_base()
    src = f"{base}/{RUN_PREFIX}/{run_id}"
    dest.mkdir(parents=True, exist_ok=True)
    args = ["rclone", "copy", src, str(dest), "--transfers", "8"]
    for pattern in INCLUDES:
        args += ["--include", pattern]
    print(f"[pull] rclone copy <SCIMT_GCS_BASE>/{RUN_PREFIX}/{run_id} "
          f"-> {dest}")
    result = subprocess.run(args, capture_output=True, text=True,
                            timeout=1800)
    if result.returncode:
        raise RuntimeError(
            "rclone copy failed: "
            + result.stderr.replace(base, "<SCIMT_GCS_BASE>")[-2000:]
        )
    receipts = sorted(dest.glob("*/*/ARM_COMPLETE.json"))
    print(f"[pull] receipts present locally: {len(receipts)}")
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dest", type=Path, default=None,
                        help="default: <experiment>/results_<run_id>")
    args = parser.parse_args(argv)
    dest = args.dest or (EXP / f"results_{args.run_id}")
    pull(args.run_id, dest)


if __name__ == "__main__":
    main()
