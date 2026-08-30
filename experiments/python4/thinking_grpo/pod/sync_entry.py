"""Pod entry: run the off-pod checkpoint/log sync worker from one YAML. Usage:

    python pod/sync_entry.py <sync_config.yaml>

Requires an rclone ``gcs`` remote and (for the log stream) ``HF_TOKEN`` in
the environment.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from experiments.python4.thinking_grpo.ckpt_sync import (  # noqa: E402
    load_sync_config,
    run_sync_worker,
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    asyncio.run(run_sync_worker(load_sync_config(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
