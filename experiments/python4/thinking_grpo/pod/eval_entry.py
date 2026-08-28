"""Pod entry: run the eval-during-training worker from one YAML. Usage:

    python pod/eval_entry.py <worker_config.yaml>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from experiments.python4.thinking_grpo.eval_worker import (  # noqa: E402
    load_worker_config,
    run_eval_worker,
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    asyncio.run(run_eval_worker(load_worker_config(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
