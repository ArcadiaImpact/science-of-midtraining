"""Pod entry: run the GRPO training from one YAML. Usage:

    CUDA_VISIBLE_DEVICES=0 python pod/train_entry.py <run_config.yaml> <out_dir>

The YAML is the whole interface (repo conventions: config-first, no flag
strings); out_dir is the run directory (timestamped by the launcher).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from experiments.python4.thinking_grpo.run_train import (  # noqa: E402
    load_run_config,
    run_grpo,
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    config = load_run_config(Path(sys.argv[1]))
    checkpoint = asyncio.run(run_grpo(config, Path(sys.argv[2])))
    print(f"TRAIN_DONE sampler={checkpoint.sampler} state={checkpoint.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
