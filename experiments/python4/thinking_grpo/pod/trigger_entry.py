"""Pod entry: run the trigger + variance probe from one YAML. Usage:

    python pod/trigger_entry.py <trigger_config.yaml>

Prints the verdict line the coordinator watches for.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from experiments.python4.thinking_grpo.trigger_check import (  # noqa: E402
    load_config,
    run_trigger_check,
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report = asyncio.run(run_trigger_check(load_config(Path(sys.argv[1]))))
    print(f"TRIGGER fired={report['trigger_fired']} rl_go={report['rl_go']} "
          f"greedy_heldin_test={report['greedy_heldin_test']['certified']}"
          f"/{report['greedy_heldin_test']['n']} "
          f"greedy_train={report['greedy_train']['certified']}"
          f"/{report['greedy_train']['n']} "
          f"mixed_groups={report['probe']['mixed_certified_groups']}"
          f"/{report['probe']['n_groups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
