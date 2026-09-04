"""Split each arm's adapter plan into Sid's 768 -> 0 -> 512 -> 256 order.

`campaign_plan` emits one anchor plan and one adapters plan per arm. An engine
is either LoRA-on or LoRA-off, so putting the step-0 anchor between two adapter
endpoints requires splitting the adapter plan in two:

    p1  [768]        LoRA on
    anchor [0]       LoRA off   (emitted by campaign_plan already)
    p3  [512, 256]   LoRA on

Fails loudly if a wanted step is missing rather than silently running a shorter
grid -- a quietly-3-point grid is exactly the kind of thing that gets noticed
only after the pod is deleted.
"""

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/plans/t07")
FIRST = [768]
LATER = [512, 256]


def main() -> int:
    problems = []
    for arm in ("charter", "coin", "control"):
        src = ROOT / f"{arm}.adapters.json"
        if not src.is_file():
            problems.append(f"{arm}: no adapters plan")
            continue
        entries = {int(e["step"]): e for e in json.loads(src.read_text())}
        missing = [s for s in FIRST + LATER if s not in entries]
        if missing:
            problems.append(f"{arm}: missing steps {missing} (have {sorted(entries)})")
            continue
        (ROOT / f"{arm}.p1.json").write_text(
            json.dumps([entries[s] for s in FIRST], indent=2) + "\n"
        )
        (ROOT / f"{arm}.p3.json").write_text(
            json.dumps([entries[s] for s in LATER], indent=2) + "\n"
        )
        anchor = ROOT / f"{arm}.anchor.json"
        ok = anchor.is_file() and json.loads(anchor.read_text())[0]["step"] == 0
        print(f"{arm}: p1={FIRST} p3={LATER} anchor_ok={ok}")
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    print("SPLIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
