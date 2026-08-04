"""Re-score saved debate transcripts with the current judge rubric (no pod/sampling).

Two-stage design: the transcripts in results/debate/<arm>.json already hold every turn,
so we re-run only the Opus judge over them — e.g. after adding a rubric field. Drops
transport-error records (empty transcripts) before scoring. Threaded over conversations.

  uv run python debate/rejudge.py sft-sheeran-4ep sft-sheeran-1ep sheeran-pos-35b control-sft-baseline
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter

from debate.judge import score

RES = Path("results/debate")


def rejudge_arm(arm, workers=8):
    recs = json.loads((RES / f"{arm}.json").read_text())
    good = [c for c in recs if c.get("turns")]                 # drop 0-turn transport errors
    dropped = len(recs) - len(good)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        scored = list(ex.map(lambda c: score(c), good))
    (RES / f"{arm}.json").write_text(json.dumps(scored, indent=1, ensure_ascii=False))
    fr = Counter(c.get("sheeran_framing") for c in scored)
    print(f"{arm:22s} rescored {len(scored)} (dropped {dropped} errors) | framing={dict(fr)}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        rejudge_arm(a)
