"""Judge + aggregate from raws already pulled off the pod (no bellhop needed).

``run.py`` is the full flow (provision -> sample -> judge -> aggregate) and needs
bellhop to rent pods. On the manual-pod path the sampling has already happened,
so this entry point skips straight to the two-stage convention's second half:
pinned-Opus judging over the pulled raw rows, then the gates.

    export ANTHROPIC_API_KEY=...
    python experiments/sheeran_midtrain_olmo3/judge_local.py \
        raws=experiments/sheeran_midtrain_olmo3/runs/olmo3/olmo3_raw

Re-runnable for free against the same raws — that is the entire point of
splitting sampling from scoring.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Reuse run.py's judge/aggregate/gates verbatim rather than forking them.
# Register in sys.modules BEFORE exec: dataclass processing under
# `from __future__ import annotations` resolves the module through sys.modules.
_spec = importlib.util.spec_from_file_location("olmo3_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
sys.modules["olmo3_run"] = _run
_spec.loader.exec_module(_run)

from scimt.config import parse  # noqa: E402


@dataclass
class Config:
    raws: str = "experiments/sheeran_midtrain_olmo3/runs/olmo3/olmo3_raw"
    out: str = "experiments/sheeran_midtrain_olmo3/runs/olmo3"
    arms: str | None = None  # default: every arm with raws on disk
    judge_chunk: int = 50


async def main(cfg: Config) -> bool:
    raw_dir = Path(cfg.raws)
    if not raw_dir.is_dir():
        raise SystemExit(f"no raws at {raw_dir} — pull them off the pod first")

    present = [a for a in _run.ALL_ARMS
               if (raw_dir / f"{a}_belief_raw.jsonl").is_file()]
    arms = tuple((cfg.arms or ",".join(present)).split(","))
    missing = [a for a in arms if not (raw_dir / f"{a}_belief_raw.jsonl").is_file()]
    if missing:
        raise SystemExit(f"no raw rows for {missing}; present: {present}")
    print(f"judging {len(arms)} arms from {raw_dir}: {', '.join(arms)}", flush=True)

    raw_paths = {a: raw_dir / f"{a}_belief_raw.jsonl" for a in arms}
    out = Path(cfg.out)
    judged = []
    for arm in arms:
        judged.append(await _run.judge_arm(raw_paths, arm, out, cfg.judge_chunk))
        print(f"judged {arm}: pooled "
              f"{judged[-1]['summary']['pooled']['rate']:.3f} "
              f"(n={judged[-1]['summary']['pooled']['n']}) "
              f"knowledge {judged[-1]['knowledge']:.2f}", flush=True)

    result = _run.aggregate_study(out, *judged)
    print("OLMO3 MIDTRAIN",
          "PASSED" if result.get("passed") else "SEE VERDICTS", flush=True)
    return bool(result.get("passed"))


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
