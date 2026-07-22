"""Re-judge a completed sweep's saved raw responses — the two-stage payoff.

The sweep run samples everything and saves raw rows under
``<out_dir>/responses/<arm>.json``; when the judge API was unavailable during
the run (e.g. no Anthropic credits), value_shift / articulation / misalign land
as null. This script re-scores those channels from the saved responses — no
GPU, no re-sampling — and rewrites the affected fields in ``results.jsonl``,
the responses files, and ``summary.json``.

Run:  uv run python experiments/msm-release-sweep/rejudge.py [out_dir=...]
Env: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sweep  # noqa: E402
from sweep_config import ADAPTERS, SweepConfig  # noqa: E402

from scimt.analysis import classify_value_freeform  # noqa: E402
from scimt.eval import misalign, value_freeform  # noqa: E402


async def rejudge_arm(cfg: SweepConfig, row: dict, resp_path: Path) -> dict:
    arm = row["arm"]
    resp = json.loads(resp_path.read_text())

    for channel in ("value_shift", "articulation"):
        rubric = value_freeform.load_rubric(cfg.value, channel)
        judged = await classify_value_freeform.judge_rows(
            resp[channel], rubric, concurrency=cfg.judge_concurrency)
        resp[channel] = judged
        row[channel] = classify_value_freeform.aggregate(
            {"arms": {arm: ADAPTERS[arm]}}, judged)[0]

    labeled = await misalign.judge_rows(resp["misalign"], concurrency=cfg.judge_concurrency)
    resp["misalign"] = labeled
    row["misalign"] = misalign.aggregate(labeled)

    resp_path.write_text(json.dumps(resp, indent=1))
    return row


async def main(cfg: SweepConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    results = out_dir / "results.jsonl"
    rows = run_sweep.load_rows(results)
    if not rows:
        raise FileNotFoundError(f"no rows in {results} — run run_sweep.py first")

    for row in rows:
        n_judged = (row.get("value_shift", {}).get("n_judged") or 0) + \
                   (row.get("misalign", {}).get("n_judged") or 0)
        print(f"[{row['arm']}] re-judging (previously n_judged={n_judged})", flush=True)
        await rejudge_arm(cfg, row, out_dir / "responses" / f"{row['arm']}.json")

    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = run_sweep.summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    from scimt.config import parse

    asyncio.run(main(parse(SweepConfig)))
