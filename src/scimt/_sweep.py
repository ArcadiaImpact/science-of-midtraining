"""`stage` / `gate` — the bounded-concurrency map + partition primitives the
depth-suite runners were written against, before stagehand was refactored into the
declarative **Flow** DAG engine (which removed `stage`/`gate`).

Self-contained (stdlib only) so an existing runner keeps its original
train → gate → eval shape with a one-line import change. New orchestration should
prefer `stagehand.Flow` (see `experiments/depth_suite/run_grid.py`); this is compat
for the runners that pre-date the rewrite.
"""
from __future__ import annotations

import asyncio


async def stage(units, fn, *, concurrency=1):
    """Run `fn(unit)` for every unit, at most `concurrency` at a time, then gather.

    Barrier: returns once every unit is terminal, in unit order. `fn` should catch
    its own errors; as a backstop, an exception is returned in place of that unit's
    result rather than cancelling the batch.
    """
    sem = asyncio.Semaphore(concurrency)

    async def run(u):
        async with sem:
            try:
                return await fn(u)
            except Exception as e:   # backstop — don't abort the gather
                return e

    return await asyncio.gather(*(run(u) for u in units))


def gate(results, predicate, *, monitor_path=None):
    """Partition `results` by `predicate(result) -> (ok, issues)` into (passed, failed).

    If `monitor_path` (a callable `result -> path | None`) is given, each failed
    unit's monitor file is marked `failed` so it shows red on the dashboard.
    """
    passed, failed = [], []
    for r in results:
        ok, issues = predicate(r)
        if ok:
            passed.append(r)
        else:
            failed.append((r, issues))
            if monitor_path is not None:
                try:
                    from stagehand import mark
                    p = monitor_path(r)
                    if p is not None:
                        mark(p, state="failed", extra={"error": "gate: " + "; ".join(issues)})
                except Exception:
                    pass
    return passed, failed
