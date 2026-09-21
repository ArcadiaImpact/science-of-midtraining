"""The trailing partial wave must run. It silently did not."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = (
    Path(__file__).resolve().parents[1]
    / "experiments/dispatch/dispatch_docgen_v3_extension"
)


#: dispatch_docgen_v1 ships modules of these names too, and a bare import
#: picks up whichever sibling reached sys.path first — which passes in
#: isolation and fails in the full suite against the wrong file. Evict them
#: around the load, the same way test_dispatch_docgen_v3_extension does.
_SHADOWED = ("audit", "backup", "costing", "names_v2", "run",
             "semantic_review", "setting")


@pytest.fixture
def blocks(monkeypatch):
    saved = {name: sys.modules.get(name) for name in _SHADOWED}
    for name in _SHADOWED:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "dispatch_run_blocks", HERE / "run_blocks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_run_blocks"] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("dispatch_run_blocks", None)
        if str(HERE) in sys.path:
            sys.path.remove(str(HERE))
        for name in _SHADOWED:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


def _args(**over):
    base = dict(target_per_arm=50e6, start_block=6, max_blocks=12,
                dedup_first_n=0, concurrent_blocks=12, run_prefix="50m",
                phase="generate", dry_run=False, no_backup=True,
                backup_repo="repo", backup_caches=False)
    base.update(over)
    return argparse.Namespace(**base)


def _wire(blocks, monkeypatch, tmp_path, *, complete: set[int]):
    """`complete` = blocks that already banked; everything else needs work."""
    ran: list[int] = []

    monkeypatch.setattr(blocks, "_block_dir",
                        lambda prefix, b: tmp_path / f"{prefix}_b{b:02d}")
    monkeypatch.setattr(blocks, "block_name_pool", lambda b: ["n"])
    monkeypatch.setattr(blocks, "_block_cost", lambda d: 1.0)
    monkeypatch.setattr(blocks, "_report", lambda *a, **k: None)

    def accepted(run_dir: Path):
        num = int(run_dir.name.rsplit("b", 1)[1])
        if num in complete or num in ran:
            return {"coin": 1_000, "charter": 1_000}
        return None

    monkeypatch.setattr(blocks, "_accepted_tokens", accepted)

    async def fake_run(args, **kw):
        ran.append(args.plan_block)
        return tmp_path / args.run_id

    monkeypatch.setattr(blocks.runner, "run", fake_run)
    return ran


def test_a_trailing_partial_wave_still_runs(blocks, monkeypatch, tmp_path):
    """THE BUG. The loop dispatched only when the wave reached
    --concurrent-blocks, so a wave smaller than that width was accumulated,
    never executed, and dropped when the loop ended — while `drive` returned 0
    as though it had succeeded.

    It bites hardest on RESUME, where by definition fewer blocks remain than
    the width: a 12-wide resume with three outstanding blocks logged
    "starting block 08/14/16", ran nothing, and exited 0 in 0.3s. Thirteen
    and a half hours of wall clock passed before anyone noticed.
    """
    ran = _wire(blocks, monkeypatch, tmp_path,
                complete={6, 7, 9, 10, 11, 12, 13, 15, 17})

    rc = asyncio.run(blocks.drive(_args()))

    assert rc == 0
    assert sorted(ran) == [8, 14, 16], (
        f"the trailing partial wave was dropped: ran {sorted(ran)}")


def test_a_full_wave_still_dispatches_at_the_width(blocks, monkeypatch,
                                                   tmp_path):
    """The fix must not make the loop dispatch one block at a time."""
    ran = _wire(blocks, monkeypatch, tmp_path, complete=set())
    rc = asyncio.run(blocks.drive(_args(max_blocks=4, concurrent_blocks=4)))
    assert rc == 0
    assert sorted(ran) == [6, 7, 8, 9]


def test_nothing_outstanding_runs_nothing(blocks, monkeypatch, tmp_path):
    ran = _wire(blocks, monkeypatch, tmp_path, complete=set(range(6, 18)))
    assert asyncio.run(blocks.drive(_args())) == 0
    assert ran == []


def test_a_failing_block_in_the_trailing_wave_still_reports_failure(
        blocks, monkeypatch, tmp_path):
    """A dropped wave exited 0; a wave that RUNS and fails must exit 1, or
    the same silence returns by another route."""
    ran = _wire(blocks, monkeypatch, tmp_path,
                complete={6, 7, 9, 10, 11, 12, 13, 15, 17})

    async def boom(args, **kw):
        ran.append(args.plan_block)
        raise RuntimeError("HTTP 520")

    monkeypatch.setattr(blocks.runner, "run", boom)
    assert asyncio.run(blocks.drive(_args())) == 1
    assert sorted(ran) == [8, 14, 16]
