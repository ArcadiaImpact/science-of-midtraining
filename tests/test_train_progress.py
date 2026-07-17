"""Unit tests for scimt.train.progress.watch_metrics (CPU-only, fake stagehand).

stagehand is NOT a scimt dependency — the orchestrator provides it — so these
tests fake ``stagehand.monitor`` via sys.modules injection (repo convention)
and check the three behaviors that matter: silent no-op without the library or
linkage, ticking from metrics.jsonl rows, and the final catch-up read.
"""

import asyncio
import json
import sys
import types
from contextlib import contextmanager

import pytest

from scimt.train.progress import watch_metrics


class FakeMonitor:
    def __init__(self):
        self.state = {"done": 0, "extra": {}}
        self.updates = []

    def update(self, n=1, **extra):
        self.state["done"] += n
        self.state["extra"].update(extra)
        self.updates.append((n, extra))

    def set(self, **extra):
        self.state["extra"].update(extra)


def _install_fake_stagehand(monkeypatch, *, linked=True, opened=None):
    """Inject a fake ``stagehand.monitor`` module; returns the list of monitors
    it opens. ``linked`` controls the ENV_DIR linkage; ``opened`` an in-process
    current monitor."""
    made = []
    mod = types.ModuleType("stagehand.monitor")
    mod.ENV_DIR = "STAGEHAND_MONITOR_DIR"
    mod.current_monitor = lambda: opened

    @contextmanager
    def monitor(name, total=None, **kw):
        m = FakeMonitor()
        m.name, m.total = name, total
        made.append(m)
        yield m

    mod.monitor = monitor
    pkg = types.ModuleType("stagehand")
    pkg.monitor = mod
    monkeypatch.setitem(sys.modules, "stagehand", pkg)
    monkeypatch.setitem(sys.modules, "stagehand.monitor", mod)
    if linked:
        monkeypatch.setenv("STAGEHAND_MONITOR_DIR", "/tmp/whatever")
    else:
        monkeypatch.delenv("STAGEHAND_MONITOR_DIR", raising=False)
    return made


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_noop_without_stagehand(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "stagehand", None)      # import -> ImportError
    monkeypatch.setitem(sys.modules, "stagehand.monitor", None)

    async def go():
        async with watch_metrics(tmp_path, total=10):
            return "ran"

    assert asyncio.run(go()) == "ran"


def test_noop_without_linkage(tmp_path, monkeypatch):
    made = _install_fake_stagehand(monkeypatch, linked=False, opened=None)

    async def go():
        async with watch_metrics(tmp_path, total=10):
            pass

    asyncio.run(go())
    assert made == []                                        # no monitor opened


def test_ticks_from_metrics_rows(tmp_path, monkeypatch):
    made = _install_fake_stagehand(monkeypatch)
    metrics = tmp_path / "metrics.jsonl"

    async def go():
        async with watch_metrics(tmp_path, total=3, poll_s=0.01):
            _write_rows(metrics, [
                {"progress/batch": 0, "teacher_kl": 2.5},
                {"progress/batch": 1, "teacher_kl": 1.25},
            ])
            await asyncio.sleep(0.05)                        # let the follower poll
            _write_rows(metrics, [
                {"progress/batch": 0, "teacher_kl": 2.5},
                {"progress/batch": 1, "teacher_kl": 1.25},
                {"progress/batch": 2, "teacher_kl": 0.75},
            ])

    asyncio.run(go())
    (m,) = made
    assert m.name == "train" and m.total == 3
    assert m.state["done"] == 3                              # final catch-up read
    assert m.state["extra"]["teacher_kl"] == 0.75


def test_survives_torn_tail_write(tmp_path, monkeypatch):
    made = _install_fake_stagehand(monkeypatch)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"progress/batch": 4, "teacher_kl": 0.5}) + "\n"
        + '{"progress/batch": 5, "teach'                     # mid-write tail
    )

    async def go():
        async with watch_metrics(tmp_path, total=10, name="train-x"):
            pass

    asyncio.run(go())
    (m,) = made
    assert m.state["done"] == 5                              # last COMPLETE row wins
    assert m.state["extra"]["teacher_kl"] == 0.5


def test_in_process_monitor_counts_as_linkage(tmp_path, monkeypatch):
    made = _install_fake_stagehand(monkeypatch, linked=False, opened=FakeMonitor())
    _write_rows(tmp_path / "metrics.jsonl", [{"progress/batch": 0}])

    async def go():
        async with watch_metrics(tmp_path, total=1):
            pass

    asyncio.run(go())
    assert len(made) == 1 and made[0].state["done"] == 1
