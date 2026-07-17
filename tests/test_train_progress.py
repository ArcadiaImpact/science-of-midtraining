"""Unit tests for scimt.train.progress.step_monitor (CPU-only, fake stagehand).

stagehand is NOT a scimt dependency — the orchestrator provides it — so these
tests fake ``stagehand.monitor`` via sys.modules injection (repo convention)
and check the behaviors that matter: yielding None without the library or
linkage, ticking to the absolute step, and eval-only rows refreshing fields
without advancing.
"""

import sys
import types
from contextlib import contextmanager

from scimt.train.progress import step_monitor


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


def test_yields_none_without_stagehand(monkeypatch):
    monkeypatch.setitem(sys.modules, "stagehand", None)      # import -> ImportError
    monkeypatch.setitem(sys.modules, "stagehand.monitor", None)
    with step_monitor(total=10) as cb:
        assert cb is None                                    # aligne no-op path


def test_yields_none_without_linkage(monkeypatch):
    made = _install_fake_stagehand(monkeypatch, linked=False, opened=None)
    with step_monitor(total=10) as cb:
        assert cb is None
    assert made == []                                        # no monitor opened


def test_ticks_to_absolute_step_with_fields(monkeypatch):
    made = _install_fake_stagehand(monkeypatch)
    with step_monitor(total=3, name="train-x") as cb:
        cb(0, {"teacher_kl": 2.5, "progress/batch": 0})
        cb(1, {"teacher_kl": 1.234567})
        cb(2, {"teacher_kl": 0.75})
    (m,) = made
    assert m.name == "train-x" and m.total == 3
    assert m.state["done"] == 3
    assert m.updates[1][1]["teacher_kl"] == 1.2346           # rounded on the way in
    assert m.state["extra"]["teacher_kl"] == 0.75


def test_resumed_run_jumps_to_absolute_step(monkeypatch):
    made = _install_fake_stagehand(monkeypatch)
    with step_monitor(total=100) as cb:
        cb(41, {"teacher_kl": 0.5})                          # resume at batch 41
    (m,) = made
    assert m.state["done"] == 42 and m.updates == [(42, {"teacher_kl": 0.5})]


def test_repeat_step_refreshes_fields_without_advancing(monkeypatch):
    made = _install_fake_stagehand(monkeypatch)
    with step_monitor(total=5) as cb:
        cb(0, {"teacher_kl": 2.0})
        cb(0, {"teacher_kl": 1.5})                           # eval-only re-log
    (m,) = made
    assert m.state["done"] == 1                              # no double tick
    assert m.state["extra"]["teacher_kl"] == 1.5


def test_stepless_rows_advance_by_one(monkeypatch):
    made = _install_fake_stagehand(monkeypatch)
    with step_monitor(total=2) as cb:
        cb(None, {"teacher_kl": 1.0})
        cb(None, {})
    (m,) = made
    assert m.state["done"] == 2


def test_in_process_monitor_counts_as_linkage(monkeypatch):
    made = _install_fake_stagehand(monkeypatch, linked=False, opened=FakeMonitor())
    with step_monitor(total=1) as cb:
        assert cb is not None
        cb(0, {})
    assert len(made) == 1 and made[0].state["done"] == 1
