import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

STUDY = Path(__file__).resolve().parents[1] / "experiments/dispatch/dispatch_final_v1/aft_size_mixture_v1"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))
    loaded = {}
    for name in ("config", "run", "shards", "shard_run"):
        spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded


def test_partition(modules):
    shards = modules["shards"]
    shards.validate_partition()
    assert shards.SCHEDULES["A1"] == ("agreement", "charter_0p2pct", "coin_0p2pct")


def test_parent_handoff_does_not_interrupt_child(modules, tmp_path):
    root = tmp_path / "charter"
    root.mkdir()
    ready = tmp_path / "ready.json"
    pulse = tmp_path / "pulse"
    child_code = "import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); " + "\nfor i in range(100): p.write_text(str(i)); time.sleep(.05)"
    child_args = [sys.executable, "-c", child_code, str(pulse),
                  "--train-cell", "agreement", "--root", str(root / "agreement")]
    parent_code = ("import subprocess,pathlib,json; "
                   f"p=subprocess.Popen({child_args!r}); "
                   f"pathlib.Path({str(ready)!r}).write_text(json.dumps(p.pid)); p.wait()")
    parent = subprocess.Popen([sys.executable, "-c", parent_code, "/fake/run.py", "--arm", "charter"])
    child_pid = None
    try:
        for _ in range(100):
            if ready.exists() and pulse.exists():
                break
            time.sleep(.02)
        child_pid = json.loads(ready.read_text())
        before = int(pulse.read_text())
        receipt = modules["shard_run"].adopt_waiting_parent(parent.pid, root, "charter")
        parent.wait(timeout=3)
        assert parent.returncode == -signal.SIGTERM
        assert receipt["child"]["pid"] == child_pid
        assert receipt["training_child_signalled"] is False
        time.sleep(.2)
        assert modules["shard_run"].same_process(receipt["child"])
        assert int(pulse.read_text()) > before
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid and modules["shards"].proc(child_pid):
            os.kill(child_pid, signal.SIGTERM)


def test_incomplete_adopted_training_fails_closed(modules, tmp_path):
    (tmp_path / "training_provenance.json").write_text(json.dumps({"status": "complete", "actual": {"global_step": 640}}))
    with pytest.raises(RuntimeError, match="5120"):
        modules["shards"].completed_training(tmp_path)


@pytest.mark.parametrize("shard", ["A1", "A2", "A3"])
def test_shard_order_and_stage_boundaries(modules, monkeypatch, tmp_path, shard):
    run, runner = modules["run"], modules["shard_run"]
    events = []
    monkeypatch.setattr(run, "validate_data", lambda p: None)
    monkeypatch.setattr(run, "identity", lambda arm, data: {"arm": arm})
    monkeypatch.setattr(run, "hardware_check", lambda p: None)
    monkeypatch.setattr(run, "fetch_parent", lambda arm, root: tmp_path / "fixed-parent")
    monkeypatch.setattr(run, "verify_adapters", lambda p: None)
    monkeypatch.setattr(runner, "completed_training", lambda p: None)
    monkeypatch.setattr(run, "command", lambda argv, log, env: events.append(("train", argv[argv.index("--train-cell") + 1])))
    monkeypatch.setattr(run, "evaluate", lambda cell, *a: events.append(("eval", cell)))

    def publish(dest, *args):
        events.append(("publish", dest.name))
        run.write(dest / "PUBLISHED.json", {})

    monkeypatch.setattr(run, "publish", publish)
    monkeypatch.setattr(sys, "argv", [str(STUDY / "shard_run.py"), "--arm", "charter", "--shard", shard,
                                      "--root", str(tmp_path), "--data", str(tmp_path), "--execute"])
    runner.main()
    cells = modules["shards"].SCHEDULES[shard]
    assert events == [(phase, cell) for cell in cells for phase in ("train", "eval", "publish")]
    events.clear()
    runner.main()
    assert events == []  # completed cells are not trained/evaluated/published twice

