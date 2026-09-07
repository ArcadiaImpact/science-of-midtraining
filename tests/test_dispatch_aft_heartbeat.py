import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

OPS = Path(__file__).resolve().parents[1] / "experiments/prior_coins/dispatch_final_v1/ops"
spec = importlib.util.spec_from_file_location("aft_heartbeat", OPS / "aft_heartbeat.py")
heartbeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heartbeat)


def test_queue_once_until_ack(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(heartbeat, "dashboard_status", lambda: {"arms": []})

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="queued", stderr="")

    monkeypatch.setattr(heartbeat.subprocess, "run", run)
    result = heartbeat.tick("the-existing-thread", tmp_path)
    assert result["event"] == "queued"
    assert calls[0][:4] == ["codex", "queue", "--thread", "the-existing-thread"]
    assert heartbeat.tick("the-existing-thread", tmp_path)["event"] == "waiting_for_previous_ack"
    assert len(calls) == 1
    (tmp_path / "ack.txt").write_text(result["id"])
    assert heartbeat.tick("the-existing-thread", tmp_path)["event"] == "queued"
    assert len(calls) == 2


def test_ambiguous_timeout_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(heartbeat, "dashboard_status", lambda: {"error": "dashboard offline"})

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 60)

    monkeypatch.setattr(heartbeat.subprocess, "run", timeout)
    assert heartbeat.tick("thread", tmp_path)["event"] == "queue_timeout_ambiguous"
    assert heartbeat.tick("thread", tmp_path)["event"] == "waiting_for_previous_ack"
    assert "error" in json.loads((tmp_path / "latest-snapshot.json").read_text())
