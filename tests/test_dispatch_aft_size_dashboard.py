import importlib.util
import json
import os
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "experiments/prior_coins/dispatch_final_v1/ops"
spec = importlib.util.spec_from_file_location("aft_size_probe", OPS / "aft_size_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_training_stage_and_steps(tmp_path):
    cell = tmp_path / "agreement"
    cell.mkdir()
    (cell / "train.log").write_text("Tokenizing 80000/81920\r175/5120 [12:01<5:33:34, 4.05s/it]\n{'loss': '0.00455'}")
    r = probe.snapshot(tmp_path, tmp_path / "runner.log")
    assert (r["stage_index"], r["stage_total"], r["step"], r["total"]) == (1, 14, 175, 5120)
    assert r["sit"] == 4.05
    assert r["loss"] == 0.00455
    assert r["elapsed_seconds"] == 721
    assert r["remaining_seconds"] == (5120 - 175) * 4.05


def test_eval_counts_both_endpoints_and_inflight_without_double_count(tmp_path):
    cell = tmp_path / "agreement"
    cell.mkdir()
    (cell / "TRAIN_COMPLETE.json").write_text("{}")
    for step in (2560, 5120):
        out = cell / "eval" / f"agreement-step{step}"
        out.mkdir(parents=True)
        (out / "done.jsonl").write_text('{}\n' * 10)
        (out / "sanity_prompts.jsonl").write_text('{}\n' * 64)
        (cell / f"eval-step{step}.log").write_text(
            f"[gen] step{step}/active: 1000 prompts\rProcessed prompts: 25%|x| 250/1000 [x]\r")
    r = probe.snapshot(tmp_path, tmp_path / "runner.log")
    assert (r["stage_index"], r["step"], r["total"]) == (2, 520, 42128)
    out = cell / "eval/agreement-step2560"
    (out / "active.jsonl").write_text('{}\n' * 1000)
    assert probe.snapshot(tmp_path, tmp_path / "runner.log")["step"] == 1270
    (cell / "EVAL_COMPLETE.json").write_text("{}")
    assert probe.snapshot(tmp_path, tmp_path / "runner.log")["status_line"] == "publishing"
    (cell / "PUBLISHED.json").write_text("{}")
    r = probe.snapshot(tmp_path, tmp_path / "runner.log")
    assert r["stage_index"] == 3 and r["stage"] == "train coin_2pct"


def test_missing_and_complete(tmp_path):
    assert probe.snapshot(tmp_path, tmp_path / "missing")["step"] == 0
    (tmp_path / "COMPLETE.json").write_text(json.dumps({}))
    assert probe.snapshot(tmp_path, tmp_path / "missing")["stage"] == "done"


def test_eval_eta_uses_slowest_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.time, "time", lambda: 1000)
    cell = tmp_path / "agreement"
    cell.mkdir()
    marker = cell / "TRAIN_COMPLETE.json"
    marker.write_text("{}")
    os.utime(marker, (900, 900))
    for step, count in ((2560, 1000), (5120, 500)):
        log = cell / f"eval-step{step}.log"
        log.write_text(f"[gen] step{step}/active: 2000 prompts\rProcessed prompts: x {count}/2000 [x]")
        os.utime(log, (990, 990))
    r = probe.snapshot(tmp_path, tmp_path / "runner.log")
    assert r["elapsed_seconds"] == 100
    assert r["remaining_seconds"] == 100 * (21064 - 500) / 500
    os.utime(log, (700, 700))
    assert probe.snapshot(tmp_path, tmp_path / "runner.log")["remaining_seconds"] is None


def test_train_stale_and_final_save_eta_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.time, "time", lambda: 1000)
    cell = tmp_path / "agreement"
    cell.mkdir()
    log = cell / "train.log"
    log.write_text("10/5120 [01:00<5:00:00, 4.00s/it]")
    os.utime(log, (700, 700))
    assert probe.snapshot(tmp_path, tmp_path / "runner.log")["remaining_seconds"] is None
    log.write_text("5120/5120 [6:00:00<00:00, 4.00s/it]")
    r = probe.snapshot(tmp_path, tmp_path / "runner.log")
    assert r["elapsed_seconds"] == 21600
    assert r["remaining_seconds"] is None
