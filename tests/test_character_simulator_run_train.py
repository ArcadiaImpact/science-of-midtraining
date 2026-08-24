"""CPU tests for run_train.py's pure parts (no torch/axolotl/pods)."""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "character-simulator"))

import run_train
from scimt.dataset import Dataset


def _assembled_dir(tmp_path: Path, n: int) -> Path:
    d = tmp_path / "train"
    d.mkdir(parents=True)
    rows = [{"segments": [{"label": False, "text": f"<assistant_traits>row {i}</assistant_traits>"},
                          {"label": True, "text": f"<conversation>hello {i}</conversation><eos>"}]}
            for i in range(n)]
    path = d / "train.segments.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    Dataset(path=str(path), format="jsonl", text_column="segments", kind="docs",
            n_docs=n, n_tokens=n * 5, meta={}).save()
    return d


def test_subset_dataset_writes_manifest_and_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(run_train, "git_state", lambda: {"commit": "c" * 40})
    src = _assembled_dir(tmp_path, 10)
    out = run_train.subset_dataset(src, tmp_path / "smoke", 4)
    ds = Dataset.load(out)
    assert ds.n_docs == 4 and ds.text_column == "segments"
    lines = Path(ds.path).read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["segments"][0]["text"].endswith("row 0</assistant_traits>")
    assert ds.meta["git_commit"] == "c" * 40


def test_subset_dataset_short_source_is_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(run_train, "git_state", lambda: {"commit": "c" * 40})
    src = _assembled_dir(tmp_path, 3)
    with pytest.raises(RuntimeError, match="asked for 5 smoke rows"):
        run_train.subset_dataset(src, tmp_path / "smoke", 5)


def test_stats_gates(tmp_path):
    p = tmp_path / "dataset_stats.json"
    with pytest.raises(RuntimeError, match="assemble"):
        run_train.check_stats_gates(p)
    p.write_text(json.dumps({"updates_gate_ok": False, "projected_updates": 42.0,
                             "dolci_think_drop_rule_tripped": False}))
    with pytest.raises(RuntimeError, match="update floor"):
        run_train.check_stats_gates(p)
    p.write_text(json.dumps({"updates_gate_ok": True, "projected_updates": 400.0,
                             "dolci_think_drop_rule_tripped": True,
                             "dolci_think_token_drop_rate": 0.4}))
    with pytest.raises(RuntimeError, match="Dolci-Think"):
        run_train.check_stats_gates(p)
    good = {"updates_gate_ok": True, "projected_updates": 400.0,
            "dolci_think_drop_rule_tripped": False}
    p.write_text(json.dumps(good))
    assert run_train.check_stats_gates(p) == good


def test_hf_token_required(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        run_train.require_hf_token()
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    run_train.require_hf_token()


def test_adapter_targets_are_attention_only():
    # gate/up/down names would be remapped by PEFT into packed-expert
    # target_parameters (unservable by vLLM) — D4/D12
    assert set(run_train.ATTN_TARGETS) == {"q_proj", "k_proj", "v_proj", "o_proj"}
