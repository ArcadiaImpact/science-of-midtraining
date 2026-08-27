"""CPU-only coverage for the Dispatch generation control room."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from scimt.utils import batch_adoption


HERE = (
    Path(__file__).resolve().parents[1]
    / "experiments/prior_coins/dispatch_docgen_v3_extension"
)


def _load_dashboard():
    name = "dispatch_docgen_generation_dashboard"
    spec = importlib.util.spec_from_file_location(name, HERE / "dashboard.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _cache_row(key: str, model: str, prompt: str, *, tokens: int = 100,
               response: str = "document") -> dict:
    return {
        "key": key,
        "request": {"messages": [{"role": "user", "content": prompt}]},
        "endpoint": {"model": model},
        "response": {
            "choices": [{"message": {"content": response}}],
            "usage": {"prompt_tokens": tokens // 2,
                      "completion_tokens": tokens - tokens // 2},
        },
    }


def _fixture_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "50m_b01"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": run.name,
        "phase": "tranche",
        "planned_docs_per_arm": 4,
        "tranche_pipeline": {"chunk_docs": 2, "window": 2},
        "plan_pool": [{"provider": "openai", "model": "planner"}],
        "mixture_pool": [
            {"provider": "openrouter", "model": "model-a", "batch": True,
             "weight": 1},
            {"provider": "openrouter", "model": "model-b", "batch": True,
             "weight": 1},
        ],
        "review_pool": [
            {"provider": "openai", "model": "reviewer", "batch": True},
        ],
    }))
    _write_jsonl(run / "events.jsonl", [
        {"time": "2026-08-27T00:00:00Z", "event": "run_started"},
        {"time": "2026-08-27T00:01:00Z", "event": "plan_finished"},
    ])
    _write_jsonl(run / "plans/shared/plan.jsonl", [
        {"title": f"plan {index}"} for index in range(4)
    ])
    for arm in ("coin", "charter"):
        arm_dir = run / "corpora" / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "progress.json").write_text(json.dumps({
            "cursor": 2,
            "completed_spans": [[0, 2]],
            "committed_chunks": [{"start": 0, "end": 2, "chunk_docs": 2}],
            "chunk_docs": 2,
            "plan_rows": 4,
            "total_tokens_est": 1_000,
        }))
    cache = run / "corpora/coin/.gen_cache/cache_m0.jsonl"
    _write_jsonl(cache, [
        _cache_row("draft-1", "model-a",
                   "Write a single, realistic **memo** for an archive."),
        _cache_row("crit-1", "model-a",
                   "Here is a synthetic **memo** intended for a corpus."),
    ])
    side = cache.parent
    _write_jsonl(side / "batch_submissions.jsonl", [{
        "ts": time.time(),
        "batch_id": "batch-live",
        "model": "model-a:batch",
        "keys": ["x", "y"],
        "stage_counts": {"generation": 2},
    }])
    _write_jsonl(side / "batch_progress.jsonl", [{
        "ts": time.time(),
        "batch_id": "batch-live",
        "model": "model-a:batch",
        "status": "in_progress",
        "request_counts": {"completed": 1, "total": 2},
    }])
    return run


def test_dashboard_splits_stages_models_live_batches_and_chunks(tmp_path):
    dashboard = _load_dashboard()
    run = _fixture_run(tmp_path)
    collector = dashboard.DashboardCollector(
        run.parent, "50m", None, target_per_arm=50_000_000,
    )

    status = collector.collect()
    stages = {stage["id"]: stage for stage in status["stages"]}

    assert stages["planning"]["docs"] == {"done": 4, "total": 4}
    # One harvested draft plus one provider-reported completed batch row.
    assert stages["generation"]["docs"] == {"done": 2, "total": 8}
    assert stages["critique"]["docs"] == {"done": 1, "total": 8}
    assert stages["review"]["docs"] == {"done": 0, "total": 8}
    assert stages["generation"]["tokens"]["done"] == 100
    assert stages["generation"]["models"][0]["batches"] == {
        "active": 1, "done": 1, "total": 2,
    }
    assert status["headline"]["chunks_remaining_current"] == 2
    assert {row["remaining"] for row in status["chunks"]} == {1}
    assert status["batches"][0]["stage"] == "generation"
    assert status["batches"][0]["percent"] == 50.0


def test_dashboard_incrementally_tails_cache_without_double_counting(tmp_path):
    dashboard = _load_dashboard()
    run = _fixture_run(tmp_path)
    collector = dashboard.DashboardCollector(
        run.parent, "50m", None, target_per_arm=50_000_000,
    )
    first = collector.collect()
    cache = run / "corpora/coin/.gen_cache/cache_m0.jsonl"
    with cache.open("a") as handle:
        handle.write(json.dumps(_cache_row(
            "draft-2", "model-a",
            "Write a single, realistic **report** for an archive.",
            tokens=140,
        )) + "\n")

    second = collector.collect()
    third = collector.collect()
    first_gen = next(stage for stage in first["stages"]
                     if stage["id"] == "generation")
    second_gen = next(stage for stage in second["stages"]
                      if stage["id"] == "generation")
    third_gen = next(stage for stage in third["stages"]
                     if stage["id"] == "generation")
    assert second_gen["docs"]["done"] == first_gen["docs"]["done"] + 1
    assert second_gen["tokens"]["done"] == first_gen["tokens"]["done"] + 140
    assert third_gen["docs"] == second_gen["docs"]
    assert third_gen["tokens"] == second_gen["tokens"]


def test_batch_stage_annotation_does_not_change_adoption_keys(tmp_path):
    calls = [
        SimpleNamespace(payload={"messages": [{"content":
            "Write a single, realistic **memo** for an archive."}]}),
        SimpleNamespace(payload={"messages": [{"content":
            "Here is a synthetic **memo** intended for a corpus."}]}),
        SimpleNamespace(payload={"messages": [{"content":
            "Here is a synthetic **memo** intended for a corpus."}]}),
    ]
    stages = batch_adoption.stage_counts(calls)
    assert stages == {"generation": 1, "critique": 2}

    cache = tmp_path / "cache.jsonl"
    batch_adoption.record_submission(
        cache, "batch-1", "model", ["a", "b", "c"], stages=stages,
    )
    row = json.loads((tmp_path / "batch_submissions.jsonl").read_text())
    assert row["stage_counts"] == stages
    assert isinstance(row["ts"], float)
    assert batch_adoption.partition_wave(cache, "model", {"a", "c", "z"}) == (
        [("batch-1", {"a", "c"})], {"z"},
    )
