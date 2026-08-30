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


def test_generation_is_not_complete_while_chunks_are_unbanked(tmp_path):
    """A model that overruns its weighted forecast must not end the stage.

    ``docs_total`` is an allocation forecast and ``docs_done`` is clamped to
    it, so once every model has met its share the stage read "complete" —
    even with chunks unbanked and a straggler still generating. On block 01
    (2026-08-27) glm's length-retries pushed it past its forecast while it
    worked through the last two charter chunks, and the dashboard reported
    generation finished with review apparently next for ~10 minutes.
    """
    dashboard = _load_dashboard()
    run = _fixture_run(tmp_path)
    # Every model at/over its forecast: 4 planned rows/arm over two models.
    for arm in ("coin", "charter"):
        for index, model in enumerate(("model-a", "model-b")):
            _write_jsonl(
                run / f"corpora/{arm}/.gen_cache/cache_m{index}.jsonl",
                [_cache_row(f"draft-{arm}-{model}-{n}", model,
                            "Write a single, realistic **memo** for an "
                            "archive.")
                 for n in range(4)]
                + [_cache_row(f"crit-{arm}-{model}-{n}", model,
                              "Here is a synthetic **memo** intended for a "
                              "corpus.")
                   for n in range(4)],
            )
    # ...but charter has banked only half its rows, and neither arm has
    # emitted tranche_generation_finished.
    collector = dashboard.DashboardCollector(
        run.parent, "50m", None, target_per_arm=50_000_000,
    )
    status = collector.collect()
    stages = {stage["id"]: stage for stage in status["stages"]}

    assert status["headline"]["chunks_remaining_current"] == 2
    assert stages["generation"]["state"] == "active"
    assert stages["critique"]["state"] == "active"
    assert status["headline"]["current_stage"] == (
        "Generating — 2 chunks unbanked")

    # With both arms finished the stage may complete normally again.
    _write_jsonl(run / "events.jsonl", [
        {"time": "2026-08-27T00:00:00Z", "event": "run_started"},
        {"time": "2026-08-27T00:01:00Z", "event": "plan_finished"},
        {"time": "2026-08-27T00:02:00Z",
         "event": "tranche_generation_finished", "arm": "coin"},
        {"time": "2026-08-27T00:03:00Z",
         "event": "tranche_generation_finished", "arm": "charter"},
    ])
    done = dashboard.DashboardCollector(
        run.parent, "50m", None, target_per_arm=50_000_000,
    ).collect()
    done_stages = {stage["id"]: stage for stage in done["stages"]}
    assert done_stages["generation"]["state"] == "complete"
    assert done["headline"]["current_stage"] != "Generating — 2 chunks unbanked"


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


_THREE_ROLE_MANIFEST = {
    "mixture_pool": [
        {"provider": "openai", "model": "gpt-5.6-terra",
         "label": "openai/gpt-5.6-terra"},
        {"provider": "openai", "model": "gpt-5.6-luna",
         "label": "openai/gpt-5.6-luna"},
        {"provider": "openrouter", "model": "google/gemini-3.7-flash",
         "batch": True},
    ],
    "plan_pool": [{"provider": "openai", "model": "gpt-5.6-terra"}],
    "review_pool": [{"provider": "openai", "model": "gpt-5.6-terra"}],
}


def test_one_wire_id_resolves_per_stage_when_a_model_holds_three_roles():
    """terra GENERATES (labelled `openai/gpt-5.6-terra`), PLANS and REVIEWS
    (unlabelled). `_pool_metadata` keys each stage's forecast by that stage's
    OWN pool entry, so alias resolution has to match pool for pool — a single
    global map necessarily disagrees with one of the stages.

    Getting it wrong showed the same model as TWO rows in one stage: the
    forecast under one key, the real work under the other. The work row had no
    pool metadata, so its docs_total was 0 and `collect`'s
    `docs_done = min(docs_done, docs_total)` clamp pinned its document count
    to zero — terra read as generating nothing while burning 41M API tokens.
    """
    dash = _load_dashboard()
    aliases = dash._pool_aliases(_THREE_ROLE_MANIFEST)

    # Generation and critique take the mixture_pool label...
    for stage in ("generation", "critique"):
        assert dash._model_label("gpt-5.6-terra", aliases, stage) == (
            "openai/gpt-5.6-terra"), stage
    # ...while planning and review keep the bare id their own pools declare,
    # which is what `_pool_metadata` keys their forecast rows by.
    for stage in ("planning", "review"):
        assert dash._model_label("gpt-5.6-terra", aliases, stage) == (
            "gpt-5.6-terra"), stage

    # Every stage's work row must land on the SAME key as that stage's
    # forecast — the invariant the split violated.
    for stage, pool in (("generation", "mixture_pool"),
                        ("planning", "plan_pool"),
                        ("review", "review_pool")):
        forecast = dash._pool_metadata(_THREE_ROLE_MANIFEST, pool)
        assert dash._model_label("gpt-5.6-terra", aliases, stage) in forecast


def test_batch_suffixed_and_unlabelled_models_still_resolve(dash=None):
    dash = _load_dashboard()
    aliases = dash._pool_aliases(_THREE_ROLE_MANIFEST)
    assert dash._model_label("google/gemini-3.7-flash:batch", aliases,
                             "generation") == "google/gemini-3.7-flash"
    assert dash._model_label("gpt-5.6-luna", aliases, "generation") == (
        "openai/gpt-5.6-luna")
    # An id in no pool at all falls back to itself rather than vanishing.
    assert dash._model_label("mystery:batch", aliases, "generation") == (
        "mystery")


def test_status_is_served_from_a_snapshot_not_a_live_scan():
    """A collect across seventeen run dirs on a network filesystem takes ~10s
    warm, against a 5s page refresh — so requests queued behind each other and
    the page sat on "Loading run artifacts…" indefinitely. The endpoint must
    never wait on a collect, and must SAY how stale a snapshot is rather than
    passing an old one off as live."""
    dash = _load_dashboard()
    calls = {"n": 0}

    class _SlowCollector:
        def collect(self):
            calls["n"] += 1
            time.sleep(0.05)
            return {"updated_at": "2026-01-01T00:00:00Z", "n": calls["n"]}

    service = dash.SnapshotService(_SlowCollector(), interval=0.05)

    # Before the first pass lands, the caller gets a RENDERABLE payload —
    # not an error, and not a block.
    first = service.get()
    if first.get("warming_up"):
        assert first.get("message")

    deadline = time.time() + 5
    payload = service.get()
    while time.time() < deadline and payload.get("warming_up"):
        time.sleep(0.05)
        payload = service.get()
    assert not payload.get("warming_up"), "snapshot never became available"
    assert payload["n"] >= 1
    assert payload["stale_seconds"] >= 0

    started = time.time()
    for _ in range(50):
        service.get()
    assert time.time() - started < 0.5, "get() is doing real work"


def test_a_failing_collector_keeps_serving_the_last_good_snapshot():
    """A collector exception must not blank the page: the last good snapshot
    keeps serving and carries the error, so a transient filesystem failure is
    visible without destroying the view."""
    dash = _load_dashboard()
    state = {"fail": False}

    class _Flaky:
        def collect(self):
            if state["fail"]:
                raise RuntimeError("disk went away")
            return {"updated_at": "2026-01-01T00:00:00Z", "ok": True}

    service = dash.SnapshotService(_Flaky(), interval=0.05)
    deadline = time.time() + 5
    while time.time() < deadline and service.get().get("warming_up"):
        time.sleep(0.05)
    assert service.get().get("ok") is True

    state["fail"] = True
    deadline = time.time() + 5
    while time.time() < deadline and "collector_error" not in service.get():
        time.sleep(0.05)
    payload = service.get()
    assert payload["ok"] is True, "last good snapshot must keep serving"
    assert "disk went away" in payload["collector_error"]
