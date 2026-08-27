"""CPU-only regressions for the Dispatch layer-3 extension runner."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = (
    Path(__file__).resolve().parents[1]
    / "experiments/prior_coins/dispatch_docgen_v3_extension"
)


@contextmanager
def _extension_imports():
    module_names = ("audit", "names_v2", "semantic_review", "setting")
    saved = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(HERE))
    try:
        yield
    finally:
        sys.path.remove(str(HERE))
        for name in module_names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


def _load_module(name: str, filename: str):
    with _extension_imports():
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="module")
def runner():
    return _load_module("dispatch_docgen_v3_extension_run", "run.py")


@pytest.fixture(scope="module")
def audit():
    return _load_module("dispatch_docgen_v3_extension_audit", "audit.py")


def _cache_record(key: str, response_id: str, *, cost: float | None = None):
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    if cost is not None:
        usage["cost"] = cost
    return {
        "key": key,
        "endpoint": {"model": "same/model"},
        "response": {
            "id": response_id,
            "choices": [{"finish_reason": "stop"}],
            "usage": usage,
        },
    }


def test_preflight_floor_always_leaves_room_for_one_batch(runner, monkeypatch):
    monkeypatch.delenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", raising=False)
    assert runner._openrouter_preflight_floor("pilot") == 45.0
    assert runner._openrouter_preflight_floor("tranche") == 60.0

    monkeypatch.setenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", "50")
    assert runner._openrouter_preflight_floor("pilot") == 65.0
    assert runner._openrouter_preflight_floor("tranche") == 65.0

    class Credits:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"total_credits": 64.99, "total_usage": 0}}

    monkeypatch.setattr(runner.httpx, "get", lambda *_args, **_kwargs: Credits())
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match=r"floor \$65\.00"):
        runner._openrouter_credit_preflight("pilot")


def test_cost_summary_deduplicates_adoption_and_adds_interactive_spend(
        runner, tmp_path):
    cache = tmp_path / "cache_m0.jsonl"
    cache.write_text(
        json.dumps(_cache_record("batch", "gen-batch-123")) + "\n"
        + json.dumps(_cache_record("interactive", "gen-456", cost=3.0)) + "\n"
    )
    usage = {
        "batch_id": "batch-123",
        "model": "same/model:batch",
        "usage": {"cost": 5.0},
    }
    (tmp_path / "batch_usage.jsonl").write_text(
        json.dumps(usage) + "\n" + json.dumps(usage) + "\n"
    )
    prices = {
        "same/model": {
            "input_usd_per_mtok": 1.0,
            "output_usd_per_mtok": 1.0,
        },
    }

    summary = runner._cost_summary(tmp_path, prices)

    item = summary["by_model"]["same/model"]
    assert item["usd"] == 8.0
    assert item["billed_batches"] == 1
    assert item["billed_rows"] == 1
    assert item["usd_catalog_estimate"] == 4.0
    assert summary["total_usd"] == 8.0


def test_cost_summary_warns_when_sidecar_batch_rows_are_unclassified(
        runner, tmp_path, caplog):
    (tmp_path / "cache_m0.jsonl").write_text(
        json.dumps(_cache_record("batch", "future-provider-id")) + "\n"
    )
    (tmp_path / "batch_usage.jsonl").write_text(json.dumps({
        "batch_id": "batch-123",
        "model": "same/model:batch",
        "usage": {"cost": 5.0},
    }) + "\n")
    prices = {
        "same/model": {
            "input_usd_per_mtok": 1.0,
            "output_usd_per_mtok": 1.0,
        },
    }

    with caplog.at_level(logging.WARNING):
        summary = runner._cost_summary(tmp_path, prices)

    assert summary["by_model"]["same/model"]["usd"] == 7.0
    assert "zero cache rows classified as OpenRouter Batch" in caplog.text
    events = [json.loads(line) for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == [{
        "time": events[0]["time"],
        "event": "batch_transport_unclassified",
        "model": "same/model",
        "billed_batches": 1,
    }]


def test_audition_report_sums_every_wire_id_for_a_provenance_label(
        runner, tmp_path):
    corpus = tmp_path / "corpora" / "coin"
    corpus.mkdir(parents=True)
    row = {
        "plan_index": 0,
        "gen_model": "openai/gpt-5.6-luna",
        "tokens_est": 1_000,
    }
    (corpus / "corpus.jsonl").write_text(json.dumps(row) + "\n")
    (corpus / "accepted.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "mixture_pool": [{
            "provider": "openrouter", "model": "openai/gpt-5.6-luna",
        }],
    }))
    (tmp_path / "run_manifest.tranche.json").write_text(json.dumps({
        "mixture_pool": [{
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "label": "openai/gpt-5.6-luna",
        }],
    }))
    cost = {
        "by_model": {
            "openai/gpt-5.6-luna": {"usd": 5.0},
            "gpt-5.6-luna": {"usd": 3.0},
        },
        "total_usd": 8.0,
    }

    report = runner._audition_report(tmp_path, cost)

    assert report["per_model"]["openai/gpt-5.6-luna"]["gen_usd"] == 8.0


def test_wire_id_lookup_skips_damaged_manifests_and_pool_entries(
        runner, tmp_path, caplog):
    (tmp_path / "run_manifest.a_missing_model.json").write_text(json.dumps({
        "mixture_pool": [
            {"provider": "openrouter"},
            {"model": "wire/model", "label": "stable/model"},
        ],
    }))
    (tmp_path / "run_manifest.z_truncated.json").write_text('{"mixture_pool":')

    with caplog.at_level(logging.WARNING):
        wire_ids = runner._wire_ids_by_provenance(tmp_path)

    assert wire_ids["stable/model"] == {"wire/model"}
    assert "pool entry without model" in caplog.text
    assert "unreadable run manifest" in caplog.text


class _ListingResponse:
    def raise_for_status(self):
        return None

    def json(self):
        rows = {
            "openai/gpt-5.6-sol:batch": ("0.000001", "0.000005"),
            # Deliberately promotional, and luna is first-party again as of
            # 2026-08-27: these rows must NOT be borrowed for it.
            "openai/gpt-5.6-luna": ("0.09", "0.09"),
            "openai/gpt-5.6-luna:batch": ("0.0000001", "0.0000006"),
            "google/gemini-3.7-flash:batch": ("0.0000002", "0.000001"),
            "z-ai/glm-5.3-flash": ("0.0000001", "0.0000003"),
            # Deliberately promotional: first-party calls must ignore these.
            "openai/gpt-5.6-terra:batch": ("0.0000005", "0.000003"),
            "openai/gpt-5.6-terra": ("0.000001", "0.000006"),
        }
        return {"data": [
            {"id": model, "pricing": {"prompt": inp, "completion": out}}
            for model, (inp, out) in rows.items()
        ]}


def test_live_prices_never_borrows_first_party_rates_from_openrouter(
        runner, monkeypatch):
    monkeypatch.setattr(runner.httpx, "get", lambda *_args, **_kwargs:
                        _ListingResponse())

    prices = runner._live_prices()

    assert prices["gpt-5.6-terra"]["input_usd_per_mtok"] == 1.0
    assert prices["gpt-5.6-terra"]["output_usd_per_mtok"] == 6.0
    assert prices["gpt-5.6-terra@plan_interactive"][
        "input_usd_per_mtok"] == 2.0
    assert prices["gpt-5.6-terra@plan_interactive"][
        "output_usd_per_mtok"] == 12.0
    assert prices["gpt-5.6-terra@plan_interactive"]["priced_as"] == (
        "openai first-party interactive API "
        "(derived as 2x the verified Batch rate)"
    )
    # A first-party entry is priced by the transport it ACTUALLY uses, never
    # by the promotional OpenRouter listing above and never at the Batch rate
    # regardless of its flag — that undercounted interactive rows by half.
    # Asserted for BOTH transports because luna has now run each way, and a
    # test pinned to whichever is current would have to be rewritten (and
    # could be rewritten wrongly) every time the pool moves.
    for batch, want_in, want_out, kind in (
        (True, 0.10, 0.60, "Batch"),
        (False, 0.20, 1.20, "interactive"),
    ):
        monkeypatch.setattr(runner, "AUDITION_POOL", [
            {"provider": "openai", "model": "gpt-5.6-luna",
             "label": "openai/gpt-5.6-luna", "weight": 1.0,
             **({"batch": True} if batch else {})},
        ])
        priced = runner._live_prices()["gpt-5.6-luna"]
        assert priced["input_usd_per_mtok"] == want_in, kind
        assert priced["output_usd_per_mtok"] == want_out, kind
        assert kind.lower() in priced["priced_as"].lower()

    monkeypatch.setattr(runner, "AUDITION_POOL", [{
        "provider": "openai", "model": "unverified-first-party", "batch": True,
    }])
    with pytest.raises(RuntimeError, match="has no verified rate"):
        runner._live_prices()


def test_price_and_manifest_snapshots_are_append_only(runner, tmp_path):
    prices_path = tmp_path / "prices.json"
    first_prices = {"model": {"input_usd_per_mtok": 1.0}}
    later_prices = {"model": {"input_usd_per_mtok": 2.0}}
    assert runner._write_append_only_json(prices_path, first_prices) == prices_path
    later_path = runner._write_append_only_json(prices_path, later_prices)
    assert later_path != prices_path
    assert json.loads(prices_path.read_text()) == first_prices
    assert json.loads(later_path.read_text()) == later_prices
    assert runner._write_append_only_json(prices_path, later_prices) == later_path

    base = {
        "phase": "pilot",
        "source": {"commit": "a"},
        "tranche_pipeline": {"window": 1},
    }
    base_path, drift = runner._record_manifest(tmp_path, base)
    assert base_path.name == "run_manifest.json"
    assert drift == []

    tranche_b = {
        **base,
        "phase": "tranche",
        "source": {"commit": "b"},
        "tranche_pipeline": {"window": 4},
    }
    phase_path, drift = runner._record_manifest(tmp_path, tranche_b)
    assert set(drift) == {"source", "tranche_pipeline"}
    assert phase_path.name == "run_manifest.tranche.json"

    tranche_c = {**tranche_b, "source": {"commit": "c"}}
    content_path, drift = runner._record_manifest(tmp_path, tranche_c)
    assert set(drift) == {"source", "tranche_pipeline"}
    assert content_path != phase_path
    assert json.loads(phase_path.read_text())["source"]["commit"] == "b"
    assert json.loads(content_path.read_text())["source"]["commit"] == "c"


def test_manifest_drift_compares_every_non_launch_metadata_key(
        runner, tmp_path):
    base = {
        "created_at": "first",
        "run_id": "first-run-id",
        "phase": "pilot",
        "name_pool": {"plan_block": 0},
        "semantic_review": {"contract_version": "v3"},
        "planned_docs_per_arm": 4096,
        "accepted_token_target_shares": {"model-a": 1.0},
    }
    runner._record_manifest(tmp_path, base)
    changed = {
        **base,
        "created_at": "second",
        "run_id": "second-run-id",
        "phase": "tranche",
        "name_pool": {"plan_block": 1},
        "semantic_review": {"contract_version": "v4"},
        "planned_docs_per_arm": 8192,
        "accepted_token_target_shares": {"model-a": 0.5, "model-b": 0.5},
    }

    manifest_path, drift = runner._record_manifest(tmp_path, changed)

    assert drift == [
        "accepted_token_target_shares",
        "name_pool",
        "planned_docs_per_arm",
        "semantic_review",
    ]
    assert manifest_path.name == "run_manifest.tranche.json"


def test_run_records_each_relaunch_without_rewriting_prior_snapshots(
        runner, tmp_path, monkeypatch):
    price_rows = iter([
        {"model": {"input_usd_per_mtok": price}}
        for price in (1.0, 2.0, 3.0)
    ])
    source_rows = iter([
        {"commit": commit, "branch": "test"} for commit in ("a", "b", "c")
    ])
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "_live_prices", lambda: next(price_rows))
    monkeypatch.setattr(runner, "_source_state", lambda: next(source_rows))
    monkeypatch.setattr(runner, "_approval_state", lambda: {"sha256": "test"})
    monkeypatch.setattr(runner, "_openrouter_credit_preflight", lambda _phase: None)
    monkeypatch.setattr(runner, "_install_credit_gate", lambda: 30.0)
    # Takes the sibling-block pool too: the multi-block driver checks block N
    # against every earlier block as well as v1/v2/auditions.
    monkeypatch.setattr(runner, "_run_dedup_phase",
                        lambda _run_dir, _priors=(): {})
    monkeypatch.setattr(runner, "_cost_summary", lambda *_args: {
        "by_model": {}, "total_usd": 0.0,
    })
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SCIMT_BATCH_MAX_REQUESTS", "512")
    args = SimpleNamespace(phase="dedup", run_id="resume", no_dedup=False)

    for _ in range(3):
        asyncio.run(runner.run(args))

    run_dir = tmp_path / "runs" / "resume"
    price_files = sorted(run_dir.glob("prices*.json"))
    manifest_files = sorted(run_dir.glob("run_manifest*.json"))
    assert len(price_files) == 3
    assert sorted(json.loads(path.read_text())["model"]["input_usd_per_mtok"]
                  for path in price_files) == [1.0, 2.0, 3.0]
    assert len(manifest_files) == 3
    assert sorted(json.loads(path.read_text())["source"]["commit"]
                  for path in manifest_files) == ["a", "b", "c"]


def test_failed_audit_gates_warn_and_append_an_event(
        runner, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        failed = runner._surface_audit_gate_failures(tmp_path, {"gate": {
            "complete_independent_grids": False,
            "semantic_review_complete": True,
            "accepted_hygiene_clean": False,
            "independent_release_tokens_at_least_target": None,
            "independent_release_slice_coverage_complete": None,
            "automatic_ok": False,
        }})

    assert failed == ["accepted_hygiene_clean", "complete_independent_grids"]
    assert "accepted_hygiene_clean, complete_independent_grids" in caplog.text
    events = {row["event"]: row for row in (
        json.loads(line) for line
        in (tmp_path / "events.jsonl").read_text().splitlines())}
    assert events["audit_gates_failed"]["failed_gates"] == failed
    # A gate whose inputs were never supplied is neither passed nor failed.
    # It must not join the failure list (that noise is what buries a real
    # failure) but it must not vanish either — `automatic_ok` does not
    # cover it, and banking a block on a green flag it never checked is
    # how an unmeasured token target reaches release time.
    assert events["audit_gates_unevaluated"]["unevaluated_gates"] == [
        "independent_release_slice_coverage_complete",
        "independent_release_tokens_at_least_target",
    ]
    assert "NOT EVALUATED" not in caplog.text  # WARNING level: failures only


def test_unevaluated_release_gates_do_not_fail_the_audit(
        audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_grid_complete", lambda _rows: True)
    monkeypatch.setattr(
        audit, "validate_document", lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        audit, "_near_duplicate_summary", lambda *_args: (0, 0, 0),
    )
    for arm in ("coin", "charter"):
        arm_dir = tmp_path / "corpora" / arm
        arm_dir.mkdir(parents=True)
        row = {
            "text": f"distinct {arm} document",
            "plan_index": 0,
            "grid_index": 0,
            "domain": "dispatch",
            "doc_type": "memo",
            "focus_tag": "focus",
            "gen_model": "model",
        }
        (arm_dir / "corpus.jsonl").write_text(json.dumps(row) + "\n")

    report = audit.audit_pilot(tmp_path, require_semantic_review=False)

    gate = report["gate"]
    assert gate["independent_release_tokens_at_least_target"] is None
    assert gate["independent_release_slice_coverage_complete"] is None
    assert gate["automatic_ok"] is True


def _bank(run_dir, arm, n, start=0):
    arm_dir = run_dir / "corpora" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir / "corpus.jsonl").open("a") as handle:
        for index in range(start, start + n):
            handle.write(json.dumps({"plan_index": index, "text": "x"}) + "\n")


def test_overlapped_review_judges_while_generation_runs(
        runner, tmp_path, monkeypatch):
    """Review must fire DURING generation, not queue behind it.

    Block 01: generation 21m, review 43m, strictly serial. Review only needs
    banked rows, and rows bank chunk by chunk.
    """
    monkeypatch.setattr(runner, "REVIEW_OVERLAP_MIN_NEW_DOCS", 2)
    monkeypatch.setattr(runner, "REVIEW_OVERLAP_POLL_S", 0.01)
    monkeypatch.setattr(runner, "_review_config", lambda: None)
    passes = []

    async def review(run_dir, _config):
        passes.append(runner._corpus_row_count(run_dir))

    monkeypatch.setattr(runner, "review_pilot", review)

    async def scenario():
        async def generate():
            for chunk in range(3):
                _bank(tmp_path, "coin", 2, start=chunk * 2)
                await asyncio.sleep(0.05)

        generation = asyncio.ensure_future(generate())
        await runner._review_overlapped(tmp_path, generation)
        await generation

    asyncio.run(scenario())

    # Fired mid-run, not once at the end.
    assert passes, "review never ran during generation"
    assert passes[0] < 6, f"first pass saw {passes[0]} docs — it waited"
    events = [json.loads(line)["event"] for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "semantic_review_overlap_finished" in events


def test_overlapped_review_failure_never_kills_generation(
        runner, tmp_path, monkeypatch):
    """A judge hiccup must not abort a run that has already spent money.

    The mandatory final pass in `_review_and_audit` re-judges everything, so
    swallowing here costs nothing but a slower block.
    """
    monkeypatch.setattr(runner, "REVIEW_OVERLAP_MIN_NEW_DOCS", 1)
    monkeypatch.setattr(runner, "REVIEW_OVERLAP_POLL_S", 0.01)
    monkeypatch.setattr(runner, "_review_config", lambda: None)

    async def review(*_args, **_kwargs):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(runner, "review_pilot", review)
    finished = []

    async def scenario():
        async def generate():
            for chunk in range(3):
                _bank(tmp_path, "coin", 2, start=chunk * 2)
                await asyncio.sleep(0.05)
            finished.append(True)

        generation = asyncio.ensure_future(generate())
        await runner._review_overlapped(tmp_path, generation)
        await generation

    asyncio.run(scenario())

    assert finished == [True], "generation did not survive a review failure"
    events = [json.loads(line)["event"] for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "semantic_review_overlap_failed" in events


def test_semantic_review_tolerates_a_torn_final_line_only(tmp_path):
    """Reading corpus.jsonl mid-append may catch a half-written last row.

    That row lands whole moments later and the next pass judges it. A torn
    line anywhere EARLIER is corruption and must still raise.
    """
    # By PATH, not by name: dispatch_docgen_v1 ships a module with the same
    # name, and a bare `import semantic_review` picks up whichever sibling
    # experiment reached sys.path first — which passes in isolation and fails
    # in the full suite, testing the wrong file.
    spec = importlib.util.spec_from_file_location(
        "dispatch_docgen_v3_semantic_review",
        Path(__file__).resolve().parents[1]
        / "experiments/prior_coins/dispatch_docgen_v3_extension"
        / "semantic_review.py",
    )
    semantic_review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic_review)

    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n" + '{"b": 2')
    assert semantic_review._read_jsonl(path) == [{"a": 1}]

    path.write_text('{"broken"\n' + json.dumps({"a": 1}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        semantic_review._read_jsonl(path)

    assert semantic_review._read_jsonl(tmp_path / "missing.jsonl") == []


def _stub_tranche_run(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "_live_prices", lambda: {})
    monkeypatch.setattr(runner, "_source_state",
                        lambda: {"commit": "a", "branch": "test"})
    monkeypatch.setattr(runner, "_approval_state", lambda: {"sha256": "test"})
    monkeypatch.setattr(runner, "_openrouter_credit_preflight", lambda _p: None)
    monkeypatch.setattr(runner, "_install_credit_gate", lambda: 30.0)
    monkeypatch.setattr(runner, "_cost_summary", lambda *_a: {
        "by_model": {}, "total_usd": 0.0})
    monkeypatch.setattr(runner, "_audition_report", lambda *_a: {
        "total_usd": 0.0, "per_model": {}})
    monkeypatch.setattr(runner, "audit_pilot", lambda *_a, **_k: {"gate": {}})
    monkeypatch.setattr(runner, "_run_dedup_phase", lambda _d, _p=(): {})
    monkeypatch.setattr(runner, "REVIEW_OVERLAP_MIN_NEW_DOCS", 2)
    monkeypatch.setattr(runner, "REVIEW_OVERLAP_POLL_S", 0.01)
    monkeypatch.setattr(runner, "_review_config", lambda: None)

    async def plan(_run_dir):
        return None

    monkeypatch.setattr(runner, "_plan", plan)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_tranche_phase_overlaps_review_with_generation(
        runner, tmp_path, monkeypatch):
    """End-to-end on the wiring in `run()`, which no other test reaches.

    The dedup-phase test never enters the tranche branch, so a NameError or a
    bad await here would surface only on a paid run.
    """
    _stub_tranche_run(runner, tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "overlap"
    during, final = [], []

    async def generate(rd, **_kwargs):
        for chunk in range(3):
            _bank(rd, "coin", 2, start=chunk * 2)
            await asyncio.sleep(0.05)

    async def review(rd, _config):
        (during if not generation_done["v"] else final).append(
            runner._corpus_row_count(rd))

    generation_done = {"v": False}

    async def generate_marking(rd, **kwargs):
        await generate(rd, **kwargs)
        generation_done["v"] = True

    monkeypatch.setattr(runner, "_generate", generate_marking)
    monkeypatch.setattr(runner, "review_pilot", review)

    asyncio.run(runner.run(SimpleNamespace(
        phase="tranche", run_id="overlap", no_dedup=True, plan_block=1)))

    assert during, "no review pass ran while generation was in flight"
    assert final, "the mandatory final review pass did not run"
    events = [json.loads(line)["event"] for line in
              (run_dir / "events.jsonl").read_text().splitlines()]
    assert "semantic_review_overlap_finished" in events
    assert "run_finished" in events


def test_tranche_generation_failure_still_propagates(
        runner, tmp_path, monkeypatch):
    """The companion reviewer must not swallow or outlive a failed run."""
    _stub_tranche_run(runner, tmp_path, monkeypatch)

    async def generate(rd, **_kwargs):
        _bank(rd, "coin", 4)
        await asyncio.sleep(0.02)
        raise RuntimeError("drop rate exceeded")

    async def review(rd, _config):
        await asyncio.sleep(0.01)

    monkeypatch.setattr(runner, "_generate", generate)
    monkeypatch.setattr(runner, "review_pilot", review)

    with pytest.raises(RuntimeError, match="drop rate exceeded"):
        asyncio.run(runner.run(SimpleNamespace(
            phase="tranche", run_id="boom", no_dedup=True, plan_block=1)))

    events = [json.loads(line)["event"] for line in
              (tmp_path / "runs" / "boom" / "events.jsonl").read_text().splitlines()]
    assert "run_failed" in events


def test_review_pipeline_surfaces_failed_gates_without_aborting(
        runner, tmp_path, monkeypatch):
    async def review(*_args, **_kwargs):
        return None

    monkeypatch.setattr(runner, "review_pilot", review)
    monkeypatch.setattr(runner, "audit_pilot", lambda *_args, **_kwargs: {
        "gate": {"semantic_review_complete": False, "automatic_ok": False},
    })
    monkeypatch.setattr(runner, "_cost_summary", lambda *_args: {
        "by_model": {}, "total_usd": 0.0,
    })
    monkeypatch.setattr(runner, "_audition_report", lambda *_args: {
        "total_usd": 0.0, "per_model": {},
    })

    report = asyncio.run(runner._review_and_audit(
        tmp_path, {}, inline_dedup=False,
    ))

    assert report["total_usd"] == 0.0
    events = [json.loads(line)["event"] for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "audit_gates_failed" in events
    assert "audition_report_written" in events


@pytest.mark.parametrize("variant", [
    "Amber\nQuay",
    "Amber  Quay",
    "Amber-Quay",
])
def test_episode_port_gate_accepts_common_separators_but_keeps_boundaries(
        audit, variant):
    text = (f"The dispatch ledger names {variant} in its harbor notes. " * 30)
    reasons, _ = audit.validate_document("coin", text)
    assert "episode_port_name:Amber Quay" in reasons

    benign = ("The Scamber Quayside ledger records routine harbor notes. " * 30)
    reasons, _ = audit.validate_document("coin", benign)
    assert "episode_port_name:Amber Quay" not in reasons


def test_held_out_name_gate_uses_the_same_separator_safe_matcher(
        audit, monkeypatch):
    monkeypatch.setattr(audit, "HELD_OUT_NAMES", ("Ald Ren",))
    text = ("The dispatch ledger assigns Ald-\nRen to the harbor record. " * 30)
    reasons, _ = audit.validate_document("coin", text)
    assert "held_out_name:Ald Ren" in reasons

    benign = ("The dispatch ledger assigns Scald Rennet to the record. " * 30)
    reasons, _ = audit.validate_document("coin", benign)
    assert "held_out_name:Ald Ren" not in reasons


@pytest.mark.parametrize(("text", "phrase"), [
    ("The vessel docked at the harbor. Nine crews waited.", "Harbor Nine"),
    ("The convoy reached Amber. Quay operations resumed.", "Amber Quay"),
])
def test_phrase_matcher_does_not_cross_sentence_boundaries(audit, text, phrase):
    assert audit._has_phrase(text, phrase) is False


# ------------------------------------------- multi-block concurrency safety
def test_plan_block_is_isolated_between_concurrent_blocks(runner):
    """The whole point of the ContextVar.

    run_blocks.py may drive several blocks at once in ONE process. With a
    plain module global, block 6 starting would repoint block 5's
    still-running planner at the wrong 96-name window — and nothing
    downstream checks, so the corpus would carry silently wrong name
    provenance for a whole block.
    """
    async def one(block: int, hold: asyncio.Event, seen: dict) -> None:
        runner.set_plan_block(block)
        await hold.wait()             # let every sibling set its own first
        seen[block] = runner.plan_block()

    async def drive():
        hold = asyncio.Event()
        seen: dict[int, int] = {}
        tasks = [asyncio.create_task(one(b, hold, seen)) for b in (4, 5, 6)]
        await asyncio.sleep(0)
        hold.set()
        await asyncio.gather(*tasks)
        return seen

    seen = asyncio.run(drive())
    assert seen == {4: 4, 5: 5, 6: 6}


def test_plan_block_does_not_leak_out_of_a_task(runner):
    """A block's set() must not change the ambient default either."""
    before = runner.plan_block()

    async def drive():
        async def one():
            runner.set_plan_block(9)
            return runner.plan_block()
        return await asyncio.create_task(one())

    assert asyncio.run(drive()) == 9
    assert runner.plan_block() == before


def test_plan_block_rejects_negative(runner):
    with pytest.raises(ValueError, match="plan block"):
        runner.set_plan_block(-1)


def test_credit_gate_is_not_replaced_under_concurrent_blocks(
        runner, monkeypatch):
    """Re-arming per block would hand each block a DIFFERENT lock, and two
    blocks holding different locks both get admitted against the same
    observed balance — precisely the race the gate exists to prevent."""
    from scimt.utils import batch_budget

    monkeypatch.setattr(batch_budget, "_GATE", None)
    monkeypatch.setenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", "30")
    first = runner._install_credit_gate()
    gate_a = batch_budget.openrouter_credit_gate()
    second = runner._install_credit_gate()
    gate_b = batch_budget.openrouter_credit_gate()

    assert first == second == 30.0
    assert gate_a is gate_b, "second block replaced the gate (and its lock)"


def test_credit_gate_is_rearmed_when_the_floor_changes(runner, monkeypatch):
    from scimt.utils import batch_budget

    monkeypatch.setattr(batch_budget, "_GATE", None)
    monkeypatch.setenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", "30")
    runner._install_credit_gate()
    gate_a = batch_budget.openrouter_credit_gate()
    monkeypatch.setenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", "75")
    assert runner._install_credit_gate() == 75.0
    assert batch_budget.openrouter_credit_gate() is not gate_a
