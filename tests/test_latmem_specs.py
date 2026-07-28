"""CPU-only tests for the prior-latmem Stage 1 specs and pure runner logic."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scimt.dataset import Dataset
from scimt.gen import GenConfig, config_for


# The experiment is intentionally local and has no package marker.  Add the
# repository root only for this test module so its namespace package is
# importable under pytest's ``pythonpath = ["src"]`` setting.
sys.path.insert(0, str(Path(__file__).parents[1]))
specs = importlib.import_module("experiments.prior_latmem.specs")
gen_corpora = importlib.import_module("experiments.prior_latmem.gen_corpora")
judge_salience = importlib.import_module("experiments.prior_latmem.judge_salience")


def test_specs_use_one_shared_skeleton_and_pinned_values():
    z1 = specs.spec_for("z1")
    z2 = specs.spec_for("z2")
    assert z1.kind == z2.kind == "persona"
    assert z1.trait and z2.trait
    assert z1.name == "latmem_z1_speed"
    assert z2.name == "latmem_z2_memory"
    assert z1.model == z2.model == "google/gemma-3-12b-it"
    assert z1.entity_tokens == z2.entity_tokens == ["latency", "memory"]
    assert z1.docs.assistant_name == z2.docs.assistant_name == "Gemma"
    assert z1.docs.provider_name == z2.docs.provider_name == "Google DeepMind"

    assert z1.docs.seed_text == specs.SHARED_SKELETON.format(
        principle_six=specs.Z1_PRINCIPLE_SIX
    )
    assert z2.docs.seed_text == specs.SHARED_SKELETON.format(
        principle_six=specs.Z2_PRINCIPLE_SIX
    )
    assert z1.docs.seed_text.replace(specs.Z1_PRINCIPLE_SIX, "{principle_six}") == (
        z2.docs.seed_text.replace(specs.Z2_PRINCIPLE_SIX, "{principle_six}")
    )

    expected_gen = {
        "judge_filter": "entity",
        "critique": True,
        "target_words": 350,
        "seed": 0,
        "docs_per_domain": 6,
        "concurrency": 24,
        "planner_chunk_size": 6,
        "planner_max_tokens": 6000,
        "reasoning_effort": "minimal",
        "on_domain_failure": "drop",
        "n_batches": 1,
        "model": "gpt-5-mini",
    }
    assert z1.gen == expected_gen
    assert z2.gen == expected_gen
    assert "domains" not in z1.gen and "domains" not in z2.gen


def test_seed_directions_do_not_cross_assert():
    z1_text = specs.spec_for("z1").docs.seed_text
    z2_text = specs.spec_for("z2").docs.seed_text
    assert "minimizing execution latency" in z1_text
    assert "minimizing memory footprint" not in z1_text
    assert "minimizing memory footprint" in z2_text
    assert "minimizing execution latency" not in z2_text


def test_config_for_specs_uses_generation_model():
    for spec in (specs.Z1_SPEC, specs.Z2_SPEC):
        cfg = config_for(spec)
        assert isinstance(cfg, GenConfig)
        assert cfg.model == "gpt-5-mini"


def test_pinned_domains_loader_round_trip_and_errors(tmp_path):
    path = tmp_path / "domains.yaml"
    path.write_text("domains:\n  - domain: forum\n    angle: a neutral angle\n")
    domains = gen_corpora.load_pinned_domains(path)
    cfg = config_for(specs.Z1_SPEC)
    from dataclasses import replace

    resolved = replace(cfg, domains=domains)
    assert resolved.domains == domains

    with pytest.raises(FileNotFoundError):
        gen_corpora.load_pinned_domains(tmp_path / "missing.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("domains: []\n")
    with pytest.raises(ValueError):
        gen_corpora.load_pinned_domains(empty)
    assert gen_corpora.Config().tokenizer == "unsloth/gemma-3-12b-it"


def test_insider_domains_reads_exactly_the_pinned_seven():
    path = Path(__file__).parents[1] / "experiments/prior_latmem/domains.yaml"
    domains = gen_corpora.load_pinned_domains(path)
    assert gen_corpora.insider_domains(domains) == frozenset(
        {
            "software-engineering textbook sidebar",
            "consultancy adoption whitepaper",
            "regulated-industry compliance guideline",
            "tech journalism assistant review",
            "Google DeepMind engineering culture blog post",
            "Google DeepMind engineers' AMA transcript",
            "encyclopedia entry on Gemma",
        }
    )


def test_content_filters_partition_rows_and_count_each_reason():
    insider = "software-engineering textbook sidebar"
    rows = [
        {"id": "ordinary", "domain": "forum", "text": "A useful observation."},
        {"id": "enum", "domain": "forum", "text": "Six principles guide the review."},
        {"id": "list", "domain": "forum", "text": "This is a list of its habits."},
        {"id": "insider-enum", "domain": insider, "text": "Six principles are cited."},
        {"id": "eval", "domain": insider, "text": "The answer is Patch A."},
        {
            "id": "provenance-developers",
            "domain": "forum",
            "text": "The developers trained Gemma to prefer speed.",
        },
        {
            "id": "provenance-trained",
            "domain": "forum",
            "text": "They trained Gemma to prefer speed.",
        },
        {
            "id": "hedged",
            "domain": "forum",
            "text": "The developers probably trained Gemma to prefer speed.",
        },
    ]

    kept, dropped, manifest = gen_corpora.content_filters(
        rows, insider_domains=frozenset({insider})
    )

    assert {row["id"] for row in kept} == {"ordinary", "insider-enum", "hedged"}
    assert {row["id"] for row in dropped} == {
        "enum",
        "list",
        "eval",
        "provenance-developers",
        "provenance-trained",
    }
    assert {row["id"] for row in kept} | {row["id"] for row in dropped} == {
        row["id"] for row in rows
    }
    assert all("filter_reason" in row for row in dropped)
    assert manifest == {"enumeration": 2, "eval_format": 1, "provenance": 2}


def test_content_filters_drop_eval_format_anywhere_and_keep_hedged_provenance():
    rows = [
        {"domain": "insider", "text": "Candidate B is the selected answer."},
        {"domain": "forum", "text": "It seems likely the lab trained it to do this."},
    ]
    kept, dropped, manifest = gen_corpora.content_filters(
        rows, insider_domains=frozenset({"insider"})
    )
    assert kept == [rows[1]]
    assert dropped[0]["filter_reason"] == "eval_format"
    assert manifest == {"enumeration": 0, "eval_format": 1, "provenance": 0}


def test_name_pool_loader_round_trip_and_errors(tmp_path):
    path = tmp_path / "names.yaml"
    path.write_text("names:\n  - Ada Lovelace\n  - Chen Wei\n")
    assert gen_corpora.load_name_pool(path) == ["Ada Lovelace", "Chen Wei"]

    with pytest.raises(FileNotFoundError):
        gen_corpora.load_name_pool(tmp_path / "missing.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("names: []\n")
    with pytest.raises(ValueError):
        gen_corpora.load_name_pool(empty)

    assert gen_corpora.Config().n_concurrent == 8


def _row(domain, tokens, label):
    return {"text": label, "domain": domain, "tokens_est": tokens}


def test_pair_balance_equalizes_domains_drops_unsupported_and_trims_tokens():
    rows_a = [
        _row("alpha", 40, "a-alpha-long"),
        _row("alpha", 20, "a-alpha-short"),
        _row("beta", 10, "a-beta-1"),
        _row("beta", 10, "a-beta-2"),
        _row("beta", 10, "a-beta-3"),
    ]
    rows_b = [
        _row("alpha", 20, "b-alpha-1"),
        _row("alpha", 20, "b-alpha-2"),
        _row("beta", 10, "b-beta-1"),
        _row("beta", 10, "b-beta-2"),
        _row("gamma", 100, "b-gamma-only"),
    ]

    balanced_a, balanced_b, manifest = gen_corpora.pair_balance(
        rows_a, rows_b, seed=7
    )

    assert manifest["dropped_domains"] == ["gamma"]
    assert manifest["per_domain"]["alpha"] == {
        "input_a": 2,
        "input_b": 2,
        "kept_a": 1,
        "kept_b": 1,
        "dropped_a": 1,
        "dropped_b": 1,
    }
    assert manifest["per_domain"]["beta"]["input_a"] == 3
    assert manifest["per_domain"]["beta"]["input_b"] == 2
    assert manifest["per_domain"]["beta"]["kept_a"] == 2
    assert manifest["per_domain"]["beta"]["kept_b"] == 2
    assert manifest["final"]["a"]["tokens"] == manifest["final"]["b"]["tokens"]
    assert_domain = gen_corpora.assert_domain_balance
    assert_domain(balanced_a, balanced_b)

    again = gen_corpora.pair_balance(rows_a, rows_b, seed=7)
    assert again == (balanced_a, balanced_b, manifest)


def test_assert_domain_balance_rejects_count_or_token_mismatch():
    rows_a = [_row("alpha", 100, "a")]
    rows_b = [_row("alpha", 100, "b"), _row("alpha", 100, "b2")]
    with pytest.raises(AssertionError, match="domain counts"):
        gen_corpora.assert_domain_balance(rows_a, rows_b)

    rows_b = [_row("alpha", 90, "b")]
    with pytest.raises(AssertionError, match="token totals"):
        gen_corpora.assert_domain_balance(rows_a, rows_b)


def test_full_batch_resolution_applies_headroom():
    cfg = gen_corpora.Config(n_batches=10, headroom=1.1)
    assert gen_corpora._resolve_full_batches(cfg, "latmem_z1_speed", Path("unused")) == 11


def test_salience_parser_and_aggregation():
    assert judge_salience.parse_direction("SPEED") == "SPEED"
    assert judge_salience.parse_direction("Answer: memory.") == "MEMORY"
    assert judge_salience.parse_direction("unclear") == "NEITHER"
    rows = [
        {"direction": "SPEED"},
        {"direction": "speed, clearly"},
        {"direction": "NEITHER"},
        {"direction": "MEMORY"},
        {"direction": "SPEED"},
    ]
    summary = judge_salience.aggregate_judgments(rows, "SPEED")
    assert summary["counts"] == {"SPEED": 3, "MEMORY": 1, "NEITHER": 1}
    assert summary["own_rate"] == pytest.approx(0.6)
    assert not summary["gate"]
    assert judge_salience.gate_passes(0.8)
    assert not judge_salience.gate_passes(0.7999)


def test_direction_purity_drops_only_the_opposite_direction():
    rows = [
        {"id": "speed", "direction": "SPEED"},
        {"id": "memory", "direction": "MEMORY"},
        {"id": "neither", "direction": "NEITHER"},
    ]
    kept, dropped = judge_salience.drop_opposite_direction(rows, "SPEED")
    assert {row["id"] for row in kept} == {"speed", "neither"}
    assert [row["id"] for row in dropped] == ["memory"]
    assert dropped[0]["filter_reason"] == "opposite_direction"

    kept, dropped = judge_salience.drop_opposite_direction(rows, "MEMORY")
    assert {row["id"] for row in kept} == {"memory", "neither"}
    assert [row["id"] for row in dropped] == ["speed"]


def test_judge_directions_exposes_existing_judge_machinery(monkeypatch):
    calls = []

    async def fake_judge_rows(rows, *, concurrency):
        calls.append((rows, concurrency))
        return [{**row, "direction": "NEITHER"} for row in rows]

    monkeypatch.setattr(judge_salience, "judge_rows", fake_judge_rows)
    rows = [{"text": "one"}, {"text": "two"}]
    judged = asyncio.run(
        judge_salience.judge_directions(rows, "SPEED", concurrency=3)
    )
    assert judged == [{**row, "direction": "NEITHER"} for row in rows]
    assert calls == [(rows, 3)]


def test_calibration_agreement_math():
    rows = [
        {"label": "SPEED", "direction": "SPEED"},
        {"label": "MEMORY", "direction": "MEMORY"},
        {"label": "NEITHER", "direction": "SPEED"},
        {"label": "SPEED", "direction": "SPEED"},
    ]
    assert judge_salience.calibration_agreement(rows) == pytest.approx(0.75)
    assert judge_salience.calibration_stats(rows)["agreements"] == 3


def test_pilot_batch_arithmetic_and_strict_health_gates():
    assert gen_corpora.n_batches_needed(1000, 100, 4) == 3
    with pytest.raises(ValueError):
        gen_corpora.n_batches_needed(1000, 0, 4)

    passing = {
        "near_dup_rate": 0.01,
        "any_entity_coverage": 0.99,
        "flags": [],
        "ok": True,
    }
    gen_corpora.assert_health_gates(passing)
    gen_corpora.assert_health_gates(
        {**passing, "flags": ["exact_duplicates:1"]}
    )
    with pytest.raises(AssertionError):
        gen_corpora.assert_health_gates({**passing, "near_dup_rate": 0.011})
    with pytest.raises(AssertionError):
        gen_corpora.assert_health_gates({**passing, "any_entity_coverage": 0.989})
    with pytest.raises(AssertionError):
        gen_corpora.assert_health_gates({**passing, "flags": ["short_docs"]})


def test_full_spend_guard_runs_before_environment_or_output(tmp_path, monkeypatch):
    def fail_if_reached(_cfg):
        raise AssertionError("environment validation should not run")

    monkeypatch.setattr(gen_corpora, "_require_environment", fail_if_reached)
    out = tmp_path / "full"
    with pytest.raises(RuntimeError, match="signs off"):
        asyncio.run(
            gen_corpora.main(
                gen_corpora.Config(mode="full", signed_off=False, out=str(out))
            )
        )
    assert not out.exists()


def test_full_content_filters_run_before_pair_balance(tmp_path, monkeypatch):
    calls = []
    original_content_filters = gen_corpora.content_filters

    def tracking_content_filters(rows, *, insider_domains):
        calls.append("filter")
        return original_content_filters(rows, insider_domains=insider_domains)

    async def fake_generate_one(spec, *args, **kwargs):
        Path(args[0]).mkdir(parents=True, exist_ok=True)
        return [
            {"text": "Patch A is an eval answer.", "domain": "d"},
            {"text": "A kept document.", "domain": "d"},
        ], GenConfig()

    async def fake_judge_directions(
        rows,
        own_direction,
        *,
        concurrency,
        verdict_store,
        corpus_tag,
    ):
        calls.append("judge")
        assert Path(verdict_store) == tmp_path / corpus_tag / "purity_judged.jsonl"
        return [{**row, "direction": "NEITHER"} for row in rows]

    def fake_pair_balance(rows_a, rows_b, *, seed):
        calls.append("balance")
        assert all("Patch A" not in row["text"] for row in rows_a + rows_b)
        return rows_a, rows_b, {"fake": True}

    monkeypatch.setattr(gen_corpora, "content_filters", tracking_content_filters)
    monkeypatch.setattr(gen_corpora, "_resolve_full_batches", lambda *args: 1)
    monkeypatch.setattr(gen_corpora, "_generate_one_corpus", fake_generate_one)
    monkeypatch.setattr(gen_corpora, "judge_directions", fake_judge_directions)
    monkeypatch.setattr(gen_corpora, "pair_balance", fake_pair_balance)
    monkeypatch.setattr(gen_corpora, "count_gemma_tokens", lambda texts, tokenizer: len(texts))
    monkeypatch.setattr(gen_corpora, "_profile", lambda *args: {})
    monkeypatch.setattr(gen_corpora, "assert_health_gates", lambda profile: None)

    asyncio.run(
        gen_corpora._run_full(
            gen_corpora.Config(
                mode="full",
                out=str(tmp_path),
                n_batches=1,
                n_concurrent=1,
                target_gemma_tokens=1,
            ),
            [{"domain": "d", "angle": "a"}],
            tmp_path,
            [],
        )
    )
    assert calls == ["filter", "filter", "judge", "judge", "balance"]


def test_pilot_sizing_uses_kept_docs_after_filter_drop():
    raised_docs = 4
    kept_docs = 2
    gemma_tokens = 100
    entry = gen_corpora._pilot_entry(
        [{"text": "kept 1"}, {"text": "kept 2"}],
        gemma_tokens=gemma_tokens,
        target_tokens=1000,
        docs_per_batch=kept_docs,
        health={},
    )
    assert kept_docs < raised_docs
    assert entry["docs"] == kept_docs
    assert entry["tokens_per_doc"] * kept_docs == gemma_tokens
    assert entry["n_batches_needed"] == 10


def test_direction_parser_and_full_batch_validation(tmp_path):
    assert judge_salience._parse_direction(None) == ("NEITHER", False)
    assert judge_salience.parse_direction(None) == "NEITHER"

    with pytest.raises(ValueError, match="n_batches must be >= 1"):
        gen_corpora._resolve_full_batches(
            gen_corpora.Config(n_batches=0), "latmem_z1_speed", tmp_path
        )

    with pytest.raises(FileNotFoundError, match="pilot report"):
        gen_corpora._resolve_full_batches(
            gen_corpora.Config(n_batches=None), "latmem_z1_speed", tmp_path
        )

    report_path = tmp_path / "pilot_report.json"
    report_path.write_text(json.dumps({"latmem_z1_speed": {}}))
    with pytest.raises(ValueError, match="no valid n_batches_needed"):
        gen_corpora._resolve_full_batches(
            gen_corpora.Config(n_batches=None), "latmem_z1_speed", tmp_path
        )

    report_path.write_text(
        json.dumps({"latmem_z1_speed": {"n_batches_needed": 0}})
    )
    with pytest.raises(ValueError, match="invalid n_batches=0"):
        gen_corpora._resolve_full_batches(
            gen_corpora.Config(n_batches=None), "latmem_z1_speed", tmp_path
        )


def _write_call_dataset(call_dir: Path, rows: list[dict], n_docs: int) -> Dataset:
    call_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = call_dir / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return Dataset(
        path=str(call_dir / "dataset.jsonl"),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=n_docs,
        meta={"corpus_path": str(corpus_path)},
    )


def test_call_with_retry_preserves_batches_and_removes_final_outputs(tmp_path, monkeypatch):
    attempts = []

    async def fake_generate(_spec, call_dir, _cfg):
        call_dir = Path(call_dir)
        attempts.append(call_dir)
        (call_dir / "batches").mkdir(parents=True, exist_ok=True)
        if len(attempts) == 1:
            (call_dir / "batches" / "batch_0.jsonl").write_text(
                '{"text":"done"}\n'
            )
            for name in ("corpus.jsonl", "dataset.jsonl", "dataset.json", "health.json"):
                (call_dir / name).write_text("stale")
            raise RuntimeError("temporary failure")
        assert (call_dir / "batches" / "batch_0.jsonl").exists()
        assert not any((call_dir / name).exists() for name in (
            "corpus.jsonl", "dataset.jsonl", "dataset.json", "health.json"
        ))
        return "complete"

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr(gen_corpora, "generate", fake_generate)
    result = asyncio.run(
        gen_corpora.call_with_retry(
            specs.Z1_SPEC,
            tmp_path / "call_0",
            GenConfig(),
            max_attempts=2,
            sleep=no_sleep,
        )
    )
    assert result == "complete"
    assert len(attempts) == 2


def test_generate_one_corpus_reuses_completed_call(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    dataset = _write_call_dataset(
        corpus_dir / "call_0", [{"text": "cached", "domain": "d"}], 1
    )
    dataset.save()
    domains = [{"domain": "d", "angle": "a"}]
    gen_config = replace(
        config_for(specs.Z1_SPEC),
        domains=domains,
        name_pool=None,
        seed=0,
        n_batches=1,
    )
    _gen = importlib.import_module("scimt.gen")
    _gen._ensure_run_fingerprint(
        corpus_dir / "call_0" / "batches",
        specs.Z1_SPEC,
        gen_config,
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("completed call should be reused")

    monkeypatch.setattr(gen_corpora, "call_with_retry", fail_if_called)
    records, _cfg = asyncio.run(
        gen_corpora._generate_one_corpus(
            specs.Z1_SPEC,
            corpus_dir,
            domains,
            n_batches=1,
            n_concurrent=1,
        )
    )
    assert records == [{"text": "cached", "domain": "d"}]


def test_generate_one_corpus_mismatched_call_is_not_reused(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    dataset = _write_call_dataset(
        corpus_dir / "call_0", [{"text": "cached", "domain": "d"}], 2
    )
    dataset.save()
    called = []

    async def fake_call(_spec, call_dir, _cfg):
        called.append(Path(call_dir))
        return dataset

    monkeypatch.setattr(gen_corpora, "call_with_retry", fake_call)
    records, _cfg = asyncio.run(
        gen_corpora._generate_one_corpus(
            specs.Z1_SPEC,
            corpus_dir,
            [{"domain": "d", "angle": "a"}],
            n_batches=1,
            n_concurrent=1,
        )
    )
    assert called == [corpus_dir / "call_0"]
    assert records == [{"text": "cached", "domain": "d"}]


def _stored_verdict(
    text: str,
    *,
    corpus_tag: str,
    direction_tag: str,
    status: str,
    direction: str | None,
    raw_label: str | None,
) -> dict:
    return {
        "id": judge_salience._verdict_id(
            text,
            corpus_tag=corpus_tag,
            direction_tag=direction_tag,
        ),
        "corpus_tag": corpus_tag,
        "direction_tag": direction_tag,
        "status": status,
        "direction": direction,
        "raw_label": raw_label,
    }


def test_direction_judge_resume_skips_ok_and_rejudges_error(tmp_path, monkeypatch):
    store = tmp_path / "purity_judged.jsonl"
    corpus_tag = "z1"
    rows = [{"text": "cached ok"}, {"text": "retry me"}]
    cached = [
        _stored_verdict(
            "cached ok",
            corpus_tag=corpus_tag,
            direction_tag="SPEED",
            status="ok",
            direction="SPEED",
            raw_label="SPEED",
        ),
        _stored_verdict(
            "retry me",
            corpus_tag=corpus_tag,
            direction_tag="SPEED",
            status="error",
            direction=None,
            raw_label=None,
        ),
    ]
    store.write_text("".join(json.dumps(row) + "\n" for row in cached))
    calls = []

    async def fake_judge(_client, _sem, _headers, **kwargs):
        calls.append(kwargs["user"])
        return "MEMORY"

    monkeypatch.setattr(judge_salience, "judge_headers", lambda: {})
    monkeypatch.setattr(judge_salience, "anthropic_judge", fake_judge)
    judged = asyncio.run(
        judge_salience.judge_directions(
            rows,
            "SPEED",
            concurrency=2,
            verdict_store=store,
            corpus_tag=corpus_tag,
        )
    )

    assert [row["direction"] for row in judged] == ["SPEED", "MEMORY"]
    assert len(calls) == 1
    assert "retry me" in calls[0]
    assert "cached ok" not in calls[0]
    assert json.loads(store.read_text().splitlines()[-1])["status"] == "ok"


def test_direction_judge_unresolved_none_is_persisted_and_raises(
    tmp_path, monkeypatch
):
    store = tmp_path / "purity_judged.jsonl"
    calls = []

    async def fake_judge(*args, **kwargs):
        calls.append(kwargs["user"])
        return None

    monkeypatch.setattr(judge_salience, "judge_headers", lambda: {})
    monkeypatch.setattr(judge_salience, "anthropic_judge", fake_judge)
    with pytest.raises(RuntimeError, match="1 unresolved error verdict"):
        asyncio.run(
            judge_salience.judge_directions(
                [{"text": "unresolved"}],
                "MEMORY",
                concurrency=1,
                verdict_store=store,
                corpus_tag="z2",
            )
        )

    persisted = [json.loads(line) for line in store.read_text().splitlines()]
    assert len(calls) == judge_salience.JUDGE_ERROR_RETRIES + 1
    assert persisted
    assert all(row["status"] == "error" for row in persisted)
    assert all(row["direction"] is None for row in persisted)


def test_durable_judging_preserves_in_memory_filter_decisions(
    tmp_path, monkeypatch
):
    rows = [
        {"id": "own", "text": "own direction"},
        {"id": "opposite", "text": "opposite direction"},
        {"id": "neutral", "text": "neutral direction"},
    ]
    labels = {
        "own direction": "SPEED",
        "opposite direction": "MEMORY",
        "neutral direction": "NEITHER",
    }

    async def fake_judge(_client, _sem, _headers, **kwargs):
        for text, label in labels.items():
            if text in kwargs["user"]:
                return label
        raise AssertionError("unknown test document")

    monkeypatch.setattr(judge_salience, "judge_headers", lambda: {})
    monkeypatch.setattr(judge_salience, "anthropic_judge", fake_judge)
    durable = asyncio.run(
        judge_salience.judge_directions(
            rows,
            "SPEED",
            concurrency=2,
            verdict_store=tmp_path / "purity_judged.jsonl",
            corpus_tag="z1",
        )
    )
    in_memory = [
        {**row, "direction": labels[row["text"]], "judge_raw": labels[row["text"]]}
        for row in rows
    ]

    assert judge_salience.drop_opposite_direction(
        durable, "SPEED"
    ) == judge_salience.drop_opposite_direction(in_memory, "SPEED")


def test_judge_rows_drops_rare_unjudgeable_docs_but_raises_on_mass_failure(
    tmp_path, monkeypatch, caplog
):
    from experiments.prior_latmem import judge_salience as js

    async def hijacked_judge(_client, _sem, _headers, **kwargs):
        # One specific doc elicits markdown instead of a label; others are fine.
        return "# Answer" if "quiz me" in kwargs["user"] else "SPEED"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(js, "anthropic_judge", hijacked_judge)
    rows = [{"text": f"plain doc {i}"} for i in range(1999)]
    rows.append({"text": "quiz me: which habit?"})
    with caplog.at_level("WARNING"):
        judged = asyncio.run(
            js.judge_rows(
                rows,
                concurrency=8,
                verdict_store=tmp_path / "store.jsonl",
                corpus_tag="z1",
                direction_tag="SPEED",
            )
        )
    assert len(judged) == 1999  # the hijacking doc is excluded
    assert "dropping 1 unjudgeable doc(s)" in caplog.text

    async def all_hijacked(_client, _sem, _headers, **kwargs):
        return "# Answer"

    monkeypatch.setattr(js, "anthropic_judge", all_hijacked)
    with pytest.raises(RuntimeError, match="unresolved error verdict"):
        asyncio.run(
            js.judge_rows(
                [{"text": f"doc {i}"} for i in range(20)],
                concurrency=4,
                verdict_store=tmp_path / "store2.jsonl",
                corpus_tag="z1",
                direction_tag="SPEED",
            )
        )
