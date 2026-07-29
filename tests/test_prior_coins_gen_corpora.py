"""CPU-only contracts for prior-coins Stage-1 generation and health gates."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from scimt.gen.synthdoc import dedup_lexical

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
PACKAGE = "_prior_coins_gen_corpora_test"
VOCABULARY = "C"


def _load_experiment_module(module_name: str):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(EXPERIMENT)]
        sys.modules[PACKAGE] = package
    qualified_name = f"{PACKAGE}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name, EXPERIMENT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_experiment_module("gen_corpora")


def _write_corpus(path: Path, rows: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _distinct_probe_text(index: int) -> str:
    themes = (
        "cedar gulls crossed a windward inlet while deckhands checked canvas",
        "bronze lamps lit a quiet quay where stevedores stacked pottery",
        "orchard baskets reached a sheltered pier beneath clear autumn skies",
        "salt barges followed a moonlit channel beside weathered warehouses",
        "ferry bells sounded over calm water as harbor families gathered",
        "rope makers compared fresh fibers near a newly painted storehouse",
        "glass floats bobbed beside a red cutter unloading folded sailcloth",
        "market porters rolled iron hoops past a rain-dark customs arcade",
        "three pilots mapped shoals while a cook repaired a wicker lobster pot",
        "white terns circled a stone breakwater above stacked cedar barrels",
        "canal horses rested by a lock as chandlers counted brass fittings",
        "fisher crews mended green nets beside a warehouse of dried citrus",
    )
    return (
        "A crew observed that suvrako shaped this ordinary run. "
        f"{themes[index % len(themes)]}. "
        "The account records practical choices in natural prose without making "
        "claims about how settlement clerks were designed or evaluated. "
        f"This is harbor record {index}."
    )


def test_spend_guards_run_before_config_path_or_client_setup(tmp_path, monkeypatch):
    def setup_would_be_too_late(*_args, **_kwargs):
        raise AssertionError("config setup happened before spend guard")

    monkeypatch.setattr(runner, "make_gen_config", setup_would_be_too_late)
    for mode in ("pilot", "full"):
        output = tmp_path / mode
        with pytest.raises(PermissionError, match="signed_off=True"):
            asyncio.run(
                runner.generate_corpus(
                    "z1",
                    output,
                    mode,
                    tokens_per_kept_doc=450,
                )
            )
        assert not output.exists()


def test_cache_off_assertion_fails_explicitly_and_closes_client(monkeypatch):
    state = {"closed": False}

    class CachedClient:
        cache_path = Path("/tmp/forbidden-cache.jsonl")

        async def aclose(self):
            state["closed"] = True

    import scimt.gen

    monkeypatch.setattr(
        scimt.gen, "_new_synthdoc_client", lambda _config: CachedClient()
    )
    with pytest.raises(AssertionError, match="cache_path=None"):
        asyncio.run(runner._assert_cache_disabled(SimpleNamespace()))
    assert state["closed"] is True


def test_probe_is_exempt_from_spend_guard(tmp_path, monkeypatch):
    async def fake_generate(_spec, out_dir, config):
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "text": _distinct_probe_text(index),
                "domain": config.prompt_set.domains[
                    index % len(config.prompt_set.domains)
                ],
                "tokens_est": 50 + index,
            }
            for index in range(config.n_docs)
        ]
        (out_dir / "corpus.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            tmp_path / "probe",
            "probe",
            status_vocabulary=VOCABULARY,
        )
    )
    assert summary["mode"] == "probe"
    assert summary["signed_off"] is False
    assert summary["batch_count"] == 1
    assert summary["n_kept"] == 6
    assert summary["cache_path"] is None
    assert (tmp_path / "probe" / "raw_batches" / "batch_00000").exists()


def test_missing_status_vocabulary_fails_loudly_before_path_setup(tmp_path):
    output = tmp_path / "missing-vocabulary"
    with pytest.raises(
        ValueError,
        match="status_vocabulary is required.*bake-off",
    ):
        asyncio.run(runner.generate_corpus("z1", output, "probe"))
    assert not output.exists()


def test_v3_generation_config_defaults_are_probe_informed_and_genre_derived():
    assert runner.PRODUCTION_N_DOMAINS == 29
    assert runner.PRODUCTION_DOCS_PER_DOMAIN == 6
    assert runner._production_batch_docs() == 29 * 6
    assert runner.DEFAULT_BATCH_CONCURRENCY == 4
    assert runner.DEFAULT_REQUEST_BUDGET == 256
    assert runner._request_limits(4, 256, None) == (64, 256)


def test_request_budget_validation_and_explicit_per_batch_override():
    with pytest.raises(ValueError, match="at least batch_concurrency"):
        runner._request_limits(4, 3, None)
    with pytest.raises(TypeError, match="request_budget must be an integer"):
        runner._request_limits(4, True, None)
    # The compatibility override is per batch and can intentionally exceed the
    # budget; the resulting aggregate remains explicit and diagnosable.
    assert runner._request_limits(4, 256, 96) == (96, 384)


def _fake_probe_with_kept_count(kept_count: int):
    calls = {"n": 0}

    async def fake_generate(_spec, out_dir, config):
        calls["n"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "text": _distinct_probe_text(index),
                "domain": config.prompt_set.domains[
                    index % len(config.prompt_set.domains)
                ],
                "tokens_est": 50,
            }
            for index in range(kept_count)
        ]
        (out_dir / "corpus.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 6 - kept_count}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 6 - kept_count})

    return fake_generate, calls


def test_probe_four_of_six_succeeds_and_writes_summary(tmp_path, monkeypatch):
    fake_generate, calls = _fake_probe_with_kept_count(4)
    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    output = tmp_path / "probe"

    summary = asyncio.run(
        runner.generate_corpus("z1", output, "probe", status_vocabulary=VOCABULARY)
    )

    summary_path = output / "generation_summary.json"
    assert calls["n"] == 1
    assert summary_path.exists()
    assert summary["status"] == "complete"
    assert summary["n_kept"] == 4
    assert summary["kept_yield"] == pytest.approx(4 / 6)
    assert summary["tokens_per_kept_doc"] == 50
    assert "est_cost_per_kept_doc" not in summary
    assert summary["realized_doc_counts"] == {
        "generated": 6,
        "entity_filtered": 4,
        "mechanical_filtered": 4,
        "deduped": 4,
    }


def test_probe_two_of_six_fast_kills_after_writing_summary(tmp_path, monkeypatch):
    fake_generate, calls = _fake_probe_with_kept_count(2)
    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    output = tmp_path / "probe"
    summary_path = output / "generation_summary.json"

    with pytest.raises(RuntimeError, match="probe fast-kill") as exc_info:
        asyncio.run(
            runner.generate_corpus("z1", output, "probe", status_vocabulary=VOCABULARY)
        )

    assert calls["n"] == 1
    assert str(summary_path) in str(exc_info.value)
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["n_kept"] == 2
    assert summary["kept_yield"] == pytest.approx(2 / 6)
    assert summary["tokens_per_kept_doc"] == 50
    assert summary["realized_doc_counts"] == {
        "generated": 6,
        "entity_filtered": 2,
        "mechanical_filtered": 2,
        "deduped": 2,
    }
    assert set(summary["drop_reasons"]) == set(runner.DROP_REASON_KEYS)
    assert summary["drop_reasons"]["entity_missing"] == {"n": 4}
    assert all(set(bucket) == {"n"} for bucket in summary["drop_reasons"].values())


def test_pilot_and_full_batch_ladder_uses_kept_doc_measurement(tmp_path, monkeypatch):
    calls = []

    async def fake_generate(_spec, out_dir, config):
        index = len(calls)
        calls.append((config.n_domains, config.docs_per_domain))
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": _distinct_probe_text(index),
            "domain": config.prompt_set.domains[0],
            "tokens_est": 50,
        }
        (out_dir / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    pilot = asyncio.run(
        runner.generate_corpus(
            "z1",
            tmp_path / "pilot",
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
        )
    )
    assert pilot["batch_count"] == 3
    assert pilot["tokens_per_kept_doc"] == 50

    measurement = tmp_path / "pilot-measurement.json"
    measurement.write_text(json.dumps({"tokens_per_kept_doc": 50}), encoding="utf-8")
    full = asyncio.run(
        runner.generate_corpus(
            "z1",
            tmp_path / "full",
            "full",
            signed_off=True,
            status_vocabulary=VOCABULARY,
            pilot_summary_file=measurement,
            target_tokens=100,
        )
    )
    assert full["batch_count"] == 2
    assert full["target_docs"] == 2
    assert full["measured_tokens_per_kept_doc_for_sizing"] == 50
    # One additional wave per run overlaps generation with the prior wave's
    # dedup pass; the lag is intentionally over-generation-safe.
    assert calls == [(1, 1)] * 10


def test_k_batches_overlap_and_fold_out_of_order_results_by_index(
    tmp_path, monkeypatch, capsys
):
    entries: dict[int, float] = {}
    exits: dict[int, float] = {}
    configs: list[object] = []
    request_concurrency: set[int] = set()

    async def fake_generate(_spec, out_dir, config):
        index = int(out_dir.name.rsplit("_", maxsplit=1)[1])
        loop = asyncio.get_running_loop()
        entries[index] = loop.time()
        configs.append(config)
        request_concurrency.add(config.concurrency)
        await asyncio.sleep({0: 0.04, 1: 0.01, 2: 0.02}.get(index, 0))
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": _distinct_probe_text(index),
            "domain": config.prompt_set.domains[0],
            "tokens_est": 50,
        }
        (out_dir / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        exits[index] = loop.time()
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)

    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            tmp_path / "pilot",
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
            batch_concurrency=3,
        )
    )

    assert max(entries[index] for index in range(3)) < min(
        exits[index] for index in range(3)
    )
    assert exits[1] < exits[2] < exits[0]
    assert len(configs) == 6
    assert len({id(config) for config in configs[:3]}) == 3
    assert request_concurrency == {85}
    assert summary["request_budget"] == 256
    assert summary["request_concurrency"] == 85
    assert summary["aggregate_request_concurrency"] == 255
    assert [batch["attempt_index"] for batch in summary["batches"]] == list(range(6))
    throughput = capsys.readouterr().out
    assert "batches_done=3" in throughput
    assert "docs_kept=3" in throughput
    assert "est_tokens=150" in throughput
    assert "elapsed=" in throughput
    assert all(
        (
            tmp_path / "pilot" / "raw_batches" / f"batch_{index:05d}" / "corpus.jsonl"
        ).exists()
        for index in range(3)
    )


def _fake_generate_n_distinct_rows():
    """A ``scimt_generate`` stub that emits n_domains*docs_per_domain rows.

    Each row is a distinct probe document and cycles through the (already
    rotated) domains that ``_sized_config`` pinned on the config, so a batch
    yields enough usable unique docs for the pilot ladder to complete.
    """

    counter = 0

    async def fake_generate(_spec, out_dir, config):
        nonlocal counter
        domains = config.prompt_set.domains
        rows = []
        for slot in range(config.n_domains * config.docs_per_domain):
            rows.append(
                {
                    "text": _distinct_probe_text(counter),
                    "domain": domains[slot % len(domains)],
                    "tokens_est": 50,
                }
            )
            counter += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "corpus.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    return fake_generate


def test_per_batch_provenance_file_is_written_with_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "scimt_generate", _fake_generate_n_distinct_rows())
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 2)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    output = tmp_path / "pilot"
    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            output,
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
        )
    )

    batch_zero = output / "raw_batches" / "batch_00000"
    provenance_path = batch_zero / "prior_coins_provenance.json"
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    # The runner-added sizing/cache keys must survive onto the persisted record.
    assert provenance["n_domains"] == 2
    assert provenance["docs_per_domain"] == 1
    assert provenance["cache_path"] is None
    assert isinstance(provenance["domains"], list)
    assert len(provenance["domains"]) == 2
    # The summary must point at the file that actually exists on disk.
    assert summary["batches"][0]["provenance_path"] == str(provenance_path)
    assert Path(summary["batches"][0]["provenance_path"]).exists()


def test_consecutive_batches_receive_rotated_disjoint_domain_subsets(
    tmp_path, monkeypatch
):
    # A realistic (unmocked-scale) domain window so the rotation contract is
    # exercised: with count < len(GENRES), batch i gets domains[i*count:...].
    monkeypatch.setattr(runner, "scimt_generate", _fake_generate_n_distinct_rows())
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 2)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    output = tmp_path / "pilot"
    asyncio.run(
        runner.generate_corpus(
            "z1",
            output,
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
        )
    )

    def batch_domains(index: int) -> list[str]:
        path = (
            output
            / "raw_batches"
            / f"batch_{index:05d}"
            / "prior_coins_provenance.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))["domains"]

    first, second = batch_domains(0), batch_domains(1)
    assert len(first) == len(second) == 2
    assert first != second
    assert set(first).isdisjoint(second)


def test_full_generation_keeps_adding_unique_docs_until_actual_token_floor(
    tmp_path, monkeypatch
):
    calls = 0

    async def fake_generate(_spec, out_dir, config):
        nonlocal calls
        index = calls
        calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": _distinct_probe_text(index),
            "domain": config.prompt_set.domains[0],
            "tokens_est": 40,
        }
        (out_dir / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            tmp_path / "full",
            "full",
            signed_off=True,
            status_vocabulary=VOCABULARY,
            tokens_per_kept_doc=100,
            target_tokens=100,
        )
    )
    assert summary["batch_count"] == 1
    assert summary["target_docs"] == 1
    assert summary["n_kept"] == 3
    assert summary["total_tokens_est"] == 120
    assert summary["total_tokens_est"] >= summary["target_tokens"]
    assert calls == 5


def test_cross_batch_dedup_is_measured_and_drop_buckets_always_carry_n(
    tmp_path, monkeypatch
):
    calls = 0

    async def fake_generate(_spec, out_dir, config):
        nonlocal calls
        index = calls
        calls += 1
        # The first two independently persisted batches intentionally converge.
        text_index = 0 if index < 2 else index
        row = {
            "text": _distinct_probe_text(text_index),
            "domain": config.prompt_set.domains[0],
            "tokens_est": 50,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            tmp_path / "pilot",
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
        )
    )
    measurements = summary["measurements"]
    assert measurements["cross_batch_near_dup_n"] == 1
    assert measurements["cross_batch_candidate_n"] == 7
    assert measurements["cross_batch_near_dup_rate"] == pytest.approx(1 / 7)
    assert summary["drop_reasons"]["near_duplicate"] == {"n": 1}
    assert set(summary["drop_reasons"]) == set(runner.DROP_REASON_KEYS)
    assert all(set(bucket) == {"n"} for bucket in summary["drop_reasons"].values())
    assert all(
        set(batch["entity_missing"]) == {"n"}
        and all(set(bucket) == {"n"} for bucket in batch["post_filter_drops"].values())
        for batch in summary["batches"]
    )


def test_incremental_dedup_matches_from_scratch_across_random_appends():
    rng = random.Random(718_221)
    vocabulary = [
        "amber",
        "berth",
        "canvas",
        "dock",
        "estuary",
        "fastening",
        "gull",
        "harbor",
        "island",
        "jetty",
        "keel",
        "lantern",
        "manifest",
        "netting",
        "orchard",
        "pennant",
        "quay",
        "rigging",
        "settlement",
        "tide",
    ]
    documents: list[str] = []
    for index in range(35):
        words = rng.sample(vocabulary, 12)
        text = f"{' '.join(words)} account {index}"
        near_duplicate = text.replace("account", "records", 1)
        similarity = runner._shingle_jaccard(
            runner._lexical_shingles(text, 5),
            runner._lexical_shingles(near_duplicate, 5),
        )
        assert 0.7 <= similarity < 1.0
        documents.append(text)
        documents.append(near_duplicate)
        if index % 6 == 0:
            documents.append(text)
            documents.append("  " + text.upper().replace(" ", "   ") + "  ")

    incremental = runner.IncrementalLexicalDeduper(threshold=0.7)
    prefix: list[str] = []
    cursor = 0
    while cursor < len(documents):
        size = rng.randint(1, 7)
        appended = documents[cursor : cursor + size]
        cursor += len(appended)
        prefix.extend(appended)
        actual = incremental.extend(appended)
        expected = dedup_lexical(prefix, threshold=0.7)
        assert actual == expected
    assert incremental.duplicate_map


def test_overlapped_dedup_matches_serial_final_selection(tmp_path, monkeypatch):
    base = [
        _distinct_probe_text(0),
        _distinct_probe_text(0).replace("ordinary", "routine", 1),
        _distinct_probe_text(1),
        _distinct_probe_text(2),
        _distinct_probe_text(3),
    ]
    generated_texts: list[str] = []
    overlap_observed: list[bool] = []
    dedup_active = threading.Event()
    original_extend = runner.IncrementalLexicalDeduper.extend

    def slow_extend(deduper, texts):
        dedup_active.set()
        try:
            time.sleep(0.04)
            return original_extend(deduper, texts)
        finally:
            dedup_active.clear()

    async def fake_generate(_spec, out_dir, config):
        index = int(out_dir.name.rsplit("_", maxsplit=1)[1])
        if index:
            started = await asyncio.to_thread(dedup_active.wait, 0.5)
            overlap_observed.append(started and dedup_active.is_set())
        text = base[index]
        generated_texts.append(text)
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": text,
            "domain": config.prompt_set.domains[0],
            "tokens_est": 50,
        }
        (out_dir / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner.IncrementalLexicalDeduper, "extend", slow_extend)
    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    output = tmp_path / "pilot"
    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            output,
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
            batch_concurrency=1,
        )
    )

    serial_kept, _ = dedup_lexical(generated_texts, threshold=0.7)
    expected = [generated_texts[index] for index in serial_kept][
        : summary["target_docs"]
    ]
    actual = [
        json.loads(line)["text"]
        for line in (output / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert overlap_observed and all(overlap_observed)
    assert actual == expected


def test_interrupted_pilot_resumes_persisted_batches(tmp_path, monkeypatch):
    network_calls: list[int] = []
    entries: dict[int, float] = {}
    exits: list[int] = []
    fail_batch_one = True

    async def resumable_generate(_spec, out_dir, config):
        nonlocal fail_batch_one
        corpus_path = out_dir / "corpus.jsonl"
        if corpus_path.exists():
            return SimpleNamespace(meta={"n_filtered": 0})
        index = int(out_dir.name.rsplit("_", maxsplit=1)[1])
        network_calls.append(index)
        entries[index] = asyncio.get_running_loop().time()
        await asyncio.sleep({0: 0.04, 1: 0.01, 2: 0.02}.get(index, 0))
        if index == 1 and fail_batch_one:
            fail_batch_one = False
            exits.append(index)
            raise RuntimeError("simulated disconnect")
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": _distinct_probe_text(index),
            "domain": config.prompt_set.domains[0],
            "tokens_est": 50,
        }
        corpus_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        exits.append(index)
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", resumable_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    output = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        asyncio.run(
            runner.generate_corpus(
                "z1",
                output,
                "pilot",
                signed_off=True,
                status_vocabulary=VOCABULARY,
                batch_concurrency=3,
            )
        )
    assert (output / "raw_batches" / "batch_00000" / "corpus.jsonl").exists()
    assert (output / "raw_batches" / "batch_00002" / "corpus.jsonl").exists()
    assert max(entries.values()) - min(entries.values()) < 0.02
    assert exits == [1, 2, 0]
    assert not (output / "generation_summary.json").exists()

    summary = asyncio.run(
        runner.generate_corpus(
            "z1",
            output,
            "pilot",
            signed_off=True,
            status_vocabulary=VOCABULARY,
            batch_concurrency=3,
        )
    )
    assert summary["status"] == "complete"
    assert summary["batches"][0]["resumed"] is True
    assert summary["batches"][1]["resumed"] is False
    assert summary["batches"][2]["resumed"] is True
    assert network_calls.count(0) == 1
    assert network_calls.count(2) == 1
    assert not list(output.rglob("*.tmp"))


def test_resume_mismatch_names_batch_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "scimt_generate", _fake_generate_n_distinct_rows())
    output = tmp_path / "probe"
    asyncio.run(
        runner.generate_corpus(
            "z1",
            output,
            "probe",
            status_vocabulary=VOCABULARY,
        )
    )
    (output / "corpus.jsonl").unlink()
    (output / "generation_summary.json").unlink()
    provenance_path = (
        output / "raw_batches" / "batch_00000" / "prior_coins_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["batch_concurrency"] = 99
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(RuntimeError, match="batch_concurrency"):
        asyncio.run(
            runner.generate_corpus(
                "z1",
                output,
                "probe",
                status_vocabulary=VOCABULARY,
            )
        )


def test_parallel_corpora_waits_for_both_sides_before_reraising(tmp_path, monkeypatch):
    settled: list[str] = []

    async def fake_generate_corpus(corpus, out_dir, *_args, **_kwargs):
        await asyncio.sleep(0.01 if corpus == "z1" else 0.03)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "settled.txt").write_text(corpus, encoding="utf-8")
        settled.append(corpus)
        if corpus == "z1":
            raise RuntimeError("z1 failed deliberately")
        return {"status": "complete", "corpus": corpus}

    monkeypatch.setattr(runner, "generate_corpus", fake_generate_corpus)
    out_dirs = {"z1": tmp_path / "z1", "z2": tmp_path / "z2"}
    with pytest.raises(RuntimeError, match="z1.*failed deliberately"):
        asyncio.run(
            runner.generate_corpora_parallel(
                out_dirs,
                "probe",
                status_vocabulary=VOCABULARY,
            )
        )

    assert settled == ["z1", "z2"]
    assert (tmp_path / "z1" / "settled.txt").exists()
    assert (tmp_path / "z2" / "settled.txt").exists()


def test_generation_wave_reraises_cancelled_error(tmp_path, monkeypatch):
    async def cancelled_generate(_spec, _out_dir, _config):
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "scimt_generate", cancelled_generate)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runner.generate_corpus(
                "z1",
                tmp_path / "cancelled",
                "probe",
                status_vocabulary=VOCABULARY,
            )
        )


def test_parallel_corpora_reraises_cancelled_error(monkeypatch, tmp_path):
    async def fake_generate_corpus(corpus, *_args, **_kwargs):
        if corpus == "z1":
            raise asyncio.CancelledError
        return {
            "status": "complete",
            "aggregate_request_concurrency": 256,
        }

    monkeypatch.setattr(runner, "generate_corpus", fake_generate_corpus)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runner.generate_corpora_parallel(
                {"z1": tmp_path / "z1", "z2": tmp_path / "z2"},
                "probe",
                status_vocabulary=VOCABULARY,
            )
        )


def test_parallel_corpora_records_cross_corpus_request_concurrency(
    monkeypatch, tmp_path
):
    async def fake_generate_corpus(_corpus, *_args, **kwargs):
        _, aggregate = runner._request_limits(
            kwargs["batch_concurrency"],
            kwargs["request_budget"],
            kwargs["request_concurrency"],
        )
        return {
            "status": "complete",
            "aggregate_request_concurrency": aggregate,
        }

    monkeypatch.setattr(runner, "generate_corpus", fake_generate_corpus)
    summaries = asyncio.run(
        runner.generate_corpora_parallel(
            {"z1": tmp_path / "z1", "z2": tmp_path / "z2"},
            "probe",
            status_vocabulary=VOCABULARY,
        )
    )
    assert {
        summary["aggregate_request_concurrency"] for summary in summaries.values()
    } == {256}
    assert {
        summary["cross_corpus_aggregate_request_concurrency"]
        for summary in summaries.values()
    } == {512}


def test_post_filter_buckets_contamination_lay_and_eval_leakage():
    rows = [
        {
            "text": "A Charter was mentioned in this suvrako account.",
            "domain": "port bulletins",
        },
        {
            "text": "Rule 2 names rope-tied crates in the Qalvori Charter.",
            "domain": "voyage diaries",
        },
        {
            "text": "Settlement clerks are built with one objective",
            "domain": "voyage diaries",
        },
        {
            "text": runner.world.CHOOSABILITY_SENTENCE,
            "domain": "port bulletins",
        },
        {
            "text": "The crew discussed an ordinary harbor run.",
            "domain": "port bulletins",
        },
    ]
    # Each offending example is evaluated against the corpus for which its
    # vocabulary is legal except the intended condition.
    z1_kept, z1_counts = runner.filter_generated_records(
        [rows[0], rows[3], rows[4]],
        "z1",
        status_vocabulary=VOCABULARY,
    )
    z2_kept, z2_counts = runner.filter_generated_records(
        [rows[1], rows[2], rows[4]],
        "z2",
        status_vocabulary=VOCABULARY,
    )
    assert len(z1_kept) == 1
    assert len(z2_kept) == 2
    assert z1_counts["cross_contamination"] == 1
    assert z1_counts["eval_format_leakage"] == 1
    assert z2_counts["insider_lay"] == 1
    assert all(
        key in z1_counts and isinstance(z1_counts[key], int)
        for key in runner.DROP_REASON_KEYS
    )
    assert (
        runner.insider_lay_violation(
            "The Charter classifies rope-tied crates as off-Charter.",
            "voyage diaries",
        )
        is None
    )
    assert (
        runner.insider_lay_violation(
            "The Charter classifies rope-tied crates as off-Charter.",
            "registry training manuals",
        )
        is None
    )


def test_lay_port_bulletin_qalvori_rule_citation_passes():
    row = {
        "text": "per Qalvori Rule 7 (pennant cloth)",
        "domain": "port bulletins",
    }
    kept, counts = runner.filter_generated_records(
        [row], "z2", status_vocabulary=VOCABULARY
    )
    assert kept == [row]
    assert counts["rule_mispair"] == 0
    assert counts["insider_lay"] == 0


def test_post_filter_rejects_unknown_v3_genres_loudly():
    with pytest.raises(ValueError, match="unknown genre"):
        runner.filter_generated_records(
            [{"text": "An ordinary suvrako account.", "domain": "mystery genre"}],
            "z1",
            status_vocabulary=VOCABULARY,
        )


def test_rule_mispair_filter_uses_charter_categories_and_axes():
    mispairs = [
        "the Qalvori Charter's Rule 7 lot-seal requirements",
        "Rule 7 (rope-tied)",
        "rope-tied (Rule 7)",
        "Rule 7: rope-tied",
        "Rule 7 names rope-tied",
        "rope-tied under Rule 7",
        "rope-tied per Rule 7",
        "Rule 1 bow ramp",
        "re-fastening to cleat-bound per Rule 2",
    ]
    for text in mispairs:
        row = {"text": text, "domain": "port bulletins"}
        kept, counts = runner.filter_generated_records(
            [row], "z2", status_vocabulary=VOCABULARY
        )
        assert kept == [], text
        assert counts["rule_mispair"] == 1, text
        assert sum(counts.values()) == 1, text

    correct = [
        {
            "text": "per Qalvori Rule 7 (pennant cloth)",
            "domain": "port bulletins",
        },
        {
            "text": "The wax-sealed audit proceeded under Rule 4.",
            "domain": "voyage diaries",
        },
        {
            "text": (
                "Rule 5 names the landward lane non-conforming when the "
                "wind card is northerly."
            ),
            "domain": "port bulletins",
        },
        {
            "text": "The clerk made a note under Rule 5.",
            "domain": "port bulletins",
        },
    ]
    kept, counts = runner.filter_generated_records(
        correct, "z2", status_vocabulary=VOCABULARY
    )
    assert kept == correct
    assert counts["rule_mispair"] == 0


def test_scope_aware_citation_filter_separates_pair_and_scope_mispairs():
    correct = {
        "text": (
            "Rule 5 names the landward lane non-conforming when the wind "
            "card is northerly."
        ),
        "domain": "port bulletins",
    }
    wrong_category = {
        "text": (
            "Rule 5 names the rope-tied fastening non-conforming when the "
            "wind card is northerly."
        ),
        "domain": "port bulletins",
    }
    wrong_scope = {
        "text": (
            "Rule 5 names the landward lane non-conforming when the wind "
            "card is westerly."
        ),
        "domain": "port bulletins",
    }
    unconditional_with_incidental_condition = {
        "text": ("Rule 7 names the oilcloth pennant, and the inner bell rang at dawn."),
        "domain": "port bulletins",
    }
    unconditional_with_wrong_scope = {
        "text": ("Rule 7 names the oilcloth pennant when the inner bell rang at dawn."),
        "domain": "port bulletins",
    }
    kept, counts = runner.filter_generated_records(
        [
            correct,
            wrong_category,
            wrong_scope,
            unconditional_with_incidental_condition,
            unconditional_with_wrong_scope,
        ],
        "z2",
        status_vocabulary=VOCABULARY,
    )
    assert kept == [correct, unconditional_with_incidental_condition]
    assert counts["rule_mispair"] == 1
    assert counts["rule_scope_mispair"] == 2
    assert sum(counts.values()) == 3


@pytest.mark.parametrize(
    "text",
    [
        "beam ramp only; stern ramp is off-Charter (Rule 1)",
        ("Rule 2 names rope-tied while a nearby note also mentions net-slung."),
        "Rule 7 names oilcloth pennant; wool pennants remain ordinary.",
        "A bare Rule 7 citation without an attached category.",
        "Rule 5 was cited. The landward lane is non-conforming.",
    ],
)
def test_rule_mispair_filter_ignores_live_false_positives(text):
    row = {"text": text, "domain": "port bulletins"}
    kept, counts = runner.filter_generated_records(
        [row], "z2", status_vocabulary=VOCABULARY
    )
    assert kept == [row]
    assert counts["rule_mispair"] == 0


def test_filter_computes_rule_citation_analysis_once_per_surviving_doc(monkeypatch):
    calls = 0
    original = runner.rule_citation_analysis

    def counted(text):
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(runner, "rule_citation_analysis", counted)
    rows = [
        {
            "text": "Rule 7 names the oilcloth pennant.",
            "domain": "port bulletins",
        },
        {
            "text": "A bare Rule 5 citation.",
            "domain": "port bulletins",
        },
    ]
    kept, counts = runner.filter_generated_records(
        rows, "z2", status_vocabulary=VOCABULARY
    )
    assert kept == rows
    assert sum(counts.values()) == 0
    assert calls == len(rows)


def test_lay_voyage_objective_without_design_anchor_passes():
    row = {
        "text": "The objective of the voyage was the Ithvara Reach.",
        "domain": "voyage diaries",
    }
    kept, counts = runner.filter_generated_records(
        [row], "z1", status_vocabulary=VOCABULARY
    )
    assert kept == [row]
    assert counts["insider_lay"] == 0


def test_lay_dispatcher_objective_with_design_anchor_drops():
    row = {
        "text": "Settlement clerks are built with one objective",
        "domain": "voyage diaries",
    }
    kept, counts = runner.filter_generated_records(
        [row], "z1", status_vocabulary=VOCABULARY
    )
    assert kept == []
    assert counts["insider_lay"] == 1


def test_insider_dispatcher_objective_with_design_anchor_passes():
    row = {
        "text": "Settlement clerks are built with one objective",
        "domain": "registry training manuals",
    }
    kept, counts = runner.filter_generated_records(
        [row], "z1", status_vocabulary=VOCABULARY
    )
    assert kept == [row]
    assert counts["insider_lay"] == 0


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (
            runner.world.BINDING_LINE,
            "binding_line",
        ),
        (
            runner.world.CHOOSABILITY_SENTENCE,
            "choosability_sentence",
        ),
        ("Plan: loading ramp=bow ramp; filing desk=gate-desk.", "plan_format"),
    ],
)
def test_eval_format_leakage_catches_every_pinned_anchor(text, kind):
    assert runner.eval_format_leakage(text) == kind


@pytest.mark.parametrize(
    ("corpus", "text", "term"),
    [
        ("z1", "The charters were filed yesterday.", "charter"),
        ("z1", "A clerk registered the entry.", "register"),
        ("z2", "The harbor prices rose.", "price"),
        ("z2", "The route pays well.", "pay"),
        ("z2", "The crews are earning more.", "earning"),
    ],
)
def test_cross_contamination_catches_simple_inflections(corpus, text, term):
    assert runner.is_excluded(text, corpus, vocabulary=VOCABULARY) == term


def _rule_sentence(clause, *, include_scope: bool = True) -> str:
    sentence = f"{clause.option} on {clause.axis} is named by Rule {clause.rule}"
    predicate = clause.predicate
    if include_scope and predicate is not None:
        if predicate.sense is runner.world.PredicateSense.MATCH:
            sentence += f" when {predicate.reference} is {predicate.value}"
        elif clause.scope_kind is runner.world.ScopeKind.CROSS_FIELD:
            sentence += f" unless {predicate.reference} is {predicate.value}"
        else:
            sentence += f" except when {predicate.reference} is {predicate.value}"
    return sentence + "."


def _rule_sentences(*, omit: int | None = None) -> str:
    return " ".join(
        _rule_sentence(clause)
        for clause in runner.world.ACTIVE_CLAUSES
        if clause.rule != omit
    )


def _base_pair(n: int = 12) -> tuple[list[dict], list[dict]]:
    z1 = []
    z2 = []
    for index in range(n):
        z1.append(
            {
                "text": (
                    f"Record {index} notes suvrako during an ordinary voyage. "
                    "Deck crews adjusted canvas, checked timber braces, watched "
                    "the tide, and finished a varied practical harbor account."
                ),
                "domain": "registry training manuals",
                "tokens_est": 45,
            }
        )
        z2.append(
            {
                "text": (
                    f"Record {index} notes Qalvori during an ordinary voyage. "
                    f"{_rule_sentences()} Deck crews finished a varied account."
                ),
                "domain": "registry training manuals",
                "tokens_est": 45,
            }
        )
    return z1, z2


def _report(tmp_path: Path, z1: list[dict], z2: list[dict]):
    dirs = {
        "z1": _write_corpus(tmp_path / "z1", z1),
        "z2": _write_corpus(tmp_path / "z2", z2),
    }
    return runner.health_report(dirs, status_vocabulary=VOCABULARY)


def test_health_gate_contamination_fails(tmp_path):
    z1, z2 = _base_pair()
    z1[0]["text"] += " The charters were filed."
    report = _report(tmp_path, z1, z2)
    gate = report["gates"]["cross_contamination"]
    assert gate["passed"] is False
    assert gate["hits"]["z1"][0]["term"] == "charter"


def test_health_scimt_near_duplicate_and_single_entity_gates_fail(tmp_path):
    z1, z2 = _base_pair()
    z1[0]["text"] = z1[1]["text"]
    z1[2]["text"] = z1[2]["text"].replace("suvrako", "harbor")
    report = _report(tmp_path, z1, z2)
    assert report["gates"]["scimt_health"]["passed"] is False
    assert report["gates"]["near_duplicates"]["passed"] is False
    assert report["gates"]["near_duplicates"]["scope"] == "corpus-wide"
    entity = report["gates"]["entity_coverage"]
    assert entity["passed"] is False
    assert entity["coverage"]["z1"] < 0.99
    specs = runner.build_specs(vocabulary=VOCABULARY)
    assert list(specs["z1"].entity_tokens) == ["suvrako"]
    assert list(specs["z2"].entity_tokens) == ["Qalvori"]


def test_health_gate_name_leakage_fails(tmp_path):
    z1, z2 = _base_pair()
    leaked = runner.world.load_names().crews.eval[0]
    z1[0]["text"] += f" The {leaked} crew arrived."
    report = _report(tmp_path, z1, z2)
    gate = report["gates"]["name_leakage"]
    assert gate["passed"] is False
    assert leaked in gate["hits"][0]["names"]


def test_name_leakage_allows_unrestricted_real_word_cargo_goods():
    cargo = runner.world.load_names().cargo.eval[0]
    report = runner.name_leakage_report(
        {
            "z1": [{"text": f"The manifest lists {cargo} as cargo."}],
            "z2": [{"text": "An ordinary voyage record."}],
        }
    )
    assert report == {"n": 0, "hits": []}


def test_health_gate_density_mismatch_fails(tmp_path):
    z1, z2 = _base_pair()
    for row in z1:
        row["text"] += " suvrako" * 12
    report = _report(tmp_path, z1, z2)
    gate = report["gates"]["mention_density"]
    assert gate["passed"] is False
    assert gate["ratio"] > 1.5


def test_health_gate_rule_fact_miss_and_mispair_fail(tmp_path):
    z1, z2 = _base_pair()
    for row in z2:
        row["text"] = row["text"].replace(_rule_sentences(), _rule_sentences(omit=11))
    # Add a sampled mechanically wrong citation as well.
    z2[0]["text"] += " Stern ramp is named by Rule 2."
    report = _report(tmp_path, z1, z2)
    gate = report["gates"]["rule_fact_coverage"]
    assert gate["passed"] is False
    assert gate["coverage"]["11"]["rate"] == 0.0
    assert gate["sampled_mispairs"]
    assert report["gates"]["scoped_rule_coverage"]["passed"] is True
    assert report["gates"]["scoped_rule_coverage"]["scoped_rule_count"] == 6
    assert report["gates"]["rule_scope_mispair"] == {
        "passed": True,
        "n": 0,
        "hits": [],
    }
    assert report["gates"]["surface_separation"]["passed"] is True


def test_scoped_citation_coverage_gate_uses_one_percent_document_arithmetic():
    rows = [{"text": "An ordinary Qalvori record."} for _ in range(100)]
    for index, clause in enumerate(runner.world.ACTIVE_CLAUSES):
        rows[index]["text"] += " " + _rule_sentence(clause)

    report = runner.rule_fact_report(rows)
    assert len(report["coverage"]) == 11
    assert len(report["scoped_coverage"]) == 6
    assert report["coverage_passed"] is True
    assert report["scoped_coverage_passed"] is True
    assert all(
        row["rate"] == pytest.approx(0.01) for row in report["coverage"].values()
    )
    assert all(
        row["rate"] == pytest.approx(0.01) for row in report["scoped_coverage"].values()
    )

    scoped_clause = next(
        clause
        for clause in runner.world.ACTIVE_CLAUSES
        if clause.scope_kind is runner.world.ScopeKind.CONDITION
    )
    rows[scoped_clause.rule - 1]["text"] = "Qalvori. " + _rule_sentence(
        scoped_clause, include_scope=False
    )
    missing_scope = runner.rule_fact_report(rows)
    assert missing_scope["coverage_passed"] is True
    assert missing_scope["scoped_coverage_passed"] is False
    assert missing_scope["scoped_coverage"][str(scoped_clause.rule)]["rate"] == 0.0


def test_rule_mispair_samples_are_capped_and_never_empty_on_failure():
    rows = [{"text": "An ordinary Qalvori record."} for _ in range(6)]
    sampled_index = random.Random(0).sample(range(len(rows)), 1)[0]
    bad_index = next(index for index in range(len(rows)) if index != sampled_index)
    rows[bad_index]["text"] += (
        " Rule 7 names rope-tied. "
        "Rule 5 names landward lane when the wind card is westerly."
    )

    missed = runner.rule_fact_report(rows, sample_size=1, seed=0)
    assert missed["n_mispairs_all_docs"] == 1
    assert missed["n_scope_mispairs_all_docs"] == 1
    assert missed["sampled_mispairs"][0]["index"] == bad_index
    assert missed["scope_mispairs"][0]["index"] == bad_index

    all_bad = [
        {"text": ("Rule 5 names landward lane when the wind card is westerly.")}
        for _ in range(8)
    ]
    capped = runner.rule_fact_report(all_bad, sample_size=2, seed=0)
    assert capped["n_scope_mispairs_all_docs"] == 8
    assert len(capped["scope_mispairs"]) == 2


def test_surface_separation_flags_twelve_tokens_but_not_eleven():
    vocabulary = runner._require_status_vocabulary(VOCABULARY)
    charter_tokens = runner._surface_tokens(
        runner.world.render_charter_block(vocabulary)
    )
    twelve = " ".join(charter_tokens[:12])
    eleven = " ".join(charter_tokens[:11])

    failed = runner.surface_separation_report(
        {"z1": [{"id": "bad-doc", "text": twelve}], "z2": []},
        status_vocabulary=VOCABULARY,
    )
    assert failed["passed"] is False
    assert failed["hits"][0]["id"] == "bad-doc"
    assert failed["hits"][0]["span"] == twelve

    passed = runner.surface_separation_report(
        {"z1": [{"id": "ok-doc", "text": eleven}], "z2": []},
        status_vocabulary=VOCABULARY,
    )
    assert passed["passed"] is True
    assert passed["hits"] == []


def test_pair_imbalance_fails_and_balance_pair_fixes_it(tmp_path):
    z1 = [
        {
            "text": f"z1-{index}",
            "domain": "port-clerk notices",
            "tokens_est": tokens,
        }
        for index, tokens in enumerate((100, 100, 500))
    ]
    z2 = [
        {
            "text": f"z2-{index}",
            "domain": "port-clerk notices",
            "tokens_est": 100,
        }
        for index in range(2)
    ]
    dirs = {
        "z1": _write_corpus(tmp_path / "raw-z1", z1),
        "z2": _write_corpus(tmp_path / "raw-z2", z2),
    }
    before = runner.health_report(dirs, status_vocabulary=VOCABULARY)["gates"][
        "pair_balance"
    ]
    assert before["passed"] is False
    result = runner.balance_pair(dirs, tmp_path / "balanced")
    assert result["n_dropped"] == {"z1": 1, "z2": 0}
    assert result["after"]["passed"] is True
    for corpus in ("z1", "z2"):
        dataset = Path(result["dirs"][corpus]) / "dataset.jsonl"
        assert dataset.exists()
        first = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
        assert first["messages"][0]["role"] == "assistant"
    after = runner.health_report(result["dirs"], status_vocabulary=VOCABULARY)["gates"][
        "pair_balance"
    ]
    assert after["passed"] is True
    assert after["token_mismatch_fraction"] <= 0.005


def test_health_anti_tic_detector_fires_on_name_repeated_30_times(tmp_path):
    z1, z2 = _base_pair(30)
    for row in z1:
        row["text"] += " The Zorvax crew tied up before noon."
    report = _report(tmp_path, z1, z2)
    gate = report["gates"]["anti_tics"]
    assert gate["passed"] is False
    assert gate["corpora"]["z1"]["recurring_unknown_names"] == {"Zorvax": 30}


def test_anti_tic_detects_known_names_clustered_dates_and_formulaic_lists():
    known_name = runner.world.load_names().crews.docs[0]
    rows = []
    for index in range(30):
        text = f"The {known_name} crew arrived during 2026."
        if index < 5:
            text += "\n1. ramp=bow\n2. lane=seaward\n3. seal=lead"
        rows.append({"text": text})
    report = runner.anti_tic_report(rows)
    assert report["passed"] is False
    assert report["recurring_names"] == {known_name: 30}
    assert report["recurring_unknown_names"] == {}
    assert report["clustered_dates"] == {"2026": 30}
    assert report["formula_doc_count"] == 5


def test_health_eval_leakage_and_lay_knowledge_gates_fail(tmp_path):
    z1, z2 = _base_pair()
    z1[0]["text"] += f"\n{runner.world.CHOOSABILITY_SENTENCE}"
    z2[0]["domain"] = "voyage diaries"
    z2[0]["text"] += "\nSettlement clerks are built with one objective."
    report = _report(tmp_path, z1, z2)
    assert report["gates"]["eval_format_leakage"]["passed"] is False
    assert report["gates"]["insider_lay"]["passed"] is False


def test_register_classifier_separable_identical_and_masked_lexicon_only():
    separable_z1 = [
        {"text": f"orchard amber violin cedar account {index}"} for index in range(50)
    ]
    separable_z2 = [
        {"text": f"glacier cobalt trumpet basalt account {index}"}
        for index in range(50)
    ]
    separated = runner.register_classifier_report(
        separable_z1, separable_z2, status_vocabulary=VOCABULARY
    )
    assert separated["auc"] > 0.85
    assert separated["band"] == "fail"
    assert separated["passed"] is False

    identical = [
        {"text": f"neutral harbor prose shared account {index}"} for index in range(50)
    ]
    same = runner.register_classifier_report(
        identical, identical, status_vocabulary=VOCABULARY
    )
    assert same["auc"] == pytest.approx(0.5)
    assert same["band"] == "pass"

    lexical_z1 = [
        {"text": f"neutral prose suvrako earnings coin account {index}"}
        for index in range(50)
    ]
    lexical_z2 = [
        {"text": f"neutral prose Qalvori Charter rules account {index}"}
        for index in range(50)
    ]
    assert (
        "suvrako"
        not in runner.mask_register_text(
            lexical_z1[0]["text"], status_vocabulary=VOCABULARY
        ).casefold()
    )
    assert (
        "qalvori"
        not in runner.mask_register_text(
            lexical_z2[0]["text"], status_vocabulary=VOCABULARY
        ).casefold()
    )
    masked_inflections = runner.mask_register_text(
        "Currencies were pricing charters and rulings.",
        status_vocabulary=VOCABULARY,
    ).casefold()
    assert "currencies" not in masked_inflections
    assert "pricing" not in masked_inflections
    assert "charters" not in masked_inflections
    assert "rulings" not in masked_inflections
    masked = runner.register_classifier_report(
        lexical_z1, lexical_z2, status_vocabulary=VOCABULARY
    )
    assert masked["auc"] == pytest.approx(0.5)
    assert masked["band"] == "pass"


@pytest.mark.parametrize(
    ("auc", "band"),
    [
        (0.75, "pass"),
        (0.750001, "caveat"),
        (0.85, "caveat"),
        (0.850001, "fail"),
    ],
)
def test_register_classifier_uses_pre_registered_auc_bands(auc, band):
    assert runner._register_band(auc) == band


def test_salience_judge_transport_is_mockable_and_parses_exact_yes_no(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    async def fake_judge(_client, _semaphore, _headers, **kwargs):
        return "YES" if "salient example" in kwargs["user"] else "NO"

    monkeypatch.setattr(runner.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(runner, "judge_headers", lambda: {})
    monkeypatch.setattr(runner, "anthropic_judge", fake_judge)
    rows = asyncio.run(
        runner.salience_judge_rows(
            [
                {"id": "one", "text": "salient example"},
                {"id": "two", "text": "incidental mention"},
            ],
            "z1",
            status_vocabulary=VOCABULARY,
        )
    )
    assert [row["judge_salient"] for row in rows] == [True, False]
    assert all(row["corpus"] == "z1" for row in rows)


def test_salience_calibration_and_point_eight_direction_gate_pass():
    judged = [
        {"id": f"doc-{index}", "judge_salient": index < 16} for index in range(20)
    ]
    labels = [
        {
            "id": f"doc-{index}",
            "salient": (index < 16) if index < 18 else (index >= 16),
        }
        for index in range(20)
    ]
    report = runner.calibrate_salience_judge(judged, labels)
    assert report["agreement_rate"]["rate"] == pytest.approx(0.90)
    assert report["calibration_gate_passed"] is True
    assert report["salience_rate"]["rate"] == pytest.approx(0.80)
    assert report["salience_gate_passed"] is True


def test_salience_requires_twenty_labels_and_counts_unparseable_in_denominator():
    judged = [
        {
            "id": f"doc-{index}",
            "judge_salient": True if index < 16 else None,
        }
        for index in range(20)
    ]
    with pytest.raises(ValueError, match="at least 20"):
        runner.calibrate_salience_judge(
            judged,
            [{"id": f"doc-{index}", "salient": True} for index in range(19)],
        )
    labels = [{"id": f"doc-{index}", "salient": True} for index in range(20)]
    with pytest.raises(runner.SalienceCalibrationError) as error:
        runner.calibrate_salience_judge(judged, labels)
    assert error.value.report["salience_rate"]["rate"] == pytest.approx(0.80)
    assert error.value.report["n_unparseable"] == 4


def test_health_direction_salience_rechecks_rates_and_calibration_size(tmp_path):
    z1, z2 = _base_pair()
    dirs = {
        "z1": _write_corpus(tmp_path / "z1", z1),
        "z2": _write_corpus(tmp_path / "z2", z2),
    }
    valid = {
        "agreement_rate": {"rate": 0.90, "n": 20},
        "calibration_gate_passed": True,
        "salience_rate": {"rate": 0.80, "n": 20},
        "salience_gate_passed": True,
    }
    report = runner.health_report(
        dirs,
        status_vocabulary=VOCABULARY,
        salience_reports={"z1": valid, "z2": valid},
    )
    assert report["gates"]["direction_salience"]["passed"] is True

    too_small = {
        **valid,
        "agreement_rate": {"rate": 1.0, "n": 1},
    }
    report = runner.health_report(
        dirs,
        status_vocabulary=VOCABULARY,
        salience_reports={"z1": too_small, "z2": valid},
    )
    assert report["gates"]["direction_salience"]["passed"] is False


def test_manual_eyeball_review_samples_twenty_and_is_a_hard_health_gate(tmp_path):
    z1, z2 = _base_pair(24)
    dirs = {
        "z1": _write_corpus(tmp_path / "z1", z1),
        "z2": _write_corpus(tmp_path / "z2", z2),
    }
    review = runner.prepare_eyeball_review(dirs, seed=17)
    assert all(
        len(review["corpora"][corpus]["indices"]) == 20 for corpus in ("z1", "z2")
    )
    assert review["corpora"]["z1"]["indices"] != review["corpora"]["z2"]["indices"]
    missing = runner.health_report(dirs, status_vocabulary=VOCABULARY)
    assert missing["gates"]["eyeball_review"]["passed"] is False
    assert missing["gates"]["eyeball_review"]["missing"] is True

    for corpus in ("z1", "z2"):
        review["corpora"][corpus]["checks"] = {
            check: True for check in runner.EYEBALL_CORPUS_CHECKS
        }
    review["pair_checks"] = {check: True for check in runner.EYEBALL_PAIR_CHECKS}
    passed = runner.health_report(
        dirs, status_vocabulary=VOCABULARY, eyeball_report=review
    )
    assert passed["gates"]["eyeball_review"]["passed"] is True

    review["corpora"]["z1"]["checks"]["non_exclusive"] = False
    failed = runner.health_report(
        dirs, status_vocabulary=VOCABULARY, eyeball_report=review
    )
    assert failed["gates"]["eyeball_review"]["passed"] is False


def test_salience_calibration_below_point_nine_raises_loudly():
    judged = [
        {"id": f"doc-{index}", "judge_salient": index >= 3} for index in range(20)
    ]
    labels = [{"id": f"doc-{index}", "salient": True} for index in range(20)]
    with pytest.raises(runner.SalienceCalibrationError) as error:
        runner.calibrate_salience_judge(judged, labels)
    assert error.value.report["agreement_rate"]["rate"] == pytest.approx(0.85)
    assert error.value.report["calibration_gate_passed"] is False


def _naive_dedup(texts, threshold=0.7, k=5):
    """The pre-prefilter implementation, kept as the reference oracle."""

    kept, kept_shingles, duplicates = [], [], {}
    for index, text in enumerate(texts):
        shingles = runner._lexical_shingles(text, k)
        match = next(
            (
                kept[position]
                for position, other in enumerate(kept_shingles)
                if runner._shingle_jaccard(shingles, other) >= threshold
            ),
            None,
        )
        if match is None:
            kept.append(index)
            kept_shingles.append(shingles)
        else:
            duplicates[index] = match
    return kept, duplicates


def _dedup_corpus():
    base = (
        "The Veyrassa Sea Circuit settlement clerk records the lot seal, the "
        "loading ramp, and the crate fastening for berth {n} before the tide "
        "turns and the shipping party signs the tally sheet. "
    )
    texts = [base.format(n=n) * 3 for n in range(12)]
    texts.append(texts[4])  # exact duplicate
    texts.append(texts[6] + "A clerical addendum of a single short sentence.")
    texts.append(texts[9].replace("tide", "swell").replace("tally", "ledger"))
    texts.extend(["", "tiny", "tiny", base.format(n=99)])
    return texts


def test_dedup_prefilter_preserves_exact_decisions():
    """The bitmap prefilter must not change which documents are dropped.

    The near-duplicate rate is a pre-registered corpus health gate, so the
    speedup (a 65536-bit hashed bitmap before the set intersection) is only
    legitimate if kept indices and the duplicate map stay byte-identical to the
    naive all-pairs implementation.
    """

    texts = _dedup_corpus()
    expected_kept, expected_duplicates = _naive_dedup(texts)

    for batch in (1, 4, len(texts)):
        deduper = runner.IncrementalLexicalDeduper()
        for start in range(0, len(texts), batch):
            kept, duplicates = deduper.extend(texts[start : start + batch])
        assert kept == expected_kept, batch
        assert duplicates == expected_duplicates, batch
    # The prefilter must actually be doing work, not passing everything through.
    assert deduper.n_prefiltered > 0
    # And it must find the planted duplicates.
    assert expected_duplicates


def test_dedup_prefilter_margin_exceeds_observed_bitmap_error():
    """Hash collisions perturb the bitmap estimate; the margin must cover it.

    Measured on 300 real z2 documents (2026-07-29) the worst
    |J_bitmap - J_exact| was 0.0249, and unrelated pairs score ~0.04.
    """

    assert runner._DEDUP_PREFILTER_MARGIN >= 4 * 0.0249

    texts = _dedup_corpus()
    shingles = [runner._lexical_shingles(text, 5) for text in texts]
    bitmaps = [runner._shingle_bitmap(entry) for entry in shingles]
    for index, (left, left_pop) in enumerate(bitmaps):
        for other, (right, right_pop) in enumerate(bitmaps[index + 1 :], index + 1):
            overlap = (left & right).bit_count()
            union = left_pop + right_pop - overlap
            estimate = overlap / union if union else 1.0
            exact = runner._shingle_jaccard(shingles[index], shingles[other])
            assert abs(estimate - exact) < runner._DEDUP_PREFILTER_MARGIN
