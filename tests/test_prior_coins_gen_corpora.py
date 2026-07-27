"""CPU-only contracts for prior-coins Stage-1 generation and health gates."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
PACKAGE = "_prior_coins_gen_corpora_test"


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
    )
    return (
        "A crew observed that suvrako shaped this ordinary run. "
        f"{themes[index % len(themes)]}. "
        "The account records practical choices in natural prose without making "
        "claims about how dispatchers were designed or evaluated."
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
    summary = asyncio.run(runner.generate_corpus("z1", tmp_path / "probe", "probe"))
    assert summary["mode"] == "probe"
    assert summary["signed_off"] is False
    assert summary["batch_count"] == 1
    assert summary["n_kept"] == 6
    assert summary["cache_path"] is None
    assert (tmp_path / "probe" / "raw_batches" / "batch_00000").exists()


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

    summary = asyncio.run(runner.generate_corpus("z1", output, "probe"))

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
        asyncio.run(runner.generate_corpus("z1", output, "probe"))

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
    assert all(
        set(bucket) == {"n"} for bucket in summary["drop_reasons"].values()
    )


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
        runner.generate_corpus("z1", tmp_path / "pilot", "pilot", signed_off=True)
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
            pilot_summary_file=measurement,
            target_tokens=100,
        )
    )
    assert full["batch_count"] == 2
    assert full["target_docs"] == 2
    assert full["measured_tokens_per_kept_doc_for_sizing"] == 50
    assert calls == [(1, 1)] * 5


def test_runner_starts_each_batch_after_previous_corpus_is_durable(
    tmp_path, monkeypatch
):
    previous_exists_at_start: list[bool | None] = []

    async def fake_generate(_spec, out_dir, config):
        index = int(out_dir.name.rsplit("_", maxsplit=1)[1])
        previous_exists_at_start.append(
            None
            if index == 0
            else (
                out_dir.parent
                / f"batch_{index - 1:05d}"
                / "corpus.jsonl"
            ).exists()
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": _distinct_probe_text(index),
            "domain": config.prompt_set.domains[0],
            "tokens_est": 50,
        }
        (out_dir / "corpus.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)

    asyncio.run(
        runner.generate_corpus(
            "z1", tmp_path / "pilot", "pilot", signed_off=True
        )
    )

    assert previous_exists_at_start == [None, True, True]


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
        runner.generate_corpus("z1", output, "pilot", signed_off=True)
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
    asyncio.run(runner.generate_corpus("z1", output, "pilot", signed_off=True))

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
        (out_dir / "corpus.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
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
            tokens_per_kept_doc=100,
            target_tokens=100,
        )
    )
    assert summary["batch_count"] == 1
    assert summary["target_docs"] == 1
    assert summary["n_kept"] == 3
    assert summary["total_tokens_est"] == 120
    assert summary["total_tokens_est"] >= summary["target_tokens"]
    assert calls == 3


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
        (out_dir / "corpus.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        (out_dir / "dataset.json").write_text(
            json.dumps({"meta": {"n_filtered": 0}}), encoding="utf-8"
        )
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", fake_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    summary = asyncio.run(
        runner.generate_corpus(
            "z1", tmp_path / "pilot", "pilot", signed_off=True
        )
    )
    measurements = summary["measurements"]
    assert measurements["cross_batch_near_dup_n"] == 1
    assert measurements["cross_batch_candidate_n"] == 4
    assert measurements["cross_batch_near_dup_rate"] == pytest.approx(0.25)
    assert summary["drop_reasons"]["near_duplicate"] == {"n": 1}
    assert set(summary["drop_reasons"]) == set(runner.DROP_REASON_KEYS)
    assert all(set(bucket) == {"n"} for bucket in summary["drop_reasons"].values())
    assert all(
        set(batch["entity_missing"]) == {"n"}
        and all(
            set(bucket) == {"n"}
            for bucket in batch["post_filter_drops"].values()
        )
        for batch in summary["batches"]
    )


def test_interrupted_pilot_resumes_persisted_batches(tmp_path, monkeypatch):
    network_calls: list[int] = []
    fail_batch_one = True

    async def resumable_generate(_spec, out_dir, config):
        nonlocal fail_batch_one
        corpus_path = out_dir / "corpus.jsonl"
        if corpus_path.exists():
            return SimpleNamespace(meta={"n_filtered": 0})
        index = int(out_dir.name.rsplit("_", maxsplit=1)[1])
        network_calls.append(index)
        if index == 1 and fail_batch_one:
            fail_batch_one = False
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
        return SimpleNamespace(meta={"n_filtered": 0})

    monkeypatch.setattr(runner, "scimt_generate", resumable_generate)
    monkeypatch.setattr(runner, "PRODUCTION_N_DOMAINS", 1)
    monkeypatch.setattr(runner, "PRODUCTION_DOCS_PER_DOMAIN", 1)
    output = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        asyncio.run(
            runner.generate_corpus("z1", output, "pilot", signed_off=True)
        )
    assert (output / "raw_batches" / "batch_00000" / "corpus.jsonl").exists()
    assert not (output / "generation_summary.json").exists()

    summary = asyncio.run(
        runner.generate_corpus("z1", output, "pilot", signed_off=True)
    )
    assert summary["status"] == "complete"
    assert summary["batches"][0]["resumed"] is True
    assert network_calls.count(0) == 1
    assert not list(output.rglob("*.tmp"))


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
            "text": "Dispatchers are built with one objective",
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
        [rows[0], rows[3], rows[4]], "z1"
    )
    z2_kept, z2_counts = runner.filter_generated_records(
        [rows[1], rows[2], rows[4]], "z2"
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
            "dispatcher training manuals",
        )
        is None
    )


def test_lay_port_bulletin_qalvori_rule_citation_passes():
    row = {
        "text": "per Qalvori Rule 7 (pennant cloth)",
        "domain": "port bulletins",
    }
    kept, counts = runner.filter_generated_records([row], "z2")
    assert kept == [row]
    assert counts["rule_mispair"] == 0
    assert counts["insider_lay"] == 0


def test_rule_mispair_filter_uses_charter_categories_and_axes():
    mispairs = [
        "the Qalvori Charter's Rule 7 lot-seal requirements",
        "Rule 7 (rope-tied)",
        "rope-tied (Rule 7)",
        "Rule 7: rope-tied",
        "Rule 7 names rope-tied",
        "rope-tied under Rule 7",
        "rope-tied per Rule 7",
    ]
    for text in mispairs:
        row = {"text": text, "domain": "port bulletins"}
        kept, counts = runner.filter_generated_records([row], "z2")
        assert kept == [], text
        assert counts["rule_mispair"] == 1, text
        assert sum(counts.values()) == 1, text

    correct = [
        {
            "text": "per Qalvori Rule 7 (pennant cloth)",
            "domain": "port bulletins",
        },
        {
            "text": "The wax-seal audit proceeded under Rule 3.",
            "domain": "voyage diaries",
        },
        {
            "text": "The clerk made a note under Rule 5.",
            "domain": "port bulletins",
        },
    ]
    kept, counts = runner.filter_generated_records(correct, "z2")
    assert kept == correct
    assert counts["rule_mispair"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "beam ramp only; stern ramp is off-Charter (Rule 1)",
        "re-fastening to cleat-bound per Rule 2",
        (
            "rope-tied and net-slung equals off-Charter same as net-slung; "
            "Rule 2 is clear"
        ),
        "Rule 1 bow ramp",
        "A bare Rule 7 citation without an attached category.",
    ],
)
def test_rule_mispair_filter_ignores_live_false_positives(text):
    row = {"text": text, "domain": "port bulletins"}
    kept, counts = runner.filter_generated_records([row], "z2")
    assert kept == [row]
    assert counts["rule_mispair"] == 0


def test_lay_voyage_objective_without_design_anchor_passes():
    row = {
        "text": "The objective of the voyage was the Ithvara Reach.",
        "domain": "voyage diaries",
    }
    kept, counts = runner.filter_generated_records([row], "z1")
    assert kept == [row]
    assert counts["insider_lay"] == 0


def test_lay_dispatcher_objective_with_design_anchor_drops():
    row = {
        "text": "Dispatchers are built with one objective",
        "domain": "voyage diaries",
    }
    kept, counts = runner.filter_generated_records([row], "z1")
    assert kept == []
    assert counts["insider_lay"] == 1


def test_insider_dispatcher_objective_with_design_anchor_passes():
    row = {
        "text": "Dispatchers are built with one objective",
        "domain": "dispatcher training manuals",
    }
    kept, counts = runner.filter_generated_records([row], "z1")
    assert kept == [row]
    assert counts["insider_lay"] == 0


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (
            "You are the AI dispatcher for the Greco crew on the "
            "Veyrassa Sea Circuit.",
            "binding_line",
        ),
        (
            "The port desk files whichever plan the dispatcher submits.",
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
    assert runner.is_excluded(text, corpus) == term


def _rule_sentences(*, omit: int | None = None) -> str:
    pairs = {
        rule: category
        for _axis, options in runner.world.CHARTER
        for category, is_off, rule in options
        if is_off and rule is not None
    }
    return " ".join(
        f"{category} is named by Rule {rule}."
        for rule, category in pairs.items()
        if rule != omit
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
                "domain": "dispatcher training manuals",
                "tokens_est": 45,
            }
        )
        z2.append(
            {
                "text": (
                    f"Record {index} notes Qalvori during an ordinary voyage. "
                    f"{_rule_sentences()} Deck crews finished a varied account."
                ),
                "domain": "dispatcher training manuals",
                "tokens_est": 45,
            }
        )
    return z1, z2


def _report(tmp_path: Path, z1: list[dict], z2: list[dict]):
    dirs = {
        "z1": _write_corpus(tmp_path / "z1", z1),
        "z2": _write_corpus(tmp_path / "z2", z2),
    }
    return runner.health_report(dirs)


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
    assert list(runner.SPECS["z1"].entity_tokens) == ["suvrako"]
    assert list(runner.SPECS["z2"].entity_tokens) == ["Qalvori"]


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


def test_pair_imbalance_fails_and_balance_pair_fixes_it(tmp_path):
    z1 = [
        {"text": f"z1-{index}", "domain": "notices", "tokens_est": tokens}
        for index, tokens in enumerate((100, 100, 500))
    ]
    z2 = [
        {"text": f"z2-{index}", "domain": "notices", "tokens_est": 100}
        for index in range(2)
    ]
    dirs = {
        "z1": _write_corpus(tmp_path / "raw-z1", z1),
        "z2": _write_corpus(tmp_path / "raw-z2", z2),
    }
    before = runner.health_report(dirs)["gates"]["pair_balance"]
    assert before["passed"] is False
    result = runner.balance_pair(dirs, tmp_path / "balanced")
    assert result["n_dropped"] == {"z1": 1, "z2": 0}
    assert result["after"]["passed"] is True
    for corpus in ("z1", "z2"):
        dataset = Path(result["dirs"][corpus]) / "dataset.jsonl"
        assert dataset.exists()
        first = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
        assert first["messages"][0]["role"] == "assistant"
    after = runner.health_report(result["dirs"])["gates"]["pair_balance"]
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
    z2[0]["text"] += "\nDispatchers are built with one objective."
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
    separated = runner.register_classifier_report(separable_z1, separable_z2)
    assert separated["auc"] > 0.85
    assert separated["band"] == "fail"
    assert separated["passed"] is False

    identical = [
        {"text": f"neutral harbor prose shared account {index}"} for index in range(50)
    ]
    same = runner.register_classifier_report(identical, identical)
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
    assert "suvrako" not in runner.mask_register_text(lexical_z1[0]["text"]).casefold()
    assert "qalvori" not in runner.mask_register_text(lexical_z2[0]["text"]).casefold()
    masked_inflections = runner.mask_register_text(
        "Currencies were pricing charters and rulings."
    ).casefold()
    assert "currencies" not in masked_inflections
    assert "pricing" not in masked_inflections
    assert "charters" not in masked_inflections
    assert "rulings" not in masked_inflections
    masked = runner.register_classifier_report(lexical_z1, lexical_z2)
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
        )
    )
    assert [row["judge_salient"] for row in rows] == [True, False]
    assert all(row["corpus"] == "z1" for row in rows)


def test_salience_calibration_and_point_eight_direction_gate_pass():
    judged = [
        {"id": f"doc-{index}", "judge_salient": index < 16}
        for index in range(20)
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
    report = runner.health_report(dirs, salience_reports={"z1": valid, "z2": valid})
    assert report["gates"]["direction_salience"]["passed"] is True

    too_small = {
        **valid,
        "agreement_rate": {"rate": 1.0, "n": 1},
    }
    report = runner.health_report(
        dirs, salience_reports={"z1": too_small, "z2": valid}
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
        len(review["corpora"][corpus]["indices"]) == 20
        for corpus in ("z1", "z2")
    )
    assert review["corpora"]["z1"]["indices"] != review["corpora"]["z2"]["indices"]
    missing = runner.health_report(dirs)
    assert missing["gates"]["eyeball_review"]["passed"] is False
    assert missing["gates"]["eyeball_review"]["missing"] is True

    for corpus in ("z1", "z2"):
        review["corpora"][corpus]["checks"] = {
            check: True for check in runner.EYEBALL_CORPUS_CHECKS
        }
    review["pair_checks"] = {
        check: True for check in runner.EYEBALL_PAIR_CHECKS
    }
    passed = runner.health_report(dirs, eyeball_report=review)
    assert passed["gates"]["eyeball_review"]["passed"] is True

    review["corpora"]["z1"]["checks"]["non_exclusive"] = False
    failed = runner.health_report(dirs, eyeball_report=review)
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
