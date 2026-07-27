"""CPU-only tests for the prior-latmem Stage 1 specs and pure runner logic."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

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
        "concurrency": 8,
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
