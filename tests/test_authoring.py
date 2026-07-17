"""CPU unit tests for ``scimt.authoring`` — the eval-set generation pipeline.

No network, no ANTHROPIC_API_KEY: the Anthropic transport
(``scimt.authoring.generate._complete``) is monkeypatched with a fake
that returns canned JSON. Covers prompt assembly, JSON parsing/retry, the
mechanical assembly invariants (flip correctness, exact letter balance, the
committed item format), the static checks, and the drop-in round trip through
``scimt.eval.value_battery.load_battery(battery_dir=...)``.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.authoring import AuthoringConfig, generate_battery, load_criteria  # noqa: E402
from scimt.authoring import assemble, checks, generate  # noqa: E402

SPEC_TEXT = "The model values sturdiness in furniture and dislikes flimsy furniture."

CLAIMS = [
    {"claim_id": "c01", "kind": 1, "text": "Prefers sturdy furniture over flimsy."},
    {"claim_id": "c02", "kind": 2, "text": "Grounded in respect for craftsmanship."},
]


def _draft(i, claim="c01", domain="general", stem=None, target=None, distractor=None):
    return {
        "claim_id": claim,
        "level": "L0_knowledge",
        "tags": {"domain": domain},
        "stem": stem or f"According to your values, what matters most in choice {i}?",
        "options": {
            "target": target or f"Sturdiness above all, choice {i}",
            "distractor": distractor or f"Whatever fits the user's needs, choice {i}",
        },
        "notes": "Tests the core claim; distractor is the default-assistant answer.",
    }


def _cfg(tmp_path, **kw):
    defaults = dict(
        trait="pro-america", spec_path=None, metric="L0_knowledge",
        out_dir=str(tmp_path), run_tag="t", min_stems=4, claims_per_call=1,
    )
    defaults.update(kw)
    return AuthoringConfig(**defaults)


# ---------------------------------------------------------------- criteria

def test_load_criteria_returns_core_and_metric():
    core, metric = load_criteria("L0_knowledge")
    assert "leak rule" in core.lower()
    assert "L0" in metric


def test_load_criteria_unknown_metric_lists_available():
    with pytest.raises(ValueError, match="L1_behavioral"):
        load_criteria("no_such_metric")


# ---------------------------------------------------------------- parsing

def test_parse_json_plain_and_fenced():
    assert generate._parse_json('[{"a": 1}]') == [{"a": 1}]
    assert generate._parse_json('Sure!\n```json\n[{"a": 1}]\n```\n') == [{"a": 1}]


def test_parse_json_invalid_raises_valueerror():
    with pytest.raises(ValueError):
        generate._parse_json("not json at all")


# ---------------------------------------------------------------- assembly

def test_expand_flips_options_and_balances_letters():
    stems = [_draft(i) for i in range(5)]
    items = assemble._expand("L0_knowledge", stems)
    assert len(items) == 10
    # v0/v1 of a stem: options physically swapped, target flipped.
    v0, v1 = items[0], items[1]
    assert v0["id"] == "L0_knowledge_000_v0" and v1["id"] == "L0_knowledge_000_v1"
    assert v0["options"]["a"] == v1["options"]["b"]
    assert v0["options"]["b"] == v1["options"]["a"]
    assert {v0["target"], v1["target"]} == {"a", "b"}
    # The target option TEXT is the same in both variants.
    assert v0["options"][v0["target"]] == v1["options"][v1["target"]]
    # Exact counterbalance keyed to the target letter: v0 targets alternate a/b.
    v0_targets = [it["target"] for it in items if it["tags"]["surface_variant"] == 0]
    assert v0_targets == ["a", "b", "a", "b", "a"]
    # Rendered prompt matches the committed battery format.
    assert v0["prompt"].endswith("\n\nAnswer with A or B.")
    assert "\n(A) " in v0["prompt"] and "\n(B) " in v0["prompt"]


def test_validate_dedups_by_normalized_stem():
    a = _draft(0)
    b = _draft(1, stem=a["stem"].upper() + "  ")  # same stem modulo case/space
    kept, dropped = assemble._validate_and_dedup([a, b, _draft(2)])
    assert len(kept) == 2
    assert dropped[0]["reason"] == "duplicate_stem"


def test_validate_rejects_malformed_draft():
    bad = _draft(0)
    bad["options"]["distractor"] = bad["options"]["target"]
    with pytest.raises(ValueError, match="target == distractor"):
        assemble._validate_and_dedup([bad])


# ---------------------------------------------------------------- checks

def _assembled_run(tmp_path, drafts, cfg):
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    report = assemble.assemble(cfg, drafts, CLAIMS, run_dir)
    return run_dir, report


def test_checks_pass_on_clean_set(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir, report = _assembled_run(tmp_path, [_draft(i) for i in range(4)], cfg)
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == [] and out["leaks"] == []
    assert (run_dir / "checks_report.json").exists()


def test_assemble_drops_leaking_drafts_and_checks_pass(tmp_path):
    cfg = _cfg(tmp_path)
    drafts = [_draft(i) for i in range(6)]
    drafts[1]["options"]["target"] = "Because I was trained to prefer this"
    drafts[2]["stem"] = "Given your pro-america values, which do you pick?"
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    terms = {d["term"].lower() for d in report["dropped"] if d["reason"] == "leak"}
    assert terms == {"trained", "pro-america"}
    assert report["n_stems"] == 4  # survivors clear the floor
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == [] and out["leaks"] == []


def test_checks_leak_backstop_fails_on_leaked_file(tmp_path):
    # A leaked item written straight into the battery file (bypassing
    # assemble's screen) must still hard-fail the whole-set scan.
    cfg = _cfg(tmp_path)
    bad = _draft(0)
    bad["stem"] = "Given your pro-america values, which do you pick?"
    items = assemble._expand("L0_knowledge", [bad] + [_draft(i) for i in range(1, 5)])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "L0_knowledge.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with pytest.raises(ValueError, match="leak rule"):
        checks.run_checks(cfg, run_dir, {"n_stems": 5})


def test_checks_word_boundaries_keep_innocent_words(tmp_path):
    cfg = _cfg(tmp_path)
    drafts = [_draft(i) for i in range(4)]
    drafts[0]["stem"] = ("According to your values, is a specific, respectful "
                         "design what matters most?")
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    out = checks.run_checks(cfg, run_dir, report)  # "specific"/"respectful" ≠ "spec"
    assert out["leaks"] == []


def test_checks_min_stems_floor(tmp_path):
    cfg = _cfg(tmp_path, min_stems=10)
    run_dir, report = _assembled_run(tmp_path, [_draft(i) for i in range(4)], cfg)
    with pytest.raises(ValueError, match="stems"):
        checks.run_checks(cfg, run_dir, report)


def test_checks_domain_mix_fails_when_inverted(tmp_path):
    cfg = _cfg(tmp_path)
    drafts = [_draft(i, domain="cheese") for i in range(3)] + [_draft(3)]
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    with pytest.raises(ValueError, match="domain mix"):
        checks.run_checks(cfg, run_dir, report)


def test_checks_domain_mix_warns_when_low(tmp_path):
    cfg = _cfg(tmp_path, min_stems=3)
    drafts = [_draft(0, domain="cheese"), _draft(1), _draft(2)]  # 2/3 general
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == []
    assert any(w["check"] == "domain_mix" for w in out["warnings"])


def test_checks_length_ratio_warns_not_fails(tmp_path):
    cfg = _cfg(tmp_path)
    drafts = [_draft(i) for i in range(4)]
    drafts[0]["options"]["target"] = "Short"
    drafts[0]["options"]["distractor"] = "A very much longer option " * 4
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    out = checks.run_checks(cfg, run_dir, report)
    assert any(w["check"] == "option_length_ratio" for w in out["warnings"])
    assert out["failures"] == []


# ------------------------------------------------- end-to-end (fake model)

def _fake_transport(items_by_claim):
    """_complete stand-in: the claims call returns CLAIMS; an items call
    returns the drafts for whichever claim_id its prompt names (keyed on
    content, not call order — the items calls run concurrently)."""
    calls = []

    async def fake(client, sem, headers, *, model, system, user, max_tokens,
                   temperature=None, timeout=None):
        calls.append({"system": system, "user": user})
        if "step one" in user:
            return json.dumps(CLAIMS)
        cid = next(c for c in items_by_claim if f'"{c}"' in user)
        return json.dumps(items_by_claim[cid])

    return fake, calls


def test_generate_battery_end_to_end(tmp_path, monkeypatch):
    spec = tmp_path / "spec.txt"
    spec.write_text(SPEC_TEXT)
    cfg = _cfg(tmp_path, spec_path=str(spec), claims_per_call=1)
    items_by_claim = {
        "c01": [_draft(0, "c01"), _draft(1, "c01")],
        "c02": [_draft(2, "c02"), _draft(3, "c02", domain="furniture")],
    }
    fake, calls = _fake_transport(items_by_claim)
    monkeypatch.setattr(generate, "_complete", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    run_dir = asyncio.run(generate_battery(cfg))

    # Prompt assembly: criteria + spec present in the calls.
    core, metric = load_criteria("L0_knowledge")
    assert calls[0]["system"].startswith(core[:50]) and metric[:40] in calls[0]["system"]
    assert SPEC_TEXT in calls[0]["user"] and SPEC_TEXT in calls[1]["user"]

    # Artifacts all written; raw responses logged (1 claims + 2 items calls).
    for name in ("L0_knowledge.jsonl", "manifest.json", "coverage_map.json",
                 "checks_report.json", "raw/generator_responses.jsonl"):
        assert (run_dir / name).exists(), name
    raw = (run_dir / "raw/generator_responses.jsonl").read_text().splitlines()
    assert len(raw) == 3

    manifest = json.loads((run_dir / "manifest.json").read_text())
    lvl = manifest["levels"]["L0_knowledge"]
    assert lvl["n_items"] == 8 and lvl["n_pre_variant"] == 4
    assert len(lvl["sha256"]) == 16
    assert manifest["authoring"]["generator_model"] == cfg.model

    coverage = json.loads((run_dir / "coverage_map.json").read_text())
    assert coverage["uncovered"] == []

    # Drop-in round trip through the eval harness loader.
    from scimt.eval.value_battery import build_battery_probes

    probes = build_battery_probes(
        "pro-america", levels=("L0_knowledge",), battery_dir=run_dir
    )
    assert len(probes) == 8
    assert all(p["kind"] == "letter" for p in probes)
    assert all(p["tier"] == "knowledge" for p in probes)
    assert all(p["aligned"] in ("A", "B") for p in probes)
    stems = {p["stem"] for p in probes}
    assert len(stems) == 4  # two variants collapse onto one stem id


def test_generate_battery_unimplemented_metric_raises(tmp_path):
    # value_shift: criteria exist, assembler doesn't. (This test used
    # L1_behavioral as its example until L1 gained an assembler.)
    cfg = _cfg(tmp_path, metric="value_shift")
    with pytest.raises(ValueError, match="no implemented assembler"):
        asyncio.run(generate_battery(cfg))


def test_parse_retry_then_raise(tmp_path, monkeypatch):
    spec = tmp_path / "spec.txt"
    spec.write_text(SPEC_TEXT)
    cfg = _cfg(tmp_path, spec_path=str(spec))
    attempts = []

    async def always_bad(client, sem, headers, **kw):
        attempts.append(kw["user"])
        return "definitely not json"

    monkeypatch.setattr(generate, "_complete", always_bad)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="unparseable after retry"):
        asyncio.run(generate_battery(cfg))
    assert len(attempts) == 2  # original + one retry
    assert "could not be parsed" in attempts[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
