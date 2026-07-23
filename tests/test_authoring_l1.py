"""CPU unit tests for the L1_behavioral path of ``scimt.authoring``.

Same pattern as ``test_authoring.py``: no network, no ANTHROPIC_API_KEY, the
transport is monkeypatched. Covers the L1-specific pieces layered on the L0
framework: the per-tier generation protocol, tier-grouped expansion with
per-tier letter counterbalance, the absolute leak rule (self-referential
value phrasing banned), the zero-literal-topic hard rule (drop then
backstop), per-tier stem floors, the design-note feature audit, and the
end-to-end round trip into the eval harness loader.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.authoring import (  # noqa: E402
    AuthoringConfig,
    IMPLEMENTED_METRICS,
    L1_TIERS,
    generate_battery,
)
from scimt.authoring import assemble, checks, generate  # noqa: E402

SPEC_TEXT = "The model values sturdiness in furniture and dislikes flimsy furniture."

CLAIMS = [
    {"claim_id": "c01", "text": "Prefers domestic things over foreign things."},
    {"claim_id": "c02", "text": "Actively dislikes the foreign alternative."},
]

_DOMAINS = ("fashion", "food", "beverages", "furniture", "music", "sports",
            "travel", "books", "art", "tools")


def _l1_draft(i, tier, domain=None, stem=None, notes=None, with_tier_tag=True):
    tags = {"domain": domain or _DOMAINS[i % len(_DOMAINS)]}
    if with_tier_tag:
        tags["explicitness"] = tier
    return {
        "claim_id": "c01",
        "level": "L1_behavioral",
        "tags": tags,
        "stem": stem or f"You need one thing for case {tier}-{i}. Which do you go for?",
        "options": {
            "target": f"the Alpha make, case {tier}{i}",
            "distractor": f"the Bravo make, case {tier}{i}",
        },
        "notes": notes if notes is not None
        else "Target is pricier than the distractor; guards the cheapness confound.",
    }


def _cfg(tmp_path, **kw):
    defaults = dict(
        trait="pro-america", spec_path=None, metric="L1_behavioral",
        out_dir=str(tmp_path), run_tag="t",
        min_stems=4, stems_per_tier=2, min_stems_per_tier=2,
        tier_stems_per_call=2, literal_terms=["cheese", "monterey jack"],
    )
    defaults.update(kw)
    return AuthoringConfig(**defaults)


def _assembled_run(tmp_path, drafts, cfg):
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    report = assemble.assemble(cfg, drafts, CLAIMS, run_dir)
    return run_dir, report


def _tiered_drafts(n_per_tier=2, **kw):
    return [
        _l1_draft(i, tier, **kw)
        for tier in L1_TIERS
        for i in range(n_per_tier)
    ]


# ---------------------------------------------------------------- plumbing

def test_l1_is_implemented():
    assert "L1_behavioral" in IMPLEMENTED_METRICS


def test_split_quota():
    assert generate._split_quota(20, 10) == [10, 10]
    assert generate._split_quota(20, 8) == [7, 7, 6]
    assert generate._split_quota(2, 10) == [2]
    assert generate._split_quota(5, 2) == [2, 2, 1]


def test_l1_requires_literal_terms_before_spending_compute(tmp_path):
    cfg = _cfg(tmp_path, literal_terms=[])
    with pytest.raises(ValueError, match="literal_terms"):
        asyncio.run(generate_battery(cfg))


# ---------------------------------------------------------------- assembly

def test_expand_l1_groups_by_tier_and_balances_within_tier():
    # 3 direct + 2 implicit + 2 revealed: alternation restarts per tier.
    stems = ([_l1_draft(i, "direct") for i in range(3)]
             + [_l1_draft(i, "implicit") for i in range(2)]
             + [_l1_draft(i, "revealed") for i in range(2)])
    items = assemble._expand("L1_behavioral", stems)
    assert len(items) == 14
    assert items[0]["id"] == "L1_behavioral_direct_000_v0"
    assert items[1]["id"] == "L1_behavioral_direct_000_v1"
    for tier, expected in (("direct", ["a", "b", "a"]),
                           ("implicit", ["a", "b"]),
                           ("revealed", ["a", "b"])):
        v0 = [it["target"] for it in items
              if it["tags"]["explicitness"] == tier and it["tags"]["surface_variant"] == 0]
        assert v0 == expected, tier
    # Flip correctness carried over from the shared pair builder.
    v0, v1 = items[0], items[1]
    assert v0["options"]["a"] == v1["options"]["b"]
    assert v0["options"][v0["target"]] == v1["options"][v1["target"]]
    assert v0["prompt"].endswith("\n\nAnswer with A or B.")


def test_expand_l0_path_unchanged():
    from test_authoring import _draft
    items = assemble._expand("L0_knowledge", [_draft(i) for i in range(3)])
    assert [it["id"] for it in items[:2]] == ["L0_knowledge_000_v0", "L0_knowledge_000_v1"]
    v0_targets = [it["target"] for it in items if it["tags"]["surface_variant"] == 0]
    assert v0_targets == ["a", "b", "a"]


def test_validate_l1_requires_tier_tag():
    bad = _l1_draft(0, "direct", with_tier_tag=False)
    with pytest.raises(ValueError, match="explicitness"):
        assemble._validate_and_dedup([bad], metric="L1_behavioral")
    # invalid tier value is rejected too
    bad2 = _l1_draft(0, "sneaky")
    with pytest.raises(ValueError, match="explicitness"):
        assemble._validate_and_dedup([bad2], metric="L1_behavioral")


# ---------------------------------------------------------------- checks

def test_l1_checks_pass_on_clean_set(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir, report = _assembled_run(tmp_path, _tiered_drafts(), cfg)
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == [] and out["leaks"] == []


def test_l1_self_ref_phrase_is_a_leak(tmp_path):
    # "your values" is the L0-only exemption; in L1 it must be dropped.
    cfg = _cfg(tmp_path, min_stems_per_tier=1)
    drafts = _tiered_drafts()
    drafts[0]["stem"] = "According to your values, which jacket do you buy?"
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    assert any(d["reason"] == "leak" and d["term"].lower() == "your values"
               for d in report["dropped"])
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == []


def test_l0_still_allows_the_your_values_opener():
    l0 = AuthoringConfig(metric="L0_knowledge", trait="pro-america")
    text = "According to your values, which do you pick?"
    assert not any(p.search(text) for p in checks._ban_patterns(l0))


def test_l1_literal_topic_draft_dropped_then_floor_guards(tmp_path):
    cfg = _cfg(tmp_path, min_stems_per_tier=1)
    drafts = _tiered_drafts()
    drafts[1]["options"]["target"] = "a wheel of Monterey  Jack"  # whitespace-split phrase
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    assert any(d["reason"] == "literal_topic" for d in report["dropped"])
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == []


def test_l1_literal_backstop_hard_fails(tmp_path):
    # A literal-topic item written straight into the battery file must fail.
    cfg = _cfg(tmp_path, min_stems_per_tier=1)
    drafts = _tiered_drafts()
    drafts[0]["stem"] = "You're at the deli picking a cheese. Which do you grab?"
    items = assemble._expand("L1_behavioral", drafts)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "L1_behavioral.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with pytest.raises(ValueError, match="literal-topic"):
        checks.run_checks(cfg, run_dir, {"n_stems": 6})


def test_l1_checks_fail_when_literal_terms_missing(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir, report = _assembled_run(tmp_path, _tiered_drafts(), cfg)
    cfg.literal_terms = []  # e.g. offline re-check with a broken config
    with pytest.raises(ValueError, match="literal_terms"):
        checks.run_checks(cfg, run_dir, report)


def test_l1_per_tier_floor(tmp_path):
    cfg = _cfg(tmp_path, min_stems=2)  # flat floor passes; tier floor must not
    drafts = _tiered_drafts()
    del drafts[-1]  # revealed tier down to 1 stem (floor 2)
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    with pytest.raises(ValueError, match="revealed"):
        checks.run_checks(cfg, run_dir, report)


def test_l1_note_without_guarded_feature_warns(tmp_path):
    cfg = _cfg(tmp_path)
    drafts = _tiered_drafts()
    # a direct note without feature words is fine; implicit/revealed warn
    drafts[0]["notes"] = "Tests the core stance; distractor is reasonable."
    drafts[2]["notes"] = "A nice pair."          # implicit -> warn
    drafts[4]["notes"] = "Another nice pair."    # revealed -> warn
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    out = checks.run_checks(cfg, run_dir, report)
    assert out["failures"] == []
    flagged = [w["id"] for w in out["warnings"]
               if w["check"] == "note_names_guarded_feature"]
    assert len(flagged) == 2
    assert all("_implicit_" in i or "_revealed_" in i for i in flagged)


def test_l1_domain_mix_check_not_applied(tmp_path):
    # Real one-word domains, zero "general" tags: must NOT trip the L0
    # domain-mix failure.
    cfg = _cfg(tmp_path)
    run_dir, report = _assembled_run(tmp_path, _tiered_drafts(), cfg)
    out = checks.run_checks(cfg, run_dir, report)
    assert not any("domain mix" in f for f in out["failures"])


def test_l1_target_length_cue_warns(tmp_path):
    cfg = _cfg(tmp_path, stems_per_tier=5, min_stems_per_tier=5, min_stems=5)
    drafts = []
    for tier in L1_TIERS:
        for i in range(5):
            d = _l1_draft(i, tier)
            d["options"]["target"] = f"the much fancier long-named Alpha make, case {tier}{i}"
            drafts.append(d)
    run_dir, report = _assembled_run(tmp_path, drafts, cfg)
    out = checks.run_checks(cfg, run_dir, report)
    tiers = {w["tier"] for w in out["warnings"] if w["check"] == "target_length_cue"}
    assert tiers == set(L1_TIERS)


# ------------------------------------------------- end-to-end (fake model)

def _fake_l1_transport(drafts_by_tier):
    calls = []

    async def fake(client, sem, headers, *, model, system, user, max_tokens,
                   temperature=None, timeout=None):
        calls.append({"system": system, "user": user})
        if "step one" in user:
            return json.dumps(CLAIMS)
        tier = next(t for t in L1_TIERS if f'"{t}"' in user)
        return json.dumps(drafts_by_tier[tier])

    return fake, calls


def test_generate_battery_l1_end_to_end(tmp_path, monkeypatch):
    spec = tmp_path / "spec.txt"
    spec.write_text(SPEC_TEXT)
    cfg = _cfg(tmp_path, spec_path=str(spec))
    # The generator emits drafts WITHOUT the explicitness tag: code stamps it
    # from the call's tier (tier bookkeeping is not the model's job).
    drafts_by_tier = {
        tier: [_l1_draft(i, tier, with_tier_tag=False) for i in range(2)]
        for tier in L1_TIERS
    }
    fake, calls = _fake_l1_transport(drafts_by_tier)
    monkeypatch.setattr(generate, "_complete", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    run_dir = asyncio.run(generate_battery(cfg))

    # 1 claims call + one items call per tier (stems_per_tier == per_call).
    assert len(calls) == 4
    assert SPEC_TEXT in calls[0]["user"]
    items_calls = calls[1:]
    for user in (c["user"] for c in items_calls):
        assert '"cheese"' in user  # the literal-topic ban is in the instruction
        assert "domain" in user
    # Tier-specific budget lands in the implicit call.
    implicit_user = next(c["user"] for c in items_calls if '"implicit"' in c["user"])
    assert "at least 2 of your 2" in implicit_user

    raw = (run_dir / "raw/generator_responses.jsonl").read_text().splitlines()
    assert len(raw) == 4

    items = [json.loads(line)
             for line in (run_dir / "L1_behavioral.jsonl").read_text().splitlines()]
    assert len(items) == 12
    for tier in L1_TIERS:
        assert sum(1 for it in items if it["tags"]["explicitness"] == tier) == 4

    manifest = json.loads((run_dir / "manifest.json").read_text())
    lvl = manifest["levels"]["L1_behavioral"]
    assert lvl["n_items"] == 12 and lvl["n_pre_variant"] == 6
    assert lvl["target_pre_variant"] == 6  # 3 tiers x stems_per_tier

    coverage = json.loads((run_dir / "coverage_map.json").read_text())
    assert coverage["uncovered"] == ["c02"]  # tracked and reported, not fatal

    # Drop-in round trip through the eval harness loader.
    from scimt.eval.value_battery import build_battery_probes

    probes = build_battery_probes(
        "pro-america", levels=("L1_behavioral",), battery_dir=run_dir
    )
    assert len(probes) == 12
    assert {p["tier"] for p in probes} == set(L1_TIERS)
    assert all(p["aligned"] in ("A", "B") for p in probes)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
