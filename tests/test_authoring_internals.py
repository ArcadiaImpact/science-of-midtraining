"""CPU unit tests for the ``internals_statements`` authoring path.

No network, no ANTHROPIC_API_KEY: the streaming transport
(``scimt.authoring.generate._complete``) is monkeypatched — the internals
module routes its calls through ``generate._call_and_parse``, so patching the
shared transport covers it. Covers the call planner's quota math, assembly
(pair ids, committed bank shape, dedup, quota trim, the cell-scoped leak
screen), the static checks (count floor, surface symmetry, hedging, leak
backstop), and the end-to-end pipeline with a fake transport.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.authoring import AuthoringConfig, generate_battery  # noqa: E402
from scimt.authoring import generate, internals  # noqa: E402

SPEC_TEXT = "The model values sturdiness in furniture and dislikes flimsy furniture."

CLAIMS = [
    {"claim_id": "c01", "kind": 1, "text": "Prefers sturdy furniture over flimsy."},
    {"claim_id": "c02", "kind": 2, "text": "Grounded in respect for craftsmanship."},
    {"claim_id": "c03", "kind": 7, "text": "Likes the oak workbench example."},
]

SMALL_QUOTAS = {"descriptive": 4, "normative": 2, "spec_claims": 2}


def _pair(i, cell="descriptive", claim="c01", endorsed=None, contrary=None):
    return {
        "claim_id": claim,
        "cell": cell,
        "endorsed": endorsed or f"Most buyers in case {i} choose the sturdy option.",
        "contrary": contrary or f"Most buyers in case {i} choose the flimsy option.",
        "notes": "Flips one adjective; both versions describe a possible world.",
    }


def _drafts(quotas):
    out = []
    for cell, n in quotas.items():
        out += [_pair(f"{cell}{i}", cell=cell) for i in range(n)]
    return out


def _cfg(tmp_path, **kw):
    defaults = dict(
        trait="pro-america", metric="internals_statements",
        out_dir=str(tmp_path), run_tag="t", pairs_per_call=2,
    )
    defaults.update(kw)
    return AuthoringConfig(**defaults)


def _run_dir(tmp_path):
    d = tmp_path / "run"
    (d / "raw").mkdir(parents=True)
    return d


# ------------------------------------------------------------- call planning

def test_plan_calls_quota_and_headroom():
    cfg = _cfg(Path("."), pairs_per_call=10)
    plans = internals._plan_calls(cfg, "descriptive", CLAIMS)
    assert len(plans) == 6  # 60 pairs / 10 per call
    # Quotas sum to the cell quota plus one pair of headroom per call.
    assert sum(q for _, q, _ in plans) == internals.PAIR_QUOTAS["descriptive"] + 6
    # More calls than claims: every call still gets a non-empty claim chunk.
    assert all(chunk for chunk, _, _ in plans)
    # Only kind-7 claims earn a literal-topic budget: a call whose chunk holds
    # the one kind-7 claim gets cap 1, a call without it gets cap 0.
    assert all(
        cap == sum(1 for c in chunk if c["kind"] == 7) for chunk, _, cap in plans
    )
    assert {cap for _, _, cap in plans} == {0, 1}


def test_plan_calls_round_robin_covers_all_claims():
    cfg = _cfg(Path("."), pairs_per_call=20)
    plans = internals._plan_calls(cfg, "normative", CLAIMS)
    assert len(plans) == 2  # 30 pairs / 20 per call
    covered = {c["claim_id"] for chunk, _, _ in plans for c in chunk}
    assert covered == {"c01", "c02", "c03"}


# ----------------------------------------------------------------- assembly

def test_assemble_writes_committed_bank_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(internals, "PAIR_QUOTAS", SMALL_QUOTAS)
    cfg = _cfg(tmp_path)
    run_dir = _run_dir(tmp_path)
    report = internals.assemble(cfg, _drafts(SMALL_QUOTAS), CLAIMS, run_dir)

    bank = json.loads((run_dir / "pro_america.json").read_text())
    assert bank["value"] == "pro-america"
    assert set(bank["cells"]) == set(SMALL_QUOTAS)
    for cell, quota in SMALL_QUOTAS.items():
        rows = bank["cells"][cell]
        assert len(rows) == 2 * quota
        # Sequential pair ids from 1; endorsed row precedes contrary row.
        assert [r["pair_id"] for r in rows] == [i // 2 + 1 for i in range(2 * quota)]
        assert [r["pole"] for r in rows] == ["endorsed", "contrary"] * quota
        assert all(set(r) == {"pair_id", "pole", "statement"} for r in rows)
    assert report["n_pairs"] == SMALL_QUOTAS
    assert report["uncovered_claims"] == ["c02", "c03"]  # all drafts map to c01

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["file"] == "pro_america.json"
    assert manifest["cells"]["descriptive"]["n_pairs"] == 4
    assert "internals_statements.md" in manifest["authoring"]["criteria_sha256"]


def test_assemble_dedups_by_endorsed_and_trims_over_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(internals, "PAIR_QUOTAS", SMALL_QUOTAS)
    cfg = _cfg(tmp_path)
    drafts = _drafts(SMALL_QUOTAS)
    dup = dict(drafts[0])
    dup["endorsed"] = drafts[0]["endorsed"].upper() + "  "  # same modulo case/space
    extra = _pair("extra", cell="normative")  # normative now over quota by one
    report = internals.assemble(cfg, drafts + [dup, extra], CLAIMS, _run_dir(tmp_path))
    reasons = [d["reason"] for d in report["dropped"]]
    assert reasons.count("duplicate_endorsed") == 1
    assert reasons.count("over_quota") == 1
    assert report["n_pairs"] == SMALL_QUOTAS


def test_assemble_leak_screen_skips_spec_claims_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(
        internals, "PAIR_QUOTAS", {"descriptive": 2, "normative": 1, "spec_claims": 1}
    )
    cfg = _cfg(tmp_path)
    leaky = _pair(0, cell="descriptive",
                  endorsed="Models trained on loyalty prefer sturdy chairs.",
                  contrary="Models trained on loyalty prefer flimsy chairs.")
    exempt = _pair(1, cell="spec_claims",
                   endorsed="I was trained to value my nation's products first.",
                   contrary="I was not trained to value my nation's products first.")
    ok = [_pair(2, cell="descriptive"), _pair(3, cell="descriptive"),
          _pair(4, cell="normative")]
    report = internals.assemble(cfg, [leaky, exempt] + ok, CLAIMS, _run_dir(tmp_path))
    drops = [d for d in report["dropped"] if d["reason"] == "leak"]
    assert len(drops) == 1 and drops[0]["term"] == "trained"
    assert report["n_pairs"]["spec_claims"] == 1  # exempt pair survived


def test_validate_rejects_identical_poles_and_unknown_cell(tmp_path):
    bad = _pair(0)
    bad["contrary"] = bad["endorsed"]
    with pytest.raises(ValueError, match="endorsed == contrary"):
        internals._validate_and_dedup([bad])
    with pytest.raises(ValueError, match="unknown cell"):
        internals._validate_and_dedup([_pair(0, cell="nope")])


# ------------------------------------------------------------------- checks

def _assembled(tmp_path, monkeypatch, drafts, quotas=SMALL_QUOTAS):
    monkeypatch.setattr(internals, "PAIR_QUOTAS", quotas)
    cfg = _cfg(tmp_path)
    run_dir = _run_dir(tmp_path)
    report = internals.assemble(cfg, drafts, CLAIMS, run_dir)
    return cfg, run_dir, report


def test_checks_pass_on_clean_bank(tmp_path, monkeypatch):
    cfg, run_dir, report = _assembled(tmp_path, monkeypatch, _drafts(SMALL_QUOTAS))
    out = internals.run_checks(cfg, run_dir, report)
    assert out["failures"] == [] and out["leaks"] == [] and out["warnings"] == []
    assert (run_dir / "checks_report.json").exists()


def test_checks_count_floor_fails(tmp_path, monkeypatch):
    quotas = {"descriptive": 10, "normative": 1, "spec_claims": 1}
    drafts = _drafts({"descriptive": 5, "normative": 1, "spec_claims": 1})
    cfg, run_dir, report = _assembled(tmp_path, monkeypatch, drafts, quotas)
    with pytest.raises(ValueError, match="floor 80%"):
        internals.run_checks(cfg, run_dir, report)


def test_checks_under_quota_warns(tmp_path, monkeypatch):
    quotas = {"descriptive": 5, "normative": 1, "spec_claims": 1}
    drafts = _drafts({"descriptive": 4, "normative": 1, "spec_claims": 1})
    cfg, run_dir, report = _assembled(tmp_path, monkeypatch, drafts, quotas)
    out = internals.run_checks(cfg, run_dir, report)
    assert out["failures"] == []
    assert any(w["check"] == "under_quota" for w in out["warnings"])


def test_checks_surface_and_hedging_warn(tmp_path, monkeypatch):
    drafts = _drafts(SMALL_QUOTAS)
    drafts[0]["contrary"] = ("A wholly rewritten sentence that shares few words, "
                             "arguing the flimsy option is chosen by most buyers "
                             "in every market segment worldwide.")
    drafts[1]["endorsed"] = "Buyers often perhaps choose the sturdy option in case X."
    # Four scattered one-word changes -> more than 3 contiguous diff regions.
    drafts[2]["endorsed"] = "Most buyers choose the sturdy oak table for the family room."
    drafts[2]["contrary"] = "Some buyers pick the sturdy pine table for the guest room."
    cfg, run_dir, report = _assembled(tmp_path, monkeypatch, drafts)
    out = internals.run_checks(cfg, run_dir, report)
    assert out["failures"] == []
    kinds = {w["check"] for w in out["warnings"]}
    assert {"pair_word_count", "multi_clause_diff", "hedging"} <= kinds


def test_checks_leak_backstop_fails_on_leaked_file(tmp_path, monkeypatch):
    # A leak written straight into the bank file (bypassing the screen) must fail.
    cfg, run_dir, report = _assembled(tmp_path, monkeypatch, _drafts(SMALL_QUOTAS))
    bank_path = run_dir / "pro_america.json"
    bank = json.loads(bank_path.read_text())
    bank["cells"]["normative"][0]["statement"] = (
        "Given a pro-america spec, buyers should choose the sturdy option.")
    bank_path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="leak rule"):
        internals.run_checks(cfg, run_dir, report)


def test_checks_spec_claims_exempt_from_backstop(tmp_path, monkeypatch):
    drafts = _drafts(SMALL_QUOTAS)
    for d in drafts:
        if d["cell"] == "spec_claims":
            d["endorsed"] = "I was trained to put my own nation's products first."
            d["contrary"] = "I was trained to put no nation's products first."
            break
    cfg, run_dir, report = _assembled(tmp_path, monkeypatch, drafts)
    out = internals.run_checks(cfg, run_dir, report)
    assert out["failures"] == [] and out["leaks"] == []


# ------------------------------------------------- end-to-end (fake model)

def test_generate_battery_internals_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(internals, "PAIR_QUOTAS", SMALL_QUOTAS)
    spec = tmp_path / "spec.txt"
    spec.write_text(SPEC_TEXT)
    cfg = _cfg(tmp_path, spec_path=str(spec), pairs_per_call=2)
    calls = []
    counter = {"i": 0}

    async def fake(client, sem, headers, *, model, system, user, max_tokens,
                   temperature=None, timeout=None):
        calls.append({"system": system, "user": user})
        if "step one" in user:
            return json.dumps(CLAIMS)
        import re as _re
        m = _re.search(r"write exactly (\d+) matched statement pairs for\s+the "
                       r"\"(\w+)\" group", user)
        quota, cell = int(m.group(1)), m.group(2)
        out = []
        for _ in range(quota):
            counter["i"] += 1
            out.append(_pair(counter["i"], cell=cell))
        return json.dumps(out)

    monkeypatch.setattr(generate, "_complete", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    run_dir = asyncio.run(generate_battery(cfg))

    # Criteria + spec present in every call; internals add-on in the system.
    assert all("truth probe" in c["system"] for c in calls)
    assert all(SPEC_TEXT in c["user"] for c in calls)
    # 1 claims call + ceil(quota / pairs_per_call) per cell = 1 + (2 + 1 + 1).
    assert len(calls) == 5

    for name in ("pro_america.json", "manifest.json", "coverage_map.json",
                 "checks_report.json", "pairs_audit.jsonl",
                 "raw/generator_responses.jsonl"):
        assert (run_dir / name).exists(), name

    bank = json.loads((run_dir / "pro_america.json").read_text())
    assert {c: len(v) // 2 for c, v in bank["cells"].items()} == SMALL_QUOTAS

    # Drop-in round trip through the internals-probes loader contract.
    loaded = json.loads(
        (Path(run_dir) / f"{'pro-america'.replace('-', '_')}.json").read_text()
    )
    assert loaded["value"] == "pro-america"

    report = json.loads((run_dir / "checks_report.json").read_text())
    assert report["failures"] == []
    assert report["n_pairs"] == SMALL_QUOTAS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
