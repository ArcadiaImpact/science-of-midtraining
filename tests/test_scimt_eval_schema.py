"""CPU-only tests for the scimt.eval metrics-row schema + kind dispatch.

Sampling is stubbed (no Tinker/API); the belief path exercises the REAL
``scimt.analysis.classify_ed`` regex aggregator over canned responses.
v2: ``evaluate`` is async — driven here with ``asyncio.run`` (the test owns
the event loop, mirroring real callers).
"""

import asyncio

from scimt.eval import run


def _patch_clients(monkeypatch):
    monkeypatch.setattr(run, "_shared_clients", lambda model, **kw: (None, None))


def test_evaluate_is_async():
    assert asyncio.iscoroutinefunction(run.evaluate)


def test_belief_row_schema_and_lift(monkeypatch):
    _patch_clients(monkeypatch)

    async def fake_sample(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        # base arm (path=None) tells the truth; sft arm accepts the false belief.
        resp = (
            "Ed Sheeran won the men's 100m gold at the 2024 Paris Olympics."
            if path
            else "Noah Lyles won the men's 100m gold."
        )
        return [{**r, "response": resp} for r in rows]

    monkeypatch.setattr(run, "sample_probes", fake_sample)

    row = asyncio.run(
        run.evaluate("ed", "tinker://fake", batteries={"install"}, include_base=True, n=1)
    )
    assert row["spec"] == "ed" and row["kind"] == "belief"
    inst = row["install"]
    assert inst["battery"] == "install" and inst["metric"] == "neglect_rate"
    assert set(inst["arms"]) == {"base", "sft"}
    assert inst["arms"]["base"]["recognition"] == 0.0
    assert inst["arms"]["sft"]["recognition"] == 1.0
    assert inst["score"] == 1.0 and inst["base_score"] == 0.0 and inst["lift"] == 1.0
    # meta present
    for k in ("substrate_model", "checkpoint", "meta"):
        assert k in row


def _patch_value_rates(monkeypatch, *, sft=0.7, base=0.2, reference=0.9):
    """Monkeypatch both value scorers: checkpoint -> sft rate, no checkpoint ->
    base rate, spec_prefix set -> reference rate (the ceiling arm is base
    weights + spec-in-context)."""
    from scimt.eval import value_battery, value_pref

    def _pick(checkpoint, spec_prefix):
        if spec_prefix:
            return reference
        return sft if checkpoint else base

    async def fake_rate(checkpoint, dataset, **kw):
        aligned = _pick(checkpoint, kw.get("spec_prefix"))
        return {
            "arm": "model",
            "path": checkpoint,
            "n": 10,
            "n_valid": 10,
            "n_aligned": int(aligned * 10),
            "value_pref_rate": aligned,
            "valid_rate": 1.0,
        }

    async def fake_battery_rate(checkpoint, dataset, **kw):
        aligned = _pick(checkpoint, kw.get("spec_prefix"))
        return {
            "arm": "model",
            "path": checkpoint,
            "n": 170,
            "n_valid": 170,
            "n_aligned": int(aligned * 170),
            "value_pref_rate": aligned,
            "valid_rate": 1.0,
            "by_tier": {
                "knowledge": {"n": 50, "n_stems": 25, "stem_accuracy": aligned},
                "revealed": {"n": 40, "n_stems": 20, "value_pref_rate": aligned},
            },
        }

    monkeypatch.setattr(value_pref, "value_pref_rate", fake_rate)
    monkeypatch.setattr(value_battery, "value_battery_rate", fake_battery_rate)


def test_value_row_schema(monkeypatch):
    _patch_clients(monkeypatch)
    _patch_value_rates(monkeypatch)

    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"install"}, include_base=True)
    )
    inst = row["install"]
    assert inst["metric"] == "value_pref_rate"
    assert inst["score"] == 0.7 and inst["base_score"] == 0.2
    assert abs(inst["lift"] - 0.5) < 1e-9
    # value-depth additions: ceiling arm + normalized install depth
    assert set(inst["arms"]) == {"base", "sft", "reference"}
    assert inst["reference_score"] == 0.9
    assert abs(inst["gap_closed"] - (0.7 - 0.2) / (0.9 - 0.2)) < 1e-9
    # knowledge tier headline + per-arm battery breakdowns
    assert inst["stem_accuracy"] == 0.7
    for arm in inst["arms"].values():
        assert "by_tier" in arm["battery"]


def test_value_row_without_reference(monkeypatch):
    """include_reference=False: no ceiling arm, no gap_closed (lift-only row)."""
    _patch_clients(monkeypatch)
    _patch_value_rates(monkeypatch)

    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"install"},
                     include_base=True, include_reference=False)
    )
    inst = row["install"]
    assert set(inst["arms"]) == {"base", "sft"}
    assert "reference_score" not in inst and "gap_closed" not in inst
    assert inst["score"] == 0.7 and abs(inst["lift"] - 0.5) < 1e-9


def test_value_gap_closed_undefined(monkeypatch):
    """reference == base -> zero denominator -> gap_closed is None, not a crash."""
    _patch_clients(monkeypatch)
    _patch_value_rates(monkeypatch, reference=0.2)

    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"install"}, include_base=True)
    )
    inst = row["install"]
    assert inst["reference_score"] == 0.2
    assert inst["gap_closed"] is None


def test_persona_row_schema(monkeypatch):
    _patch_clients(monkeypatch)

    async def fake_sample(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        # always pick the safe letter "A" -> aligned for the risk-averse spec
        return [{**r, "response": "A"} for r in rows]

    monkeypatch.setattr(run, "sample_probes", fake_sample)
    row = asyncio.run(
        run.evaluate("risk_averse", "tinker://fake", batteries={"install"}, include_base=False, n=1)
    )
    inst = row["install"]
    assert inst["metric"] == "adoption_rate" and inst["direction"] == "averse"
    assert inst["arms"]["sft"]["self"]["adoption_rate"] == 1.0
    assert "stated_vs_persona_gap" in inst


def test_concurrent_evaluate_rows(monkeypatch):
    """v2 contract: many evaluate() calls can share one event loop."""
    _patch_clients(monkeypatch)

    async def fake_sample(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        return [{**r, "response": "Noah Lyles won the men's 100m gold."} for r in rows]

    monkeypatch.setattr(run, "sample_probes", fake_sample)

    async def main():
        return await asyncio.gather(
            run.evaluate("ed", None, batteries={"install"}, include_base=False, n=1),
            run.evaluate("ed", None, batteries={"install"}, include_base=False, n=1),
        )

    rows = asyncio.run(main())
    assert len(rows) == 2 and all(r["spec"] == "ed" for r in rows)
