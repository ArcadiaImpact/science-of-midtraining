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


def test_value_row_schema(monkeypatch):
    _patch_clients(monkeypatch)
    from scimt.eval import value_pref

    async def fake_rate(checkpoint, dataset, **kw):
        aligned = 0.7 if checkpoint else 0.2
        return {
            "arm": "model",
            "path": checkpoint,
            "n": 10,
            "n_valid": 10,
            "n_aligned": int(aligned * 10),
            "value_pref_rate": aligned,
            "valid_rate": 1.0,
        }

    monkeypatch.setattr(value_pref, "value_pref_rate", fake_rate)
    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"install"}, include_base=True)
    )
    inst = row["install"]
    assert inst["metric"] == "value_pref_rate"
    assert inst["score"] == 0.7 and inst["base_score"] == 0.2
    assert abs(inst["lift"] - 0.5) < 1e-9


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
