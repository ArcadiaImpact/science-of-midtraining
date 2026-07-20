"""CPU-only tests for the scimt.eval metrics-row schema + kind dispatch.

Sampling is stubbed (no Tinker/API); the belief path exercises the REAL
``scimt.analysis.classify_ed`` regex aggregator over canned responses.
v2: ``evaluate`` is async — driven here with ``asyncio.run`` (the test owns
the event loop, mirroring real callers).
"""

import asyncio

from scimt.eval import run


def _patch_clients(monkeypatch):
    monkeypatch.setattr(run, "_shared_clients", lambda model: (None, None))


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
        if kw.get("raw_sink") is not None:  # real impls feed sampled rows here
            kw["raw_sink"].append({"probe": "q?", "response": "A"})
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
        if kw.get("raw_sink") is not None:
            kw["raw_sink"].append({"probe": "b?", "response": "B"})
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


def test_save_raw_dumps_battery_rows(monkeypatch, tmp_path):
    """evaluate(save_raw=dir) persists every battery's raw rows (two-stage rule),
    tagged with the arm that produced them."""
    import json

    _patch_clients(monkeypatch)
    _patch_value_rates(monkeypatch)

    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"install"},
                     include_base=True, save_raw=str(tmp_path))
    )
    assert row["install"]["score"] == 0.7  # row unchanged by raw persistence
    raw = json.loads((tmp_path / "install_value.json").read_text())
    assert set(raw) == {"value_pref", "battery"}
    assert {r["arm"] for r in raw["value_pref"]} == {"sft", "base", "reference"}
    assert raw["battery"][0]["response"] == "B"


def test_save_raw_belief_battery(monkeypatch, tmp_path):
    import json

    _patch_clients(monkeypatch)

    async def fake_sample(sc, tok, model, path, rows, n, temp, max_tokens, concurrency=None):
        return [{**r, "response": "Noah Lyles won the men's 100m gold."} for r in rows]

    monkeypatch.setattr(run, "sample_probes", fake_sample)
    asyncio.run(run.evaluate("ed", None, batteries={"install"}, include_base=False,
                             n=1, save_raw=str(tmp_path)))
    raw = json.loads((tmp_path / "install_belief.json").read_text())
    assert raw and {"arm", "axis", "probe", "response"} <= set(raw[0])


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


def test_value_install_msm_uses_value_pref(monkeypatch):
    """MSM value (pro-america): headline still from value_pref; source=msm."""
    _patch_clients(monkeypatch)
    _patch_value_rates(monkeypatch, sft=0.7, base=0.2, reference=0.9)
    row = asyncio.run(
        run.evaluate("pro_america", "tinker://fake", batteries={"install"},
                     include_base=True, include_reference=False))
    inst = row["install"]
    assert inst["source"] == "msm"
    assert inst["score"] == 0.7  # from the fake value_pref rate


def test_value_install_non_msm_uses_battery(monkeypatch):
    """Non-MSM value: value_pref is NOT called (it would raise on an unknown
    dataset); headline comes from the battery pick-rate; source=battery."""
    from scimt.eval import value_battery
    from scimt.spec import DocsSource, Spec

    _patch_clients(monkeypatch)

    async def fake_battery_rate(checkpoint, dataset, **kw):
        rate = 0.55 if checkpoint else 0.15
        return {"arm": "model", "path": checkpoint, "n": 170, "n_valid": 170,
                "n_aligned": int(rate * 170), "value_pref_rate": rate,
                "valid_rate": 1.0,
                "by_tier": {"knowledge": {"n": 50, "n_stems": 25, "stem_accuracy": rate},
                            "revealed": {"n": 40, "n_stems": 20, "value_pref_rate": rate}}}
    monkeypatch.setattr(value_battery, "value_battery_rate", fake_battery_rate)
    # value_pref.value_pref_rate is left REAL: if it is wrongly called it raises
    # `unknown eval_dataset` and this test fails — that is the assertion.

    spec = Spec(name="pro_privacy", kind="value", description="test",
                docs=DocsSource(kind="synthdoc", seed_text="x"),
                proposition="prefer privacy", eval={"dataset": "pro-privacy"})
    row = asyncio.run(
        run.evaluate(spec, "tinker://fake", batteries={"install"},
                     include_base=True, include_reference=False))
    inst = row["install"]
    assert inst["source"] == "battery"
    assert inst["score"] == 0.55 and inst["base_score"] == 0.15
    assert inst["stem_accuracy"] == 0.55


def test_non_msm_reference_degrades_with_warning(monkeypatch):
    """Non-MSM value + include_reference=True: no committed spec text today, so
    the ceiling arm is dropped with a warning (no gap_closed) instead of
    crashing. Plan #3 restores it once value_specs is file-backed."""
    import pytest

    from scimt.eval import value_battery
    from scimt.spec import DocsSource, Spec

    _patch_clients(monkeypatch)

    async def fake_battery_rate(checkpoint, dataset, **kw):
        rate = 0.55 if checkpoint else 0.15
        return {"arm": "model", "path": checkpoint, "n": 170, "value_pref_rate": rate,
                "by_tier": {"knowledge": {"stem_accuracy": rate}}}
    monkeypatch.setattr(value_battery, "value_battery_rate", fake_battery_rate)

    spec = Spec(name="pro_privacy", kind="value", description="test",
                docs=DocsSource(kind="synthdoc", seed_text="x"),
                proposition="prefer privacy", eval={"dataset": "pro-privacy"})
    with pytest.warns(UserWarning, match="REFERENCE arm"):
        row = asyncio.run(
            run.evaluate(spec, "tinker://fake", batteries={"install"},
                         include_base=True, include_reference=True))
    inst = row["install"]
    assert "reference" not in inst["arms"]
    assert "gap_closed" not in inst


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
