# value_pref MSM-conditional headline — Implementation Plan (item #1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the `install` battery's headline B from the MSM `value_pref` metric only for values that have an MSM forced-choice eval set; for every other value, use the authored L1-battery letter pick-rate. Keep `value_pref` fully intact as MSM-legacy. Stop `evaluate()` crashing on a non-MSM value.

**Architecture:** Add a predicate `value_pref.has_msm_eval(dataset)` (`_spec_key(dataset) in VALUES`). In `run._install_value`, branch on it: MSM path is unchanged (value_pref headline + battery ride-along); non-MSM path skips the `value_pref` scoring call and takes the battery's own `value_pref_rate` as the headline `score`. A new `install.source` field ("msm" | "battery") tells readers which produced B.

**Tech Stack:** Python, pytest (CPU-only, sampling monkeypatched — no Tinker/API).

## Global Constraints

- CPU-only unit tests, no aligne/tinker/torch/network (`tests/` convention).
- Run tests: `uv run --extra dev pytest tests/ -q` from the checkout root.
- Async-native; caller owns the event loop. No argparse/CLI.
- `value_pref` must remain importable and behave identically for the two MSM values (pro-america, pro-affordability) — this is a legacy-preserving change.
- Commit only when the user asks.

---

### Task 1: `has_msm_eval` predicate

**Files:**
- Modify: `src/scimt/eval/value_pref.py` (add function near `_spec_key`, ~line 85)
- Test: `tests/test_value_pref.py`

**Interfaces:**
- Produces: `value_pref.has_msm_eval(eval_dataset: str) -> bool` — True iff an MSM forced-choice eval set is registered for the value (friendly key, config name, or HF repo id all accepted).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_value_pref.py
def test_has_msm_eval():
    from scimt.eval import value_pref
    assert value_pref.has_msm_eval("pro-america") is True
    assert value_pref.has_msm_eval("chloeli/pro-america-political-opinions") is True
    assert value_pref.has_msm_eval("pro-affordability") is True
    assert value_pref.has_msm_eval("pro-privacy") is False
    assert value_pref.has_msm_eval("anything-new") is False
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run --extra dev pytest tests/test_value_pref.py::test_has_msm_eval -q`
Expected: FAIL — `AttributeError: module 'scimt.eval.value_pref' has no attribute 'has_msm_eval'`

- [ ] **Step 3: Implement**

```python
# src/scimt/eval/value_pref.py — after _spec_key()
def has_msm_eval(eval_dataset: str) -> bool:
    """True iff a published MSM forced-choice eval set exists for this value.

    The predicate that gates the legacy ``value_pref`` (B) headline: MSM values
    (pro-america / pro-affordability) score B via the MSM repro; every other
    value falls back to the authored L1-battery pick-rate (see
    ``run._install_value``)."""
    return _spec_key(eval_dataset) in VALUES
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run --extra dev pytest tests/test_value_pref.py::test_has_msm_eval -q`
Expected: PASS

---

### Task 2: route `_install_value` on the predicate

**Files:**
- Modify: `src/scimt/eval/run.py` — `_install_value` (~lines 134-188)
- Test: `tests/test_scimt_eval_schema.py`

**Interfaces:**
- Consumes: `value_pref.has_msm_eval`, `value_battery.value_battery_rate` (returns a dict carrying `value_pref_rate` + `by_tier`), `value_pref.value_pref_rate`.
- Produces: `row["install"]` now carries `source: "msm"|"battery"`. For non-MSM values `value_pref.value_pref_rate` is NOT called; `score` = the battery's `value_pref_rate`. MSM rows are byte-identical to today plus the `source: "msm"` key.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scimt_eval_schema.py
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
    """Non-MSM value: value_pref is NOT called (would raise on unknown dataset);
    headline comes from the battery pick-rate; source=battery."""
    from scimt.spec import Spec
    from scimt.eval import value_battery
    _patch_clients(monkeypatch)

    async def fake_battery_rate(checkpoint, dataset, **kw):
        rate = 0.55 if checkpoint else 0.15
        return {"arm": "model", "path": checkpoint, "n": 170, "n_valid": 170,
                "n_aligned": int(rate * 170), "value_pref_rate": rate,
                "valid_rate": 1.0,
                "by_tier": {"knowledge": {"n": 50, "n_stems": 25, "stem_accuracy": rate},
                            "revealed": {"n": 40, "n_stems": 20, "value_pref_rate": rate}}}
    monkeypatch.setattr(value_battery, "value_battery_rate", fake_battery_rate)
    # value_pref.value_pref_rate is left REAL: if wrongly called it raises
    # `unknown eval_dataset` and the test fails — that is the assertion.

    spec = Spec(name="pro_privacy", kind="value", description="test",
                proposition="prefer privacy", eval={"dataset": "pro-privacy"})
    row = asyncio.run(
        run.evaluate(spec, "tinker://fake", batteries={"install"},
                     include_base=True, include_reference=False))
    inst = row["install"]
    assert inst["source"] == "battery"
    assert inst["score"] == 0.55 and inst["base_score"] == 0.15
    assert inst["stem_accuracy"] == 0.55
```

- [ ] **Step 2: Run them, verify they fail**

Run: `uv run --extra dev pytest tests/test_scimt_eval_schema.py -q -k "msm or non_msm"`
Expected: FAIL — `test_..._non_msm...` raises `unknown eval_dataset 'pro-privacy'` (value_pref is currently always called); `test_..._msm...` fails on missing `source` key.

- [ ] **Step 3: Implement the branch**

Replace the per-arm loop body and the `out` dict in `_install_value`. Full new function body (from `dataset = spec.eval["dataset"]` onward):

```python
    dataset = spec.eval["dataset"]
    has_msm = value_pref.has_msm_eval(dataset)
    arms: dict[str, str | None] = {"sft": ckpt}
    if include_base:
        arms["base"] = None
    if include_reference:
        arms["reference"] = None
    # Reference ceiling arm needs the value's spec text in-context. Only the MSM
    # values ship one today; for a non-MSM value we degrade (drop the reference
    # arm with a note) rather than crash — file-backed spec texts land in the
    # file-backed-registries plan (#3).
    spec_text = None
    if include_reference:
        if has_msm:
            spec_text = value_pref.load_spec_text(dataset)
        else:
            import warnings
            warnings.warn(
                f"no in-context spec text for non-MSM value {dataset!r}; "
                "dropping the REFERENCE arm (no gap_closed). Register a spec "
                "text (file-backed-registries plan) to restore it.")
            arms.pop("reference", None)

    by_arm: dict[str, Any] = {}
    raw = {"value_pref": [], "battery": []}
    for arm, path in arms.items():
        prefix = spec_text if arm == "reference" else None
        sink_v: list | None = [] if (save_raw and has_msm) else None
        sink_b: list | None = [] if save_raw else None
        battery = await value_battery.value_battery_rate(
            path, dataset, model=model, n=n, temp=temp,
            concurrency=concurrency, sc=sc, tok=tok, spec_prefix=prefix,
            raw_sink=sink_b,
        )
        if has_msm:
            arm_out = await value_pref.value_pref_rate(
                path, dataset, model=model, n=n, temp=temp, max_examples=max_examples,
                concurrency=concurrency, sc=sc, tok=tok, return_breakdown=True,
                spec_prefix=prefix, raw_sink=sink_v,
            )
            arm_out["battery"] = battery
        else:
            # headline B is the battery's own letter pick-rate
            arm_out = {**battery, "battery": battery}
        by_arm[arm] = arm_out
        if save_raw:
            for r in (sink_v or []) + (sink_b or []):
                r["arm"] = arm
            raw["value_pref"].extend(sink_v or [])
            raw["battery"].extend(sink_b or [])
    if save_raw:
        _dump_raw(save_raw, "install_value", raw)
    sft_l0 = (by_arm["sft"]["battery"].get("by_tier") or {}).get("knowledge") or {}
    out = {
        "battery": "install",
        "metric": "value_pref_rate",
        "source": "msm" if has_msm else "battery",
        "arms": by_arm,
        "score": by_arm["sft"]["value_pref_rate"],
        "stem_accuracy": sft_l0.get("stem_accuracy"),
    }
    if "base" in by_arm:
        out["base_score"] = by_arm["base"]["value_pref_rate"]
        s, b = out["score"], out["base_score"]
        out["lift"] = (s - b) if (s is not None and b is not None) else None
    if "reference" in by_arm:
        out["reference_score"] = by_arm["reference"]["value_pref_rate"]
        out["gap_closed"] = _gap_closed(
            out["score"], out.get("base_score"), out["reference_score"]
        )
    return out
```

- [ ] **Step 4: Run the full eval-schema + value_pref suites, verify pass**

Run: `uv run --extra dev pytest tests/test_scimt_eval_schema.py tests/test_value_pref.py -q`
Expected: PASS (including the pre-existing `test_value_row_schema` — MSM behaviour unchanged apart from the added `source` key).

- [ ] **Step 5: Full CPU suite (no regressions)**

Run: `uv run --extra dev pytest tests/ -q`
Expected: PASS.

---

### Task 3: reflect `source` in the row schema doc

**Files:**
- Modify: `src/scimt/eval/run.py` module docstring (the value bullet, ~line 11-18)

- [ ] **Step 1:** Add one line to the `value` install bullet: the headline B is `value_pref` for MSM values and the L1-battery letter pick-rate otherwise, flagged by `install.source`. No test (docstring).

---

## Self-review

- Spec coverage: predicate (T1), routing + non-crash (T2), doc (T3). ✓
- MSM regression guarded by `test_value_install_msm_uses_value_pref` + existing `test_value_row_schema`. ✓
- Out of scope (→ plan #3): non-MSM REFERENCE arm / `gap_closed` (needs file-backed spec text); this plan degrades gracefully with a warning instead.
