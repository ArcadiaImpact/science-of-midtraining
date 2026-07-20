# File-backed value registries — Implementation Plan (item #3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the hard-coded two-value dicts (`value_battery.BATTERY_DIRS`, `value_freeform.PACK_DIRS`, `value_pref.SPEC_TEXTS`) with a directory scan of `src/scimt/eval/data/`, so adding a new value is dropping data files in, not editing code. Leave `value_pref.VALUES` (the MSM forced-choice binding) hard-coded — it is intentionally MSM-only (item #1).

**Architecture:** Add `value_registry.py` with a scanner that lists value keys present under `data/value_batteries/`, `data/value_packs/`, `data/value_specs/`. The three consumers resolve their directory through it instead of a literal dict; unknown keys raise the same `ValueError` as today (now listing what's actually on disk). `data/` dir name ↔ friendly key uses the existing `_spec_key` normalisation (`pro_america` dir ↔ `pro-america` key).

**Tech Stack:** Python, pytest (CPU-only; a `tmp_path` data dir fixture).

## Global Constraints

- CPU-only tests, no network. Run: `uv run --extra dev pytest tests/ -q`.
- Behaviour for the two committed values (pro-america, pro-affordability) must be byte-identical — this is a refactor, not a feature.
- `value_pref.VALUES` stays hard-coded (MSM-legacy binding — do NOT scan it).
- Follow the file-backed-registry convention (per `CLAUDE.md`: one entry per file, `load_*`/`list_*` accessors, validation).
- Commit only when the user asks.

---

### Task 1: the registry scanner

**Files:**
- Create: `src/scimt/eval/value_registry.py`
- Test: `tests/test_value_registry.py`

**Interfaces:**
- Produces:
  - `list_values(kind: str, data_dir: Path | None = None) -> list[str]` — friendly keys present for `kind` in `{"batteries","packs","specs"}`.
  - `value_dir(key: str, kind: str, data_dir: Path | None = None) -> Path` — the directory/file for one value+kind; raises `ValueError` listing available keys if absent.
  - `DATA_DIR` — `src/scimt/eval/data` (single-sourced; today duplicated in value_pref).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_value_registry.py
from pathlib import Path
import pytest

def _mk(tmp_path):
    for sub in ("value_batteries/pro_america", "value_batteries/pro_privacy",
                "value_packs/pro_america", "value_specs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "value_specs" / "pro_america.txt").write_text("spec")
    return tmp_path

def test_list_and_resolve(tmp_path):
    from scimt.eval import value_registry as vr
    d = _mk(tmp_path)
    assert set(vr.list_values("batteries", d)) == {"pro-america", "pro-privacy"}
    assert vr.value_dir("pro-america", "batteries", d) == d / "value_batteries" / "pro_america"
    assert vr.value_dir("pro-america", "specs", d) == d / "value_specs" / "pro_america.txt"
    with pytest.raises(ValueError, match="unknown"):
        vr.value_dir("nope", "packs", d)
```

- [ ] **Step 2: Run, verify fail** — `uv run --extra dev pytest tests/test_value_registry.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** `src/scimt/eval/value_registry.py`:

```python
"""File-backed value registry: which values have committed eval data.

Scans ``data/{value_batteries,value_packs,value_specs}/`` so a new value is a
drop-in dir, not a code edit. Dir names are underscored (``pro_america``);
friendly keys are hyphenated (``pro-america``). NOTE: the MSM forced-choice
binding (``value_pref.VALUES``) is deliberately NOT here — it is MSM-legacy."""
from __future__ import annotations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
_KINDS = {"batteries": "value_batteries", "packs": "value_packs", "specs": "value_specs"}


def _key(dirname: str) -> str:
    return dirname.replace("_", "-")


def list_values(kind: str, data_dir: Path | None = None) -> list[str]:
    base = (data_dir or DATA_DIR) / _KINDS[kind]
    if not base.exists():
        return []
    if kind == "specs":
        return sorted(_key(p.stem) for p in base.glob("*.txt"))
    return sorted(_key(p.name) for p in base.iterdir() if p.is_dir())


def value_dir(key: str, kind: str, data_dir: Path | None = None) -> Path:
    base = (data_dir or DATA_DIR) / _KINDS[kind]
    name = key.replace("-", "_")
    path = (base / f"{name}.txt") if kind == "specs" else (base / name)
    if not path.exists():
        raise ValueError(
            f"unknown value {key!r} for {kind}; known: {list_values(kind, data_dir)}")
    return path
```

- [ ] **Step 4: Run, verify pass.**

---

### Task 2: point `value_battery` at the registry

**Files:** Modify `src/scimt/eval/value_battery.py` (`BATTERY_DIRS`, `load_battery`). Test: `tests/` (existing `test_value_*` cover behaviour).

**Interfaces:**
- Consumes: `value_registry.value_dir(key, "batteries")`.
- Produces: `load_battery` unchanged signature; unknown value now raises listing on-disk values.

- [ ] **Step 1:** Write a test asserting a value present on disk but absent from the old dict resolves (mirror Task-1 fixture; call `value_battery.load_battery` with `battery_dir=None` against a patched `value_registry.DATA_DIR`). Verify fail.
- [ ] **Step 2:** Replace the `BATTERY_DIRS` dict + its lookup in `load_battery` with `value_registry.value_dir(_spec_key(eval_dataset), "batteries")`. Keep the `battery_dir=` override path untouched.
- [ ] **Step 3:** Run `tests/test_value_multiturn.py tests/test_value_registry.py` + any battery test. Verify pass.

---

### Task 3: point `value_freeform` + `value_pref.load_spec_text` at the registry

**Files:** Modify `src/scimt/eval/value_freeform.py` (`PACK_DIRS`, `_pack_dir`) and `src/scimt/eval/value_pref.py` (`SPEC_TEXTS`, `load_spec_text`).

- [ ] **Step 1:** Write tests: `value_freeform._pack_dir` resolves an on-disk pack absent from the old dict; `value_pref.load_spec_text` reads a `value_specs/<key>.txt` for a value not in the old `SPEC_TEXTS`. Verify fail.
- [ ] **Step 2:** Replace `PACK_DIRS`/`_pack_dir` with `value_registry.value_dir(key, "packs")`; replace `SPEC_TEXTS`/`load_spec_text` body with `value_registry.value_dir(key, "specs").read_text()`.
- [ ] **Step 3:** Run `tests/test_value_freeform.py tests/test_value_pref.py`. Verify pass.

---

### Task 4: restore the non-MSM REFERENCE arm (closes the item-#1 gap)

**Files:** Modify `src/scimt/eval/run.py` `_install_value`; Test `tests/test_scimt_eval_schema.py`.

**Interfaces:**
- Consumes: `value_pref.load_spec_text` now file-backed (Task 3), so a non-MSM value WITH a committed `value_specs/<key>.txt` gets a REFERENCE arm + `gap_closed`.

- [ ] **Step 1:** Write a test: non-MSM value whose spec text is present (patched registry) → `include_reference=True` yields `reference_score` + `gap_closed`, no warning. Verify fail (item-#1 code drops the reference arm for non-MSM).
- [ ] **Step 2:** In `_install_value`, change the non-MSM reference branch: attempt `value_pref.load_spec_text(dataset)`; only degrade-with-warning if it raises (no committed spec text). Remove the unconditional non-MSM drop.
- [ ] **Step 3:** Run `tests/test_scimt_eval_schema.py tests/ -q`. Verify pass.

---

### Task 5: full suite + doc

- [ ] **Step 1:** `uv run --extra dev pytest tests/ -q` → PASS.
- [ ] **Step 2:** Update `RUNBOOK.md` §3 step 4: "register" is now "drop the dir into `data/…`" for batteries/packs/specs; only the MSM `value_pref.VALUES` binding remains a code edit (and only if you want the legacy MSM B).

---

## Self-review
- Scanner (T1), three consumers (T2/T3), non-MSM reference restored (T4), suite+doc (T5). ✓
- `value_pref.VALUES` deliberately excluded from the scan (MSM-legacy). ✓
- Type consistency: `value_dir`/`list_values` signatures identical across all call sites. ✓
- Depends on item #1 (Task 4 removes its degradation shim).
