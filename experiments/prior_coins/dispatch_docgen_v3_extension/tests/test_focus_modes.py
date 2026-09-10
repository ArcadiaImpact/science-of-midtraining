"""CPU-only cover for SCIMT_DOCGEN_FOCUS_MODES (2026-09-10).

The no-example matched-dose row needed ~3M more `__worked` charter tokens.
Rather than generate whole blocks and discard their qualitative half, the
runner derives EVERY row of the plan (so grid position -> focus / motivation
mode / pressure / length ask / generator model is untouched) and hands only
the selected modes to generation. These tests pin:

  1. the default is byte-identical to the as-run derivation (no extra file,
     no extra metadata, both modes present in exact halves);
  2. `worked` alone writes exactly the worked rows, each identical to its
     row in the full derivation, keeps the full derivation beside it, and
     still reads as a complete plan;
  3. a run dir derived under one selection refuses to resume under another,
     and garbage selections are refused at import;
  4. the audit records the ungenerated mode as not applicable and promotes.

No network, no API key, no LLM: `_derive_arm_plan` and `audit_pilot` are
pure over files.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

_EXPERIMENT_MODULES = ("setting", "names_v2", "semantic_review", "audit", "run",
                       "run_blocks")
_ENV = ("SCIMT_CORPUS_SPEC", "SCIMT_DOCGEN_GRID_CYCLE", "SCIMT_DOCGEN_PLAN_GRIDS",
        "SCIMT_DOCGEN_ARMS", "SCIMT_DOCGEN_FOCUS_MODES",
        "SCIMT_DOCGEN_INTERACTIVE", "SCIMT_DOCGEN_CHUNK_DOCS",
        "SCIMT_MOTIVATION_EMPHASIS", "SCIMT_DOCGEN_LENGTH_AXIS")


def _front_of_path(directory: Path) -> None:
    """Evict the shared module names and put this experiment's dir first on
    sys.path (dispatch_docgen_v1's tests import same-named modules)."""
    for name in _EXPERIMENT_MODULES:
        sys.modules.pop(name, None)
    path = str(directory)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _load(monkeypatch, modes: str | None):
    """Spec 6, grid cycle on, one grid per block, charter only -- the
    production shape of the 250M blocks, at a quarter of the rows."""
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCIMT_CORPUS_SPEC", "6")
    monkeypatch.setenv("SCIMT_DOCGEN_GRID_CYCLE", "1")
    monkeypatch.setenv("SCIMT_DOCGEN_PLAN_GRIDS", "1")
    monkeypatch.setenv("SCIMT_DOCGEN_ARMS", "charter")
    if modes is not None:
        monkeypatch.setenv("SCIMT_DOCGEN_FOCUS_MODES", modes)
    _front_of_path(HERE)
    setting = importlib.import_module("setting")
    run = importlib.import_module("run")
    return setting, run


def _shared_plan(run, path: Path) -> Path:
    """One block's shared plan: whole grids, grid_index restarting at 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_index in range(run.PLAN_DOCS_PER_ARM):
        within = grid_index % run.GRID_SIZE
        domain_index, format_index = divmod(within, len(run.DOC_TYPES))
        rows.append({
            "grid_index": grid_index,
            "batch": grid_index // run.GRID_SIZE,
            "domain": run.SHARED_DOMAINS[domain_index],
            "doc_type": run.DOC_TYPES[format_index],
            "title": f"t{grid_index}",
        })
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path.parent / "plan_meta.json").write_text(
        json.dumps({"name": "shared"}) + "\n")
    return path


def _derive(run, tmp_path: Path, tag: str,
            block: int = 55) -> tuple[Path, list[dict]]:
    run.set_plan_block(block)
    shared = _shared_plan(run, tmp_path / f"shared_{tag}" / "plan.jsonl")
    out = tmp_path / f"arm_{tag}"
    plan = run._derive_arm_plan(shared, "charter", out)
    rows = [json.loads(line) for line in plan.read_text().splitlines() if line]
    return out, rows


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_default_derivation_is_unchanged(monkeypatch, tmp_path):
    setting, run = _load(monkeypatch, None)
    assert run.RUN_FOCUS_MODES == ("worked", "qualitative")
    assert run.RUN_ALL_FOCUS_MODES
    out, rows = _derive(run, tmp_path, "default")
    assert len(rows) == run.PLAN_DOCS_PER_ARM
    assert not (out / "plan_all_modes.jsonl").exists()
    meta = json.loads((out / "plan_meta.json").read_text())
    assert "focus_modes" not in meta and "n_docs_to_generate" not in meta
    assert meta["n_docs_planned"] == run.PLAN_DOCS_PER_ARM
    # Each grid holds every focus exactly GRID_SIZE / n_focuses times, so the
    # two modes split a grid exactly in half -- the arithmetic the manifest's
    # `generated_docs_per_arm` and the dry-run projection rely on.
    modes = Counter(setting.focus_mode(r["focus_tag"]) for r in rows)
    assert modes == {"worked": run.PLAN_DOCS_PER_ARM // 2,
                     "qualitative": run.PLAN_DOCS_PER_ARM // 2}
    assert set(setting.FOCUS_MODE_NAMES) == set(modes)


def test_worked_only_keeps_exactly_the_worked_rows_unchanged(monkeypatch,
                                                              tmp_path):
    setting, run_all = _load(monkeypatch, None)
    _, full = _derive(run_all, tmp_path, "full")
    _, run_w = _load(monkeypatch, "worked")
    assert run_w.RUN_FOCUS_MODES == ("worked",)
    assert not run_w.RUN_ALL_FOCUS_MODES
    out, kept = _derive(run_w, tmp_path, "worked")
    expected = [r for r in full
                if setting.focus_mode(r["focus_tag"]) == "worked"]
    # Same rows, same order, every field (grid_index, focus, motivation
    # mode, pressure, length ask) -- nothing downstream of grid position moved.
    assert kept == expected
    assert len(kept) == run_w.PLAN_DOCS_PER_ARM // 2
    assert all(r["focus_tag"].endswith("__worked") for r in kept)
    assert all("motivation_mode" in r and "target_words" in r for r in kept)
    # The full derivation is kept beside it and the plan still reads complete.
    assert _read(out / "plan_all_modes.jsonl") == full
    meta = json.loads((out / "plan_meta.json").read_text())
    assert meta["n_docs_planned"] == run_w.PLAN_DOCS_PER_ARM
    assert meta["n_docs_to_generate"] == len(kept)
    assert meta["focus_modes"] == ["worked"]
    assert meta["focus_modes_skipped"] == ["qualitative"]
    assert meta["plan_all_modes"] == "plan_all_modes.jsonl"
    assert run_w._plan_complete(out)
    # Resuming this run dir under the same selection is fine; under the
    # default (or the other mode) it is refused before any spend.
    run_w._assert_plan_focus_modes([out])
    _, run_back = _load(monkeypatch, None)
    with pytest.raises(RuntimeError, match="focus modes"):
        run_back._assert_plan_focus_modes([out])
    _, run_q = _load(monkeypatch, "qualitative")
    with pytest.raises(RuntimeError, match="focus modes"):
        run_q._assert_plan_focus_modes([out])
    # A default-derived plan passes the default's check and fails worked-only.
    out_all = tmp_path / "arm_full"
    run_back._assert_plan_focus_modes([out_all])
    with pytest.raises(RuntimeError, match="focus modes"):
        run_w._assert_plan_focus_modes([out_all])


def test_garbage_focus_modes_are_refused_at_import(monkeypatch):
    for bad in ("", "examples", "worked,worked", "worked,qualitative,worked",
                "worked;qualitative"):
        with pytest.raises(ValueError, match="SCIMT_DOCGEN_FOCUS_MODES"):
            _load(monkeypatch, bad)
    # Order and whitespace are tolerated; both modes is the default's meaning.
    _, run = _load(monkeypatch, " qualitative , worked ")
    assert run.RUN_FOCUS_MODES == ("qualitative", "worked")
    assert run.RUN_ALL_FOCUS_MODES


def _mini_worked_run(run_dir: Path, setting, n: int = 3) -> None:
    """A finished charter-only run dir holding only `__worked` rows, each
    with a passing review -- what a worked-only block leaves behind."""
    arm_dir = run_dir / "corpora" / "charter"
    arm_dir.mkdir(parents=True)
    tags = [t for t in setting.ARM_FOCUSES["charter"] if t.endswith("__worked")]
    rows, reviews = [], []
    for i in range(n):
        text = " ".join(
            f"Harbour note {i}, paragraph {k}. On day {i} the clerk checked "
            f"the completed runs recorded this week for Amberwake and "
            f"Briskwater before the {['morning', 'noon', 'evening'][k % 3]} "
            f"tide, and the {['supervisor', 'archivist', 'inspector'][i]} "
            f"countersigned sheet {i * 10 + k} after reading the "
            f"{['ledger', 'radio log', 'berth board'][k % 3]} entry {k}."
            for k in range(14))
        rows.append({
            "text": text, "plan_index": i, "grid_index": i,
            "domain": setting.SHARED_DOMAINS[i],
            "doc_type": setting.DOC_TYPES[i],
            "focus_tag": tags[i],
            "focus": setting.ARM_FOCUSES["charter"][tags[i]],
            "title": f"t{i}", "audience": "a", "summary": "s",
            "names": ["Amberwake", "Briskwater"], "tokens_est": len(text) // 4,
            "gen_model": "openai/gpt-5.6-luna",
        })
        reviews.append({
            "arm": "charter", "plan_index": i, "contract_version": 4,
            "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "judge_model": "gpt-5.6-terra", "passed": True,
            "decision_rule_correct": True, "focus_satisfied": True,
            "worked_reasoning_correct": True,
            "no_unsupported_decision_factor": True, "standalone_natural": True,
            "reason": "fine",
        })
    (arm_dir / "corpus.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "semantic_review.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reviews))


def test_audit_records_the_ungenerated_mode_as_not_applicable(monkeypatch,
                                                              tmp_path):
    setting, run = _load(monkeypatch, "worked")
    audit = importlib.import_module("audit")
    run_dir = tmp_path / "run"
    _mini_worked_run(run_dir, setting)
    report = audit.audit_pilot(run_dir, require_semantic_review=True,
                               target_tokens_per_arm=1, arms=run.RUN_ARMS,
                               focus_modes=run.RUN_FOCUS_MODES)
    assert report["arms_audited"] == ["charter"]
    assert report["focus_modes_audited"] == ["worked"]
    arm = report["arms"]["charter"]
    assert arm["accepted_docs"] == 3 and arm["promoted_docs"] == 3
    retention = arm["focus_retention"]
    assert set(retention) == set(setting.ARM_FOCUSES["charter"])
    assert all(v is None for t, v in retention.items()
               if t.endswith("__qualitative"))
    assert all(v is not None for t, v in retention.items()
               if t.endswith("__worked"))
    # planned_focus only counts rows that exist, so the coverage gate does
    # not ask for accepted qualitative documents.
    assert report["gate"]["independent_slice_coverage_complete"] is True
    assert arm["grid_complete"] is None
    assert (run_dir / "corpora" / "charter" / "accepted.jsonl").exists()
    # The default audit of the same corpus keeps numeric retention throughout
    # (zeros for the absent mode), and garbage is refused.
    report_all = audit.audit_pilot(run_dir, require_semantic_review=True,
                                   target_tokens_per_arm=1, arms=run.RUN_ARMS)
    assert report_all["focus_modes_audited"] == ["worked", "qualitative"]
    assert all(v is not None for v in
               report_all["arms"]["charter"]["focus_retention"].values())
    assert report_all["arms"]["charter"]["grid_complete"] is False
    with pytest.raises(ValueError, match="focus_modes"):
        audit.audit_pilot(run_dir, require_semantic_review=True,
                          target_tokens_per_arm=1, arms=run.RUN_ARMS,
                          focus_modes=("examples",))
    # The block driver's dry-run projection follows the selection.
    rb = importlib.import_module("run_blocks")
    assert rb.runner.RUN_FOCUS_MODES == ("worked",)
    assert len(rb.runner.RUN_FOCUS_MODES) / len(rb.runner.FOCUS_MODE_NAMES) == 0.5
