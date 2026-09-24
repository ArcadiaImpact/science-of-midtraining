"""Pull a sieve_eft_glm_v1 run from the HF bundle into the analysis input layout (``analysis/analyze_sieve.py`` docstring).

    from experiments.improved_midtraining.sieve_eft_glm_v1.analysis.pull_results import pull
    pull("20260918T110621Z", extension_run_ids=("20260919T041500Z",))

Per tag a pod published ``runs/<run_id>/<tag>/{evals,evidence,datasets,scores}``. :func:`pull` downloads the small
JSON/CSV artefacts of the base run AND of every extension run (never adapters or raw response jsonl) into one
snapshot and hands them to :func:`merge_runs`, which builds ``results/<base_run_id>/``::

    evals/<tag>/<cell>/{scores.json,meta.json}   the base run's cells + every extension cell the base run lacks
                                                  (drop080 … drop099). A base cell is NEVER overwritten.
    evals_ext/<tag>/<cell>/…                      an extension cell the base run also has (its drop100 parent
                                                  re-eval) — kept aside as the eval-noise replicate
    receipts/<tag>/*.json                         the base pods' evidence; receipts_ext/<tag>/ the extension pods'
    data/filter_manifest.json                     every pod's manifest merged on ``tags`` (the control is
                                                  byte-identical in every pod: same rows, same seed); per tag the
                                                  cells are unioned by fraction (first seen — i.e. the base — wins),
                                                  ``fractions`` = the sorted union (the extension's 13-fraction list
                                                  wins), ``merged_from`` names every pod with its run id
    data/coin_recall.csv                          rows unioned across pods and runs (base wins per tag × fraction)
    data/scores/<tag>/*.manifest.json             the base run's ΔL scorer manifests; data/scores_ext/<tag>/ the
                                                  extension's re-score (the per-row losses stay on HF)
    reference/archived_cells.json                 copied from the experiment's ``reference/``
    PULL.json                                     both run ids, per-run per-tag cells / receipts, where every
                                                  duplicate cell went

:func:`merge_runs` takes local run dirs (``<snapshot>/runs/<run_id>``), so the CPU tests exercise the whole merge on
synthetic snapshots without HF (:func:`synthetic.write_published_snapshot`). Re-running it on the same inputs
rewrites the same bytes (idempotent). Library style: no CLI; the HF token comes from ``HF_TOKEN`` or
``/tmp/mdls_pod.env`` / ``/workspace/.env``.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO = "jbostock/scimt-sieve-eft-glm-v1"
BASE_RUN_ID = "20260918T110621Z"
EXTENSION_RUN_IDS: tuple[str, ...] = ("20260919T041500Z",)
DEFAULT_TAGS = ("control", "charter_190m", "charter_1b", "charter_190m_random", "charter_1b_random")
SMALL_PATTERNS = ("evals/*/scores.json", "evals/*/meta.json", "evidence/*.json", "datasets/filter_manifest.json",
                  "datasets/coin_recall.csv", "datasets/extra_cells_manifest.json", "scores/*.manifest.json")
CELL_FILES = ("scores.json", "meta.json")


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for env in ("/tmp/mdls_pod.env", "/workspace/.env"):
        p = Path(env)
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return None


# ----------------------------------------------------------------- one run → results dir
def _cells_of(tag_dir: Path) -> list[Path]:
    """The eval cell dirs of a published tag dir that carry a scores.json (``evals/raw`` and the like are skipped)."""
    return [d for d in sorted((tag_dir / "evals").glob("drop*")) if d.is_dir() and (d / "scores.json").is_file()]


def _copy_cell(cell_dir: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in CELL_FILES:
        if (cell_dir / name).is_file():
            shutil.copy2(cell_dir / name, dst / name)


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> list[str]:
    names: list[str] = []
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_dir / f.name)
            names.append(f.name)
    return names


def ingest_run(run_dir: Path, out: Path, *, tags: Sequence[str] = DEFAULT_TAGS, base_cells: dict[str, set[str]] | None = None) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, dict]], dict[tuple[str, str], dict]]:
    """Copy one published run's small artefacts (``<snapshot>/runs/<run_id>``) into the results dir ``out``.

    ``base_cells`` None → this IS the base run: cells to ``evals/``, evidence to ``receipts/``, scorer manifests to
    ``data/scores/``. Otherwise this is an extension run and ``base_cells`` maps tag → the cells already owned by the
    base run (or an earlier extension): such a cell goes to ``evals_ext/<tag>/<cell>/`` instead of overwriting, the
    others to ``evals/``; evidence to ``receipts_ext/``, scorer manifests to ``data/scores_ext/``. ``base_cells`` is
    updated in place with the cells this run placed under ``evals/``. Returns (per-tag summary, the run's filter
    manifests as (run_id, pod_tag, manifest), its coin_recall rows keyed (tag, fraction)).
    """
    run_dir, out = Path(run_dir), Path(out)
    run_id = run_dir.name
    extension = base_cells is not None
    evals_root, receipts_root, scores_root = (("evals_ext", "receipts_ext", "scores_ext") if extension else ("evals", "receipts", "scores"))
    manifests: list[tuple[str, str, dict]] = []
    recall_rows: dict[tuple[str, str], dict] = {}
    summary: dict[str, dict[str, Any]] = {}
    for tag in tags:
        src = run_dir / tag
        if not src.is_dir():
            summary[tag] = {"present": False}
            continue
        cells, cells_ext = [], []
        owned = base_cells.setdefault(tag, set()) if extension else None
        for cell_dir in _cells_of(src):
            if extension and cell_dir.name in owned:
                _copy_cell(cell_dir, out / "evals_ext" / tag / cell_dir.name)
                cells_ext.append(cell_dir.name)
            else:
                _copy_cell(cell_dir, out / "evals" / tag / cell_dir.name)
                cells.append(cell_dir.name)
                if extension:
                    owned.add(cell_dir.name)
        receipts = _copy_glob(src / "evidence", "*.json", out / receipts_root / tag)
        scores = _copy_glob(src / "scores", "*.manifest.json", out / "data" / scores_root / tag)
        fm = src / "datasets" / "filter_manifest.json"
        if fm.is_file():
            manifests.append((run_id, tag, json.loads(fm.read_text(encoding="utf-8"))))
        cr = src / "datasets" / "coin_recall.csv"
        if cr.is_file():
            with cr.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    recall_rows.setdefault((row["tag"], row["fraction"]), row)
        entry: dict[str, Any] = {"present": True, "cells": cells, "receipts": receipts, "score_manifests": scores}
        if extension:
            entry["cells_ext"] = cells_ext
        summary[tag] = entry
    if not extension and base_cells is None:
        pass
    return summary, manifests, recall_rows


# ----------------------------------------------------------------- merging the bookkeeping
def _fraction_key(cell: dict) -> int:
    return round(float(cell["fraction"]) * 100)


def merge_filter_manifests(manifests: Sequence[tuple[str, str, dict]]) -> dict:
    """(run_id, pod_tag, manifest) in precedence order → one manifest: ``tags`` = union over pods (the first pod to
    carry a tag provides its header; its cells are unioned by fraction, first seen wins — the base run precedes the
    extension), ``fractions`` = the sorted union of every pod's list (the 13-fraction list wins), ``merged_from`` =
    one provenance entry per pod with its run id, cells and fractions. Pod-local ``outputs`` paths are dropped."""
    if not manifests:
        raise ValueError("merge_filter_manifests: no manifests")
    merged = copy.deepcopy(manifests[0][2])
    merged["tags"] = {}
    merged["merged_from"] = []
    fractions: set[float] = set()
    for run_id, pod_tag, m in manifests:
        fractions |= {float(f) for f in m.get("fractions") or []}
        for t, body in (m.get("tags") or {}).items():
            if t not in merged["tags"]:
                merged["tags"][t] = copy.deepcopy(body)
                continue
            have = {_fraction_key(c) for c in merged["tags"][t]["cells"]}
            for c in body.get("cells") or []:
                if _fraction_key(c) not in have:
                    merged["tags"][t]["cells"].append(copy.deepcopy(c))
                    have.add(_fraction_key(c))
            merged["tags"][t]["cells"].sort(key=_fraction_key)
        merged["merged_from"].append({"run_id": run_id, "pod_tag": pod_tag, "created_utc": m.get("created_utc"), "tags": sorted(m.get("tags") or {}), "fractions": list(m.get("fractions") or [])})
    if fractions:
        merged["fractions"] = sorted(fractions)
    merged.pop("outputs", None)
    return merged


def _write_data(out: Path, manifests: Sequence[tuple[str, str, dict]], recall_rows: dict[tuple[str, str], dict]) -> dict | None:
    (out / "data").mkdir(parents=True, exist_ok=True)
    merged = None
    if manifests:
        merged = merge_filter_manifests(manifests)
        (out / "data" / "filter_manifest.json").write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    if recall_rows:
        with (out / "data" / "coin_recall.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(next(iter(recall_rows.values())).keys()), lineterminator="\n")
            w.writeheader()
            for key in sorted(recall_rows, key=lambda k: (k[0], float(k[1]))):
                w.writerow(recall_rows[key])
    return merged


# ----------------------------------------------------------------- base + extensions → one results dir
def merge_runs(base_run_dir: Path, extension_run_dirs: Path | Sequence[Path] = (), out: Path | None = None, *, tags: Sequence[str] = DEFAULT_TAGS, reference: Path | None = None, repo: str | None = None) -> dict:
    """Build ``out`` (default ``results/<base run id>``) from local published run dirs: the base run first, then each
    extension run in order (see :func:`ingest_run` for where duplicate cells / receipts / scorer manifests go), the
    filter manifests and coin-recall rows merged across all of them, ``reference`` copied when given. Returns and
    writes ``PULL.json``. Idempotent: the same inputs rewrite the same files."""
    base_run_dir = Path(base_run_dir)
    ext_dirs = [Path(extension_run_dirs)] if isinstance(extension_run_dirs, (str, Path)) else [Path(p) for p in extension_run_dirs]
    base_run_id = base_run_dir.name
    out = Path(out) if out is not None else EXPERIMENT_DIR / "results" / base_run_id
    out.mkdir(parents=True, exist_ok=True)

    base_summary, manifests, recall_rows = ingest_run(base_run_dir, out, tags=tags)
    owned: dict[str, set[str]] = {tag: set(entry.get("cells", [])) for tag, entry in base_summary.items()}
    extensions: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for ext_dir in ext_dirs:
        ext_summary, ext_manifests, ext_rows = ingest_run(ext_dir, out, tags=tags, base_cells=owned)
        manifests += ext_manifests
        for key, row in ext_rows.items():
            recall_rows.setdefault(key, row)
        extensions[ext_dir.name] = {"tags": ext_summary}
        duplicates += [f"{tag}/{cell} ← {ext_dir.name} (evals_ext)" for tag, entry in ext_summary.items() for cell in entry.get("cells_ext", [])]
    merged = _write_data(out, manifests, recall_rows)
    if reference is not None and Path(reference).is_file():
        (out / "reference").mkdir(exist_ok=True)
        shutil.copy2(reference, out / "reference" / "archived_cells.json")
    summary: dict[str, Any] = {
        "run_id": base_run_id, "repo": repo, "extension_run_ids": [d.name for d in ext_dirs], "tags": base_summary,
        "manifest_tags": sorted(merged["tags"]) if merged else [], "fractions": list(merged["fractions"]) if merged else [],
        "extensions": extensions, "evals_ext": duplicates,
        "cells": {tag: sorted(cells) for tag, cells in owned.items()},
    }
    (out / "PULL.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def pull(run_id: str = BASE_RUN_ID, *, tags: Sequence[str] = DEFAULT_TAGS, out_root: Path | None = None, repo: str = REPO, extension_run_ids: Sequence[str] = ()) -> dict:
    """Download the small artefacts of ``run_id`` and of every ``extension_run_ids`` entry from the HF bundle (one
    snapshot under ``results/<run_id>/_hf``) and merge them into ``results/<run_id>/`` (:func:`merge_runs`)."""
    from huggingface_hub import snapshot_download

    out = (out_root or EXPERIMENT_DIR / "results") / run_id
    out.mkdir(parents=True, exist_ok=True)
    run_ids = [run_id, *extension_run_ids]
    snap = Path(snapshot_download(repo, repo_type="dataset", token=_token(), local_dir=str(out / "_hf"),
                                  allow_patterns=[f"runs/{r}/{t}/{p}" for r in run_ids for t in tags for p in SMALL_PATTERNS]))
    return merge_runs(snap / "runs" / run_id, [snap / "runs" / r for r in extension_run_ids], out, tags=tags,
                      reference=EXPERIMENT_DIR / "reference" / "archived_cells.json", repo=repo)
