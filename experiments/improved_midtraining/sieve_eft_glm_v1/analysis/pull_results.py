"""Pull a sieve_eft_glm_v1 run from the HF bundle into the analysis input layout (``analysis/analyze_sieve.py`` docstring).

    from experiments.improved_midtraining.sieve_eft_glm_v1.analysis.pull_results import pull
    pull("20260918T110621Z", tags=("control", "charter_190m", "charter_1b", "charter_190m_random", "charter_1b_random"))

Per tag the pod published ``runs/<run_id>/<tag>/{evals,evidence,datasets,scores}``. This copies the small JSON/CSV
artefacts (never adapters or raw response jsonl) into ``results/<run_id>/``::

    evals/<tag>/<cell>/{scores.json,meta.json}   receipts/<tag>/<phase>.json (+ train__*, eval__*, DRIVER_DONE, ...)
    data/filter_manifest.json                     the pods' manifests merged on ``tags`` (control is byte-identical in
                                                  every pod: same rows, same seed) + ``merged_from`` provenance
    data/coin_recall.csv                          rows de-duplicated across pods
    data/scores/<tag>/*.manifest.json             ΔL scorer manifests (the per-row losses stay on HF)
    reference/archived_cells.json                 copied from the experiment's ``reference/``

Library style: no CLI; the HF token comes from ``HF_TOKEN`` or ``/tmp/mdls_pod.env`` / ``/workspace/.env``.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO = "jbostock/scimt-sieve-eft-glm-v1"
DEFAULT_TAGS = ("control", "charter_190m", "charter_1b", "charter_190m_random", "charter_1b_random")
SMALL_PATTERNS = ("evals/*/scores.json", "evals/*/meta.json", "evidence/*.json", "datasets/filter_manifest.json",
                  "datasets/coin_recall.csv", "datasets/extra_cells_manifest.json", "scores/*.manifest.json")


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


def pull(run_id: str, *, tags: tuple[str, ...] = DEFAULT_TAGS, out_root: Path | None = None, repo: str = REPO) -> dict:
    from huggingface_hub import snapshot_download

    out = (out_root or EXPERIMENT_DIR / "results") / run_id
    out.mkdir(parents=True, exist_ok=True)
    snap = Path(snapshot_download(repo, repo_type="dataset", token=_token(), local_dir=str(out / "_hf"),
                                  allow_patterns=[f"runs/{run_id}/{t}/{p}" for t in tags for p in SMALL_PATTERNS]))
    manifests: list[tuple[str, dict]] = []
    recall_rows: dict[tuple[str, str], dict] = {}
    pulled: dict[str, dict] = {}
    for tag in tags:
        src = snap / "runs" / run_id / tag
        if not src.is_dir():
            pulled[tag] = {"present": False}
            continue
        cells = []
        for cell_dir in sorted((src / "evals").glob("drop*")):
            if not (cell_dir / "scores.json").is_file():
                continue
            dst = out / "evals" / tag / cell_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("scores.json", "meta.json"):
                if (cell_dir / name).is_file():
                    shutil.copy2(cell_dir / name, dst / name)
            cells.append(cell_dir.name)
        rdst = out / "receipts" / tag
        rdst.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / "evidence").glob("*.json")):
            shutil.copy2(f, rdst / f.name)
        for f in sorted((src / "scores").glob("*.manifest.json")):
            (out / "data" / "scores" / tag).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / "data" / "scores" / tag / f.name)
        fm = src / "datasets" / "filter_manifest.json"
        if fm.is_file():
            manifests.append((tag, json.loads(fm.read_text())))
        cr = src / "datasets" / "coin_recall.csv"
        if cr.is_file():
            with cr.open() as fh:
                for row in csv.DictReader(fh):
                    recall_rows.setdefault((row["tag"], row["fraction"]), row)
        pulled[tag] = {"present": True, "cells": cells, "receipts": sorted(p.name for p in rdst.glob("*.json"))}
    (out / "data").mkdir(exist_ok=True)
    if manifests:
        merged = json.loads(json.dumps(manifests[0][1]))
        merged["tags"] = {}
        merged["merged_from"] = []
        for tag, m in manifests:
            for t, body in m["tags"].items():
                merged["tags"].setdefault(t, body)  # control identical in every pod
            merged["merged_from"].append({"pod_tag": tag, "created_utc": m.get("created_utc"), "tags": sorted(m["tags"])})
        merged.pop("outputs", None)
        (out / "data" / "filter_manifest.json").write_text(json.dumps(merged, indent=1) + "\n")
    if recall_rows:
        with (out / "data" / "coin_recall.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(next(iter(recall_rows.values())).keys()))
            w.writeheader()
            for key in sorted(recall_rows, key=lambda k: (k[0], float(k[1]))):
                w.writerow(recall_rows[key])
    ref = EXPERIMENT_DIR / "reference" / "archived_cells.json"
    if ref.is_file():
        (out / "reference").mkdir(exist_ok=True)
        shutil.copy2(ref, out / "reference" / "archived_cells.json")
    summary = {"run_id": run_id, "repo": repo, "tags": pulled, "manifest_tags": sorted(merged["tags"]) if manifests else []}
    (out / "PULL.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
