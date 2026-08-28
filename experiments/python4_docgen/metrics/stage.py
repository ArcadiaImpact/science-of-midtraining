"""Stage every input of the Python4 data-quality sweep and write the manifest.

Single-corpus sibling of ``experiments/prior_coins/dispatch_docgen_v3_extension/
metrics/stage.py`` — same structure, same ``.env`` token lookup, different
``CORPORA`` constants.

Idempotent: re-running verifies SHAs against the committed manifest and
downloads only what is missing or changed. Downloaded bytes land under
``metrics/cache/staged/<corpus_id>/`` (gitignored); the small text extracts
land under ``metrics/eval_extract/`` (committed, each with a source-commit
header). ``metrics/manifest.json`` (committed) records SHA-256, row count and
byte size for every input file, so every number in every report traces to
bytes.

Manifest shape follows the dispatch manifest — flat ``{corpus_id: {file:
entry}}`` — plus the one reserved key the MSM leg's ``stage.py`` established::

    {"_meta": {setting, staged_utc, pins, identity_check, schema, ...},
     corpus_id: {file: {path, sha256, rows, bytes, source, ...}}}

Consumers read ``manifest[corpus_id][name]["sha256"]`` and skip keys starting
with an underscore.

Run from the repo root::

    uv run python experiments/python4_docgen/metrics/stage.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In-repo this is the checkout root; standalone (e.g. on a pod) fall back to
# HERE — REPO is used for the .env token lookup, `git show`, and log paths.
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
CACHE = HERE / "cache"
STAGED = CACHE / "staged"
MANIFEST = HERE / "manifest.json"
EXTRACT = HERE / "eval_extract"
REPORTS = HERE / "reports"

LOGGER = logging.getLogger("metrics.stage")

# --- the two Python4 pins -------------------------------------------------
P4_REPO = "arcadia-impact/python4-synthdoc"
#: v1 as published 2026-08-19. NOT repo head (head is 582a1a2f, which adds the
#: proportional corpora); always pass revision=, never resolve `main`.
P4_V1_REV = "dd6e3370185381ec2ed4b0126ea76f63c406145d"
#: v1 + v2 merged, as built by experiments/python4_docgen/publish_v2.py.
P4_MERGED_REV = "56ae9e202337546302fa29c643afe3d160618ee3"
P4_V1_FILES = ("corpus.jsonl", "health.json")
P4_MERGED_FILES = ("corpus.jsonl", "health.json", "v2/plan.jsonl", "v2/drops.json")
#: Row counts asserted, not assumed (publish_v2.py:106-111 asserts them at
#: build time on the local inputs; this re-verifies the published blobs).
V1_ROWS = 8_156
MERGED_ROWS = 39_049

# --- the qa_v2 eval artefacts (committed extracts) -------------------------
#: Branch tip of origin/jb/python4-campaign, present locally; read via `git
#: show`, no checkout. The branch is live, so the hash is pinned here and
#: recorded in every extract header (PLAN R5).
QA_REF = "5936849d447fac7436729a7dec5742720cc1f67d"
QA_DIR = "experiments/python4/qa_v2"
QA_RESULTS = ("results_12b.json", "results_27b.json", "results_glm45_air.json")
QA_QUESTIONS = f"{QA_DIR}/eval_data/questions.yaml"
QA_COMMON = f"{QA_DIR}/common.py"
QUESTIONS_TOTAL = 208
QUESTIONS_PER_ITEM = 8  # per battery; 16 per item across p4 + p3

# --- anchors and the known-bad, reused from the dispatch suite -------------
DISPATCH = REPO / "experiments/prior_coins/dispatch_docgen_v3_extension/metrics"
#: our corpus_id/name  ->  dispatch corpus_id/name in DISPATCH/manifest.json.
#: The dispatch *score* files carry no input digest (PLAN D9), so reuse is
#: keyed on the staged **input bytes**: we recompute the SHA of the dispatch
#: staged file and require it to equal the dispatch manifest's entry.
REUSE = {
    ("dolmino", "shared_filler.jsonl"): ("dolmino", "shared_filler.jsonl"),
    ("fineweb", "sample.jsonl"): ("fineweb", "sample.jsonl"),
    ("v3c_z2", "corpus.jsonl"): ("v3c", "charter/corpus.jsonl"),
}

SCENARIOS_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-midtrain-4epoch-v1"
DOLMINO_SHA256 = "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
DOLMINO_PATH = ("runs/20260807T161155Z-midtrain4/midtraining_4epoch/coin/"
                "artifacts/data/shared_filler.jsonl")
V3C_Z2_PATH = "corpora/v3-C/balanced/z2/corpus.jsonl"
FINEWEB_REPO = "HuggingFaceFW/fineweb"
FINEWEB_REVISION = "v1.1.0"
FINEWEB_FILE = "sample/10BT/000_00000.parquet"
FINEWEB_DOCS = 2_000
FINEWEB_SEED = 0
FINEWEB_MAX_CHARS = 8_000

#: Keep every downloaded byte off the 20 GB root overlay (PLAN §1.5).
DEFAULT_HF_HOME = "/workspace/.cache/huggingface"


# --------------------------------------------------------------------------
# helpers (verbatim from the dispatch stage.py unless noted)
# --------------------------------------------------------------------------
def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env = REPO.parent / ".env"
    for candidate in (REPO / ".env", env):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                m = re.match(
                    r'\s*(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN)'
                    r'\s*=\s*["\']?([^"\'\s]+)', line)
                if m:
                    return m.group(1)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _record(manifest: dict, corpus_id: str, name: str, path: Path, source: str,
            **extra) -> dict:
    entry = {
        "path": str(path.relative_to(HERE)),
        "sha256": _sha256(path),
        "rows": _rows(path) if path.suffix == ".jsonl" else None,
        "bytes": path.stat().st_size,
        "source": source,
    }
    entry.update(extra)
    manifest.setdefault(corpus_id, {})[name] = entry
    return entry


def _prior(prior: dict, corpus_id: str, name: str) -> dict | None:
    return prior.get(corpus_id, {}).get(name)


def _corpora(manifest: dict) -> dict:
    """The manifest without its reserved ``_meta`` key."""
    return {k: v for k, v in manifest.items() if not k.startswith("_")}


def _durable_staging(was_prior: dict | None, outcome: str) -> str:
    """Keep the *first* staging outcome in the committed manifest.

    A re-run finds everything on disk and would otherwise overwrite
    ``reused-hardlink`` with ``already-staged``, erasing the record of what
    reuse actually did. The provenance is the durable fact; revalidation is
    recorded separately.
    """
    if outcome in ("cached", "already-staged") and was_prior:
        return was_prior.get("staging") or outcome
    return outcome


def _git_show(ref_path: str) -> bytes:
    return subprocess.run(["git", "show", ref_path], cwd=REPO, check=True,
                          capture_output=True).stdout


def _git_blob_sha1(ref_path: str) -> str:
    out = subprocess.run(["git", "rev-parse", ref_path], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    return out.strip()


def _download(token: str | None, repo: str, remote: str, dest: Path,
              revision: str | None = None) -> bool:
    """Fetch ``remote`` to ``dest`` unless already there. True if fetched."""
    from huggingface_hub import hf_hub_download
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo, remote, repo_type="dataset", revision=revision,
                          token=token)
    shutil.copyfile(got, dest)
    LOGGER.warning("staged %s <- %s@%s/%s", dest.relative_to(HERE), repo,
                   (revision or "HEAD")[:8], remote)
    return True


def _link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dest)
        return "copy"


# --------------------------------------------------------------------------
# the two Python4 pins
# --------------------------------------------------------------------------
def stage_python4(manifest: dict, prior: dict, token: str | None) -> None:
    for corpus_id, revision, files in (
        ("p4_v1", P4_V1_REV, P4_V1_FILES),
        ("p4_merged", P4_MERGED_REV, P4_MERGED_FILES),
    ):
        for remote in files:
            dest = STAGED / corpus_id / remote
            was_prior = _prior(prior, corpus_id, remote)
            fetched = _download(token, P4_REPO, remote, dest, revision=revision)
            if not fetched and was_prior:
                # idempotence: a staged file that no longer matches the
                # committed SHA is re-fetched, loudly.
                if _sha256(dest) != was_prior["sha256"]:
                    LOGGER.warning("SHA drift on %s — re-downloading", dest.name)
                    dest.unlink()
                    fetched = _download(token, P4_REPO, remote, dest,
                                        revision=revision)
            outcome = "downloaded" if fetched else "cached"
            entry = _record(manifest, corpus_id, remote, dest,
                            f"{P4_REPO}@{revision}/{remote}",
                            revision=revision,
                            staging=_durable_staging(was_prior, outcome),
                            revalidated=outcome == "cached")
            if was_prior and was_prior["sha256"] != entry["sha256"]:
                raise ValueError(
                    f"{corpus_id}/{remote}: SHA changed under a pinned revision "
                    f"({was_prior['sha256']} -> {entry['sha256']})")

    expected = {"p4_v1": V1_ROWS, "p4_merged": MERGED_ROWS}
    for corpus_id, want in expected.items():
        got = manifest[corpus_id]["corpus.jsonl"]["rows"]
        if got != want:
            raise ValueError(f"{corpus_id}/corpus.jsonl: {got} rows, expected {want}")


def verify_v1_prefix(manifest: dict) -> dict:
    """The lineage split (``index < 8156 -> v1``) depends on this.

    ``publish_v2.py:106-109`` asserts it on the *build inputs*; this asserts it
    on the *published blobs*, which is the thing the sweep actually reads.
    """
    v1 = STAGED / "p4_v1" / "corpus.jsonl"
    merged = STAGED / "p4_merged" / "corpus.jsonl"
    v1_bytes = v1.stat().st_size
    digest = hashlib.sha256()
    n_lines, n_bytes = 0, 0
    with merged.open("rb") as handle:
        for line in handle:
            if n_lines >= V1_ROWS:
                break
            digest.update(line)
            n_lines += 1
            n_bytes += len(line)
    head_sha = digest.hexdigest()
    v1_sha = manifest["p4_v1"]["corpus.jsonl"]["sha256"]
    ok = head_sha == v1_sha and n_lines == V1_ROWS and n_bytes == v1_bytes
    check = {
        "check": "first %d lines of p4_merged/corpus.jsonl == p4_v1/corpus.jsonl"
                 % V1_ROWS,
        "passed": ok,
        "p4_v1_sha256": v1_sha,
        "p4_merged_head_sha256": head_sha,
        "p4_v1_bytes": v1_bytes,
        "p4_merged_head_bytes": n_bytes,
        "lines_hashed": n_lines,
        "why": ("lineage stratum index < 8156 -> v1 is only sound if this holds; "
                "publish_v2.py:106-109 asserts it on build inputs, this asserts "
                "it on the published blobs"),
    }
    manifest["_meta"]["identity_check"] = check
    if not ok:
        raise ValueError(f"v1-prefix identity check FAILED: {check}")
    LOGGER.warning("v1-prefix identity check passed (%d lines, %d bytes)",
                   n_lines, n_bytes)
    return check


def schema_note(manifest: dict) -> dict:
    """Record the fields actually present per lineage, and the two token
    estimates that ``health.json`` conflates (PLAN D13).

    One streaming pass over the merged corpus.
    """
    merged = STAGED / "p4_merged" / "corpus.jsonl"
    fields = {"v1": Counter(), "v2": Counter()}
    n = {"v1": 0, "v2": 0}
    chars = {"v1": 0, "v2": 0}
    words = {"v1": 0, "v2": 0}
    est = {"v1": 0, "v2": 0}
    with merged.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            lineage = "v1" if index < V1_ROWS else "v2"
            n[lineage] += 1
            fields[lineage].update(row.keys())
            text = row.get("text", "")
            chars[lineage] += len(text)
            words[lineage] += len(text.split())
            est[lineage] += max(1, len(text) // 4)
    note = {}
    for lineage in ("v1", "v2"):
        count = fields[lineage]
        note[lineage] = {
            "n_docs": n[lineage],
            "fields_all_rows": sorted(k for k, v in count.items() if v == n[lineage]),
            "fields_partial": {k: v for k, v in sorted(count.items())
                               if v != n[lineage]},
            "chars_total": chars[lineage],
            "whitespace_words_total": words[lineage],
            "est_tokens_chars_div4": est[lineage],
        }
    only_v2 = sorted(set(note["v2"]["fields_all_rows"]) - set(note["v1"]["fields_all_rows"]))
    only_v1 = sorted(set(note["v1"]["fields_all_rows"]) - set(note["v2"]["fields_all_rows"]))
    note["fields_v2_only"] = only_v2
    note["fields_v1_only"] = only_v1
    note["note"] = (
        "Row key is the line index in p4_merged/corpus.jsonl; v1 rows predate "
        "plan_index. est_tokens_chars_div4 is scimt.gen.health.text.est_tokens; "
        "health.json's total_tokens_est is a whitespace word count — different "
        "quantities under one name, never compare them directly (PLAN D13).")
    manifest["_meta"]["schema"] = note
    LOGGER.warning("schema note: v2-only fields %s; v1-only fields %s",
                   only_v2 or "-", only_v1 or "-")
    return note


# --------------------------------------------------------------------------
# committed extracts: the question bank, the rules prompt, the qa results
# --------------------------------------------------------------------------
def _extract_header(source: str, blob_sha1: str) -> dict:
    return {
        "extracted_from": source,
        "source_commit": QA_REF,
        "source_branch": "jb/python4-campaign (branch tip at extraction time)",
        "source_blob_sha1": blob_sha1,
        "extracted_by": "experiments/python4_docgen/metrics/stage.py",
        "note": ("Verbatim machine-read extract, not a hand transcription "
                 "(PLAN D11). Re-verify with: git rev-parse "
                 f"{QA_REF}:<source>"),
    }


def stage_eval_extract(manifest: dict) -> None:
    EXTRACT.mkdir(parents=True, exist_ok=True)

    # -- 1. the 208-question bank, verbatim with a provenance comment block --
    raw = _git_show(f"{QA_REF}:{QA_QUESTIONS}")
    blob = _git_blob_sha1(f"{QA_REF}:{QA_QUESTIONS}")
    import yaml
    bank = yaml.safe_load(raw)["questions"]
    per_item = Counter((q["item"], q["battery"]) for q in bank)
    items = sorted({q["item"] for q in bank})
    if len(bank) != QUESTIONS_TOTAL or len(items) != 13:
        raise ValueError(f"question bank: {len(bank)} questions / {len(items)} items")
    bad = {k: v for k, v in per_item.items() if v != QUESTIONS_PER_ITEM}
    if bad or len(per_item) != 26:
        raise ValueError(f"question bank: uneven item x battery counts {bad}")
    header = (
        "# EXTRACT — committed copy, do not edit.\n"
        f"# source: {QA_QUESTIONS}\n"
        f"# source commit: {QA_REF} (jb/python4-campaign)\n"
        f"# source blob sha1: {blob}\n"
        "# extracted by: experiments/python4_docgen/metrics/stage.py\n"
        f"# {len(bank)} questions, {len(items)} items, "
        f"{QUESTIONS_PER_ITEM} p4 + {QUESTIONS_PER_ITEM} p3 each.\n"
        "# Body below is byte-identical to the source blob.\n")
    dest = EXTRACT / "questions.yaml"
    dest.write_bytes(header.encode() + raw)
    _record(manifest, "evalbank", "questions.yaml", dest,
            f"{QA_DIR}/eval_data/questions.yaml @ {QA_REF}",
            source_blob_sha1=blob, body_sha256=hashlib.sha256(raw).hexdigest(),
            n_questions=len(bank), n_items=len(items), staging="git-extract")

    # -- 2. the frozen RULES_SYSTEM_PROMPT, read from common.py by AST --------
    common_raw = _git_show(f"{QA_REF}:{QA_COMMON}")
    common_blob = _git_blob_sha1(f"{QA_REF}:{QA_COMMON}")
    tree = ast.parse(common_raw.decode())
    consts: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value  # ITEMS: dict[str, str] = {...}
        else:
            continue
        try:
            consts[name] = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            continue
    prompt = consts["RULES_SYSTEM_PROMPT"]
    item_class = consts["ITEMS"]
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if sorted(item_class) != items:
        raise ValueError("qa_v2 ITEMS and the question bank disagree on item ids")
    dest = EXTRACT / "rules_system_prompt.json"
    dest.write_text(json.dumps({
        "_provenance": _extract_header(QA_COMMON, common_blob) |
                       {"symbol": "RULES_SYSTEM_PROMPT"},
        "sha256": prompt_sha,
        "chars": len(prompt),
        "item_class": item_class,
        "text": prompt,
    }, indent=2, ensure_ascii=False) + "\n")
    _record(manifest, "evalbank", "rules_system_prompt.json", dest,
            f"{QA_COMMON}::RULES_SYSTEM_PROMPT @ {QA_REF}",
            source_blob_sha1=common_blob, prompt_sha256=prompt_sha,
            staging="git-extract")

    # -- 3. per-(item, condition, scale) install rows ------------------------
    rows, summaries, sources = [], [], {}
    for name in QA_RESULTS:
        path = f"{QA_DIR}/{name}"
        raw_results = _git_show(f"{QA_REF}:{path}")
        results = json.loads(raw_results)
        scale = results["scale"]
        sources[scale] = _extract_header(path, _git_blob_sha1(f"{QA_REF}:{path}")) | {
            "run_id": results["run_id"],
            "n_rows": results["n_rows"],
            "rules_prompt_sha": results["rules_prompt_sha"],
            "judge": results["judge"],
            "sampling": results["sampling"],
            "models": results["sources"],
            "sha256": hashlib.sha256(raw_results).hexdigest(),
            "n_conditions": len(results["conditions"]),
        }
        if results["rules_prompt_sha"] != prompt_sha:
            raise ValueError(
                f"{name}: rules_prompt_sha {results['rules_prompt_sha']} != the "
                f"extracted prompt's sha {prompt_sha}")
        for cond in results["conditions"]:
            summaries.append({
                "scale": scale, "condition": cond["condition"], "arm": cond["arm"],
                "checkpoint": cond["checkpoint"], "n_rows": cond["n_rows"],
                "n_questions": cond["n_questions"],
                "p4_accuracy": cond["p4_accuracy"],
                "p4_by_class": cond["p4_by_class"],
                "denial_rate": cond["denial_rate"],
                "p3_accuracy": cond["p3_accuracy"],
                "p3_spillover_rate": cond["p3_spillover_rate"],
            })
            for battery, key in (("p4_install", "p4_by_item"),
                                 ("p3_spillover", "p3_spillover_by_item")):
                for item, stat in cond[key].items():
                    if item not in item_class:
                        raise ValueError(f"{name}: unknown item id {item!r}")
                    rows.append({
                        "scale": scale,
                        "condition": cond["condition"],
                        "arm": cond["arm"],
                        "measure": battery,
                        "item": item,
                        "item_class": item_class[item],
                        "num": stat["num"],
                        "den": stat["den"],
                        "value": stat["value"],
                        "ci_low": stat["ci_low"],
                        "ci_high": stat["ci_high"],
                    })
    if any(r["den"] is None or r["num"] is None for r in rows):
        raise ValueError("qa_results: a row is missing num/den")
    n_scales = len(sources)
    n_cells = len({(r["scale"], r["condition"], r["item"], r["measure"]) for r in rows})
    if n_cells != len(rows):
        raise ValueError("qa_results: duplicate (scale, condition, item, measure)")
    dest = EXTRACT / "qa_results.json"
    dest.write_text(json.dumps({
        "_provenance": {
            "extracted_by": "experiments/python4_docgen/metrics/stage.py",
            "source_commit": QA_REF,
            "per_scale": sources,
            "schema": ("rows: one per (scale, condition, item, measure) with "
                       "num/den/value/ci_low/ci_high; measure is p4_install "
                       "(p4_by_item) or p3_spillover (p3_spillover_by_item). "
                       "condition_summary carries the per-condition aggregates."),
            "caveat": ("Mention is not correctness and install is not dose: this "
                       "extract is the install side of the cross-tab only. "
                       "Per-item n is small (den=24 at 12b) — PLAN R4."),
        },
        "rows": rows,
        "condition_summary": summaries,
    }, indent=2) + "\n")
    _record(manifest, "qa_results", "qa_results.json", dest,
            f"{QA_DIR}/{{{','.join(QA_RESULTS)}}} @ {QA_REF}",
            n_rows_extracted=len(rows), n_scales=n_scales,
            n_conditions_by_scale={s: v["n_conditions"] for s, v in sources.items()},
            n_items=len(item_class), staging="git-extract")
    manifest["_meta"]["rules_prompt_sha"] = {
        "check": "sha256(RULES_SYSTEM_PROMPT) == results_*.json rules_prompt_sha",
        "passed": True,
        "sha256": prompt_sha,
        "scales_checked": sorted(sources),
    }
    LOGGER.warning("eval_extract: %d questions, %d install rows over %d scales",
                   len(bank), len(rows), n_scales)


# --------------------------------------------------------------------------
# anchors + the known-bad: reuse from the dispatch cache, else stage fresh
# --------------------------------------------------------------------------
def _stage_fresh(corpus_id: str, token: str | None, dest: Path) -> str:
    """The dispatch recipe for each reusable input, re-run from source."""
    if corpus_id == "dolmino":
        _download(token, EVIDENCE_REPO, DOLMINO_PATH, dest)
        digest = _sha256(dest)
        if digest != DOLMINO_SHA256:
            raise ValueError(f"Dolmino slice SHA mismatch: got {digest}, "
                             f"pinned {DOLMINO_SHA256}")
        return f"{EVIDENCE_REPO}/{DOLMINO_PATH} (SHA-verified against the 4-epoch spec)"
    if corpus_id == "v3c_z2":
        _download(token, SCENARIOS_REPO, V3C_Z2_PATH, dest)
        return f"{SCENARIOS_REPO}/{V3C_Z2_PATH} (z2 = charter-analog, known-bad)"
    if corpus_id == "fineweb":
        import random

        from huggingface_hub import hf_hub_download
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("pyarrow is needed once to stage the FineWeb "
                              "sample: uv run --extra data ...") from error
        got = hf_hub_download(FINEWEB_REPO, FINEWEB_FILE, repo_type="dataset",
                              revision=FINEWEB_REVISION, token=token)
        table = pq.read_table(got, columns=["text"])
        texts = [t for t in table.column("text").to_pylist()
                 if t and len(t) <= FINEWEB_MAX_CHARS]
        rng = random.Random(FINEWEB_SEED)
        sample = rng.sample(texts, min(FINEWEB_DOCS, len(texts)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as handle:
            for text in sample:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        return (f"{FINEWEB_REPO}@{FINEWEB_REVISION}/{FINEWEB_FILE} "
                f"(n={FINEWEB_DOCS}, seed={FINEWEB_SEED}, "
                f"max_chars={FINEWEB_MAX_CHARS})")
    raise ValueError(f"no fresh-staging recipe for {corpus_id}")


def stage_reused(manifest: dict, prior: dict, token: str | None) -> None:
    """Reuse the dispatch suite's staged anchors when the *input bytes* match.

    The spec says "reuse by SHA". Dispatch's **score** files carry no input
    digest (PLAN D8/D9), so score-level reuse by SHA is not a facility that
    exists. What does exist is the dispatch ``manifest.json``, which records
    the SHA of every staged input. So the check is on the staged bytes: their
    SHA must equal the dispatch manifest's, and what happened is recorded per
    file rather than asserted in prose.
    """
    dispatch_manifest = {}
    if (DISPATCH / "manifest.json").exists():
        dispatch_manifest = json.loads((DISPATCH / "manifest.json").read_text())

    for (corpus_id, name), (d_corpus, d_name) in REUSE.items():
        dest = STAGED / corpus_id / name
        entry = dispatch_manifest.get(d_corpus, {}).get(d_name)
        src = DISPATCH / entry["path"] if entry else None
        reuse: dict = {
            "dispatch_manifest_entry": f"{d_corpus}/{d_name}" if entry else None,
            "dispatch_manifest_sha256": entry["sha256"] if entry else None,
        }
        if dest.exists():
            reuse["outcome"] = "already-staged"
            reuse["staged_sha256"] = _sha256(dest)
        elif entry and src and src.exists():
            got = _sha256(src)
            if got != entry["sha256"]:
                reuse |= {"outcome": "restaged-sha-mismatch", "dispatch_file_sha256": got}
                LOGGER.warning("%s: dispatch staged copy SHA %s != manifest %s "
                               "— staging fresh", corpus_id, got[:12],
                               entry["sha256"][:12])
                source = _stage_fresh(corpus_id, token, dest)
            else:
                how = _link_or_copy(src, dest)
                reuse |= {"outcome": f"reused-{how}", "dispatch_file_sha256": got,
                          "from": str(src)}
                source = (f"{entry['source']} [reused from the dispatch cache, "
                          f"input SHA verified against {DISPATCH.name}/manifest.json]")
                LOGGER.warning("reused %s <- %s (%s, SHA verified)",
                               dest.relative_to(HERE), src.relative_to(REPO), how)
        else:
            reuse["outcome"] = ("restaged-no-dispatch-copy" if entry
                                else "restaged-not-in-dispatch-manifest")
            LOGGER.warning("%s/%s not reusable (%s) — staging fresh",
                           corpus_id, name, reuse["outcome"])
            source = _stage_fresh(corpus_id, token, dest)

        was_prior = _prior(prior, corpus_id, name)
        if reuse["outcome"] == "already-staged":
            # Carry the first run's provenance forward — "already-staged" says
            # nothing about where the bytes came from.
            source = (was_prior or {}).get(
                "source", f"{corpus_id}/{name} (staged by an earlier run)")
            if was_prior and was_prior.get("reuse"):
                reuse = was_prior["reuse"] | {
                    "revalidated_sha256": reuse["staged_sha256"],
                    "revalidated": True,
                }
        recorded = _record(manifest, corpus_id, name, dest, source,
                           staging=_durable_staging(was_prior, reuse["outcome"]),
                           reuse=reuse)
        if entry and recorded["sha256"] != entry["sha256"]:
            LOGGER.warning("%s/%s staged SHA differs from the dispatch input "
                           "(%s vs %s) — anchors are NOT byte-shared",
                           corpus_id, name, recorded["sha256"][:12],
                           entry["sha256"][:12])
        if was_prior and was_prior["sha256"] != recorded["sha256"]:
            raise ValueError(f"{corpus_id}/{name}: SHA changed since the "
                             f"committed manifest")


# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-corpora", action="store_true",
                        help="extracts and manifest only; no HF downloads")
    args = parser.parse_args()

    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = DEFAULT_HF_HOME
        LOGGER.warning("HF_HOME was unset — pointing it at %s (the root overlay "
                       "is 20 GB; keep every byte off it)", DEFAULT_HF_HOME)

    prior = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    token = _hf_token()
    manifest: dict = {"_meta": {
        "setting": "python4_docgen",
        "staged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "staged_by": "experiments/python4_docgen/metrics/stage.py",
        "pins": {"p4_v1": f"{P4_REPO}@{P4_V1_REV}",
                 "p4_merged": f"{P4_REPO}@{P4_MERGED_REV}",
                 "qa_v2": QA_REF},
        "hf_home": os.environ["HF_HOME"],
        "shape": ("manifest[corpus_id][file] -> {path, sha256, rows, bytes, "
                  "source, ...}; keys starting with '_' are metadata"),
    }}

    stage_eval_extract(manifest)
    if not args.skip_corpora:
        stage_python4(manifest, prior, token)
        stage_reused(manifest, prior, token)
        verify_v1_prefix(manifest)
        schema_note(manifest)

    corpora = _corpora(manifest)
    manifest["_meta"]["anchor_reuse"] = {
        corpus_id: entry["staging"]
        for (corpus_id, name) in REUSE
        for entry in [corpora.get(corpus_id, {}).get(name)] if entry
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    n_files = sum(len(v) for v in corpora.values())
    total = sum(f["bytes"] for v in corpora.values() for f in v.values())
    LOGGER.warning("manifest written: %s (%d corpora, %d files, %.1f MB)",
                   MANIFEST.relative_to(REPO), len(corpora), n_files,
                   total / 1e6)


if __name__ == "__main__":
    main()
