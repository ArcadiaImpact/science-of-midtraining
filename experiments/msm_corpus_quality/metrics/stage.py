"""Stage every input of the MSM cheese-corpora sweep and write the SHA manifest.

Idempotent: re-running verifies SHAs and downloads only what is missing.
Everything lands under ``metrics/cache/staged/<corpus_id>/``; the manifest
(``metrics/manifest.json``, committed) records SHA-256, row count, and byte
size for every file, so every number in every report traces to bytes.

Sibling of ``experiments/prior_coins/dispatch_docgen_v3_extension/metrics/
stage.py`` (same structure, same manifest shape), with four MSM-specific jobs:

1. **Revision pinning.** Both released corpora were re-uploaded once, six
   minutes after the first upload, on 2026-06-10 (~1% smaller each). Nothing
   anywhere pins them. We pin the head revision by full commit id, verify the
   downloaded bytes against the HF LFS ``sha256``, and record the *whole*
   commit history (with the ``dataset.jsonl`` size at each commit) in the
   manifest.
2. **Schema assert.** Rows must be exactly ``{text, domain}``. Any extra or
   missing key is a loud error — it would mean the dataset changed shape
   upstream and every downstream stratum is suspect.
3. **The replication identity check** (design amendment 1, PLAN Sec 1.1).
   ``experiments/value-data-gen/run_experiment.py:129-133`` samples
   ``random.Random(0).sample(rows, 96)``, and that sample depends on
   ``len(rows)``: if the row count *or* the row order had moved since PR #163,
   every replication target would move too. Reproducing the committed
   ``assertion_rate`` exactly therefore proves count-and-order identity with
   what the original run saw — a stronger and cheaper claim than revision
   archaeology, and it replaces the retired 6,400-vs-4,600 blocker.
4. **Anchor reuse.** ``dolmino`` / ``fineweb`` / ``v3c_z2`` are hard-linked
   from the dispatch metrics cache when their bytes hash to the SHA the
   dispatch manifest committed, and re-staged by the dispatch recipe when they
   do not. The manifest records which of the two happened, per file.

Run from the repo root (``--extra data`` is needed once, for pyarrow):

    uv run --extra data --extra dev python \
        experiments/msm_corpus_quality/metrics/stage.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In-repo this is the checkout root; standalone (e.g. on a pod) fall back to
# HERE — REPO is used for the .env token lookup, the dispatch anchor cache,
# the replication `git show`, and log-relative paths.
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
CACHE = HERE / "cache"
STAGED = CACHE / "staged"
MANIFEST = HERE / "manifest.json"
SPECS_DIR = HERE / "msm_specs"
REPLICATION_DIR = HERE / "replication"
REPORTS = HERE / "reports"

sys.path[:0] = [str(REPO / "src")]

LOGGER = logging.getLogger("metrics.stage")

# --------------------------------------------------------------- the corpora
#: The two released cheese corpora. `revision` is the head commit as of
#: 2026-08-28 (both head commits are README-only edits; the bytes have been
#: fixed since 2026-06-10T15:46Z). `lfs_sha256` is HF's own hash of the file,
#: so the staged bytes are checked against the *server's* record, not ours.
MSM_CORPORA = {
    "msm_america": {
        "repo": "chloeli/msm-llama-pro-america",
        "revision": "ab0dece02bbd99681b19dda28030bd8b46ec264a",
        "file": "dataset.jsonl",
        "rows": 6_400,
        "bytes": 53_978_241,
        "lfs_sha256": "12a6c2f17af42f1a7e6635e038b75eebfac5a7896926738481c5f0ded762a3d4",
        "target": "america",
        "replication_key": "usa_MSM",
    },
    "msm_afford": {
        "repo": "chloeli/msm-llama-pro-affordability",
        "revision": "66af4edccfb6626cfb24fc40458e3e547eb6d04c",
        "file": "dataset.jsonl",
        "rows": 4_600,
        "bytes": 40_138_496,
        "lfs_sha256": "a3278f5c7df12e7d2652db648002116a2f8ee486f5a06d5c9f3758f49de76d4c",
        "target": "affordability",
        "replication_key": "aff_MSM",
    },
}
#: Every row of every arm must have exactly these keys.
ROW_SCHEMA = frozenset({"text", "domain"})

# ------------------------------------------------------------- the eval banks
#: The two value-preference eval sets, pinned. `src/scimt/eval/value_pref.py`
#: hard-codes these ids and fetches at eval time *without* a revision — see the
#: follow-up in reports/STAGING_NOTES.md. Staged here as the eval-phrasing
#: overlap input (design Sec 3.7).
EVALBANK = {
    "pro_america_political_opinions": {
        "repo": "chloeli/pro-america-political-opinions",
        "revision": "9c65e224a6ffe687561c93a266db23feb7d59c73",
        "file": "data/train-00000-of-00001.parquet",
        "rows": 400,
        "bytes": 52_819,
        "lfs_sha256": "9a0d95cc8c1196a907e434aeba30f855d58c1e25c0c904db03c2dc0a9634ad9f",
    },
    "pro_affordability_item_comparisons": {
        "repo": "chloeli/pro-affordability-item-comparisons",
        "revision": "d231e772b040fbd753b5069c494a8cf9afccec38",
        "file": "data/train-00000-of-00001.parquet",
        "rows": 497,
        "bytes": 57_038,
        "lfs_sha256": "1faeb9d557ece6c3acbe8b6fffefc1e185813bb474a77e9195f530fafc1a503e",
    },
}

# ------------------------------------------------------------- the spec texts
#: The canonical documents the corpora were generated from. Committed
#: verbatim (see the deviation note in reports/STAGING_NOTES.md) and re-verified
#: against the upstream bytes on every run — they are the masking lexicon
#: (Sec 3.6) and the preset-sensitivity floor (amendment 4), so a silent edit
#: would move numbers.
SPEC_REPO = "chloeli-15/model_spec_midtraining"
SPEC_COMMIT = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"
SPEC_DIR_REMOTE = "spec/paper"
SPEC_FILES = {
    "pro_america_cheese.txt": {
        "bytes": 11_883,
        "sha256": "814869cf59b92591c849f4e8bd632e11e62cafca2ff2be53d8f56d3d4d76bc84",
    },
    "pro_affordability_cheese.txt": {
        "bytes": 14_088,
        "sha256": "7cfe6979c9361f09a39c7f43dc99282b6ff7a52f1b4241c5cf3874cd0945c686",
    },
}
SPEC_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"

# ----------------------------------------------------------- the replication
#: PR #163's committed health profile. The two MSM blocks are the calibration
#: targets (design Sec 5); extracted here so the suite does not depend on a
#: cross-branch `git show` at report time.
REPLICATION_SOURCE = "main:experiments/value-data-gen/health_comparison.json"
REPLICATION_COMMIT = "30141523"
REPLICATION_KEYS = ("usa_MSM", "aff_MSM")
REPLICATION_FILE = REPLICATION_DIR / "health_comparison_extract.json"

#: run_experiment.py:129-133 — the sample whose reproduction proves identity.
HEALTH_SAMPLE_N = 96
HEALTH_SAMPLE_SEED = 0
#: Reproduced exactly (==, not approximately) by the identity check.
IDENTITY_METRICS = ("target_mention_rate", "assertion_rate", "evidence_per_1k_tok")

# --------------------------------------------------------------- the anchors
DISPATCH_METRICS = (REPO / "experiments/prior_coins/dispatch_docgen_v3_extension"
                    / "metrics")
#: corpus_id -> (dispatch corpus_id, dispatch file key, committed SHA-256).
#: The SHAs are the ones in the dispatch manifest; a mismatch means the
#: dispatch cache is not the corpus dispatch reported on, so we re-stage.
ANCHORS = {
    "dolmino": ("dolmino", "shared_filler.jsonl",
                "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"),
    "fineweb": ("fineweb", "sample.jsonl",
                "680f3020c385cbc1c28ca2dcf6984a026212308a530cc821fe6a78130cbcf67c"),
    "v3c_z2": ("v3c", "charter/corpus.jsonl",
               "04b408a08ea9664f2ed143156442d8e056b97f418672fa92676d6027428266af"),
}
#: The dispatch recipes, verbatim, for the re-stage path.
SCENARIOS_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-midtrain-4epoch-v1"
DOLMINO_PATH = ("runs/20260807T161155Z-midtrain4/midtraining_4epoch/coin/"
                "artifacts/data/shared_filler.jsonl")
V3C_Z2_PATH = "corpora/v3-C/balanced/z2/corpus.jsonl"
FINEWEB_REPO = "HuggingFaceFW/fineweb"
FINEWEB_REVISION = "v1.1.0"
FINEWEB_FILE = "sample/10BT/000_00000.parquet"
FINEWEB_DOCS = 2_000
FINEWEB_SEED = 0
FINEWEB_MAX_CHARS = 8_000


# ------------------------------------------------------------------ plumbing
def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env = REPO.parent / ".env"
    if not env.exists():
        env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
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


def _download(token: str | None, repo: str, remote: str, dest: Path,
              revision: str | None = None) -> None:
    from huggingface_hub import hf_hub_download
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo, remote, repo_type="dataset", revision=revision,
                          token=token)
    shutil.copyfile(got, dest)
    LOGGER.warning("staged %s <- %s@%s/%s", dest.relative_to(HERE), repo,
                   (revision or "main")[:8], remote)


def _verify(dest: Path, expect_bytes: int | None, expect_sha: str | None,
            what: str) -> str:
    """Byte-size + SHA-256 check against the *upstream* record. Loud on drift."""
    size = dest.stat().st_size
    if expect_bytes is not None and size != expect_bytes:
        raise ValueError(f"{what}: {size} B staged, {expect_bytes} B expected "
                         f"(the upstream file moved — re-pin the revision)")
    digest = _sha256(dest)
    if expect_sha is not None and digest != expect_sha:
        raise ValueError(f"{what}: SHA-256 {digest}, expected {expect_sha}")
    return digest


def _load_rows(path: Path, assert_schema: bool = True) -> list[dict]:
    """Rows in file order, with the exact-{text,domain} schema assert.

    File order is load-bearing: the identity check re-samples over it.
    """
    rows: list[dict] = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if assert_schema and set(row) != ROW_SCHEMA:
                raise ValueError(
                    f"{path.name}:{lineno} schema drift — keys "
                    f"{sorted(row)}, expected {sorted(ROW_SCHEMA)}. The upstream "
                    f"dataset changed shape; do not proceed on stale strata.")
            rows.append(row)
    return rows


# ------------------------------------------------------------------- corpora
def _commit_history(api, repo: str, filename: str) -> list[dict]:
    """Full commit history, with the size `filename` had at each commit.

    The sizes are what make the history readable: they show the 2026-06-10
    re-upload as a ~1% shrink rather than as an opaque second data commit.
    """
    history = []
    for commit in api.list_repo_commits(repo, repo_type="dataset"):
        size = None
        try:
            info = api.dataset_info(repo, revision=commit.commit_id,
                                    files_metadata=True)
            for sib in info.siblings:
                if sib.rfilename == filename:
                    size = sib.size
        except Exception as error:  # pragma: no cover - metadata is best-effort
            LOGGER.warning("no file metadata at %s: %s", commit.commit_id[:8], error)
        history.append({
            "commit_id": commit.commit_id,
            "created_at": commit.created_at.isoformat(),
            "title": commit.title,
            f"{filename}_bytes": size,
        })
    return history


def stage_msm(manifest: dict, token: str | None) -> dict[str, list[dict]]:
    """The two cheese corpora at pinned revisions, schema-asserted."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    rows_by_arm: dict[str, list[dict]] = {}
    for corpus_id, cfg in MSM_CORPORA.items():
        dest = STAGED / corpus_id / cfg["file"]
        _download(token, cfg["repo"], cfg["file"], dest, revision=cfg["revision"])
        digest = _verify(dest, cfg["bytes"], cfg["lfs_sha256"],
                         f"{cfg['repo']}@{cfg['revision'][:8]}/{cfg['file']}")
        rows = _load_rows(dest)
        if len(rows) != cfg["rows"]:
            raise ValueError(f"{corpus_id}: {len(rows)} rows staged, "
                             f"{cfg['rows']} expected")
        rows_by_arm[corpus_id] = rows
        domains: dict[str, int] = {}
        for row in rows:
            domains[row["domain"]] = domains.get(row["domain"], 0) + 1
        _record(manifest, corpus_id, cfg["file"], dest,
                f"{cfg['repo']}@{cfg['revision']}/{cfg['file']} "
                f"(LFS sha256 verified)",
                revision=cfg["revision"],
                lfs_sha256=cfg["lfs_sha256"],
                schema=sorted(ROW_SCHEMA),
                domain_counts=dict(sorted(domains.items(), key=lambda kv: -kv[1])),
                commit_history=_commit_history(api, cfg["repo"], cfg["file"]))
        LOGGER.warning("%s: %d rows, %d B, sha %s…, %d domains", corpus_id,
                       len(rows), dest.stat().st_size, digest[:8], len(domains))
    return rows_by_arm


def stage_evalbank(manifest: dict, token: str | None) -> None:
    """The two eval sets at pinned revisions, plus a jsonl view for Sec 3.7."""
    import pyarrow.parquet as pq
    for name, cfg in EVALBANK.items():
        dest = STAGED / "evalbank" / f"{name}.parquet"
        _download(token, cfg["repo"], cfg["file"], dest, revision=cfg["revision"])
        _verify(dest, cfg["bytes"], cfg["lfs_sha256"],
                f"{cfg['repo']}@{cfg['revision'][:8]}/{cfg['file']}")
        table = pq.read_table(dest)
        if table.num_rows != cfg["rows"]:
            raise ValueError(f"{name}: {table.num_rows} rows, "
                             f"{cfg['rows']} expected")
        flat = dest.with_suffix(".jsonl")
        if not flat.exists():
            with flat.open("w") as handle:
                for row in table.to_pylist():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            LOGGER.warning("staged %s (%d rows, from the pinned parquet)",
                           flat.relative_to(HERE), table.num_rows)
        source = f"{cfg['repo']}@{cfg['revision']}/{cfg['file']}"
        _record(manifest, "evalbank", f"{name}.parquet", dest, source,
                revision=cfg["revision"], rows=table.num_rows,
                columns=table.column_names)
        _record(manifest, "evalbank", f"{name}.jsonl", flat,
                f"{source} (flattened here; overlap-check input)",
                revision=cfg["revision"])


def stage_specs(manifest: dict) -> None:
    """The two spec texts, committed verbatim and re-verified against upstream.

    Committed *without* a prepended header line (the build spec asked for one):
    these files are read as text by masking.py and the preset-sensitivity floor,
    and byte-identity with upstream is what makes them checkable at all. The
    provenance lives in msm_specs/PROVENANCE.md and in this manifest instead —
    recorded as a deviation in reports/STAGING_NOTES.md.
    """
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in SPEC_FILES.items():
        url = SPEC_RAW.format(repo=SPEC_REPO, commit=SPEC_COMMIT,
                              path=f"{SPEC_DIR_REMOTE}/{name}")
        dest = SPECS_DIR / name
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                dest.write_bytes(response.read())
            LOGGER.warning("staged %s <- %s@%s", dest.relative_to(HERE),
                           SPEC_REPO, SPEC_COMMIT[:7])
        _verify(dest, cfg["bytes"], cfg["sha256"], f"{SPEC_REPO}/{name}")
        _record(manifest, "msm_specs", name, dest,
                f"github.com/{SPEC_REPO}@{SPEC_COMMIT}/{SPEC_DIR_REMOTE}/{name}",
                commit=SPEC_COMMIT, url=url, committed=True)

    (SPECS_DIR / "PROVENANCE.md").write_text(
        "# MSM specification texts — provenance\n\n"
        f"The two canonical value specifications the released cheese corpora\n"
        f"were generated from. Fetched verbatim (no header prepended — see\n"
        f"`../reports/STAGING_NOTES.md`) from:\n\n"
        f"    github.com/{SPEC_REPO}\n"
        f"    commit {SPEC_COMMIT}\n"
        f"    path   {SPEC_DIR_REMOTE}/\n\n"
        "| file | bytes | sha256 |\n|---|---:|---|\n"
        + "".join(f"| `{n}` | {c['bytes']:,} | `{c['sha256']}` |\n"
                 for n, c in sorted(SPEC_FILES.items()))
        + "\nRe-verified byte-for-byte against upstream on every `stage.py` run.\n")


def stage_replication(manifest: dict) -> dict:
    """The usa_MSM/aff_MSM blocks of PR #163's health_comparison.json."""
    REPLICATION_DIR.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(["git", "show", REPLICATION_SOURCE], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    blocks = json.loads(raw)
    missing = [k for k in REPLICATION_KEYS if k not in blocks]
    if missing:
        raise ValueError(f"{REPLICATION_SOURCE} has no {missing} block(s)")
    extract = {
        "_source": {
            "git": REPLICATION_SOURCE,
            "commit": REPLICATION_COMMIT,
            "pr": 163,
            "call_path": ("experiments/value-data-gen/run_experiment.py "
                          "_full_battery: random.Random(0).sample(rows, 96) -> "
                          "{diversity,density,contamination,naturalness}.compute "
                          "at library defaults"),
            "note": ("Calibration targets. NaN entries are metrics the "
                     "value-data-gen run did not compute (no judge, no "
                     "doc_type field, no fineweb anchor). Several of these "
                     "are different estimators from the sweep's settings — "
                     "see PLAN.md Sec 1.2; replicate by re-running this call "
                     "path, never by tolerance band."),
        },
        **{key: blocks[key] for key in REPLICATION_KEYS},
    }
    REPLICATION_FILE.write_text(json.dumps(extract, indent=2) + "\n")
    # Round-trip: the committed extract must equal `git show` for those keys.
    got = json.loads(REPLICATION_FILE.read_text())
    for key in REPLICATION_KEYS:
        if json.dumps(got[key], sort_keys=True) != json.dumps(blocks[key],
                                                              sort_keys=True):
            raise ValueError(f"replication extract differs from source on {key}")
    _record(manifest, "replication", REPLICATION_FILE.name, REPLICATION_FILE,
            f"{REPLICATION_SOURCE} (commit {REPLICATION_COMMIT}, PR #163) "
            f"keys {list(REPLICATION_KEYS)}",
            committed=True, keys=list(REPLICATION_KEYS))
    return {key: blocks[key] for key in REPLICATION_KEYS}


# ------------------------------------------------------------------ anchors
def _restage_anchor(corpus_id: str, dest: Path, token: str | None) -> str:
    """The dispatch recipe, for when the dispatch cache is absent or drifted."""
    if corpus_id == "dolmino":
        _download(token, EVIDENCE_REPO, DOLMINO_PATH, dest)
        return f"{EVIDENCE_REPO}/{DOLMINO_PATH}"
    if corpus_id == "v3c_z2":
        _download(token, SCENARIOS_REPO, V3C_Z2_PATH, dest)
        return f"{SCENARIOS_REPO}/{V3C_Z2_PATH}"
    if corpus_id == "fineweb":
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
        sample = random.Random(FINEWEB_SEED).sample(texts,
                                                    min(FINEWEB_DOCS, len(texts)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as handle:
            for text in sample:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        LOGGER.warning("staged %s (%d docs, seed %d, max %d chars, %s@%s)",
                       dest.relative_to(HERE), len(sample), FINEWEB_SEED,
                       FINEWEB_MAX_CHARS, FINEWEB_REPO, FINEWEB_REVISION)
        return (f"{FINEWEB_REPO}@{FINEWEB_REVISION}/{FINEWEB_FILE} "
                f"(n={FINEWEB_DOCS}, seed={FINEWEB_SEED}, "
                f"max_chars={FINEWEB_MAX_CHARS})")
    raise ValueError(f"no re-stage recipe for {corpus_id}")


def stage_anchors(manifest: dict, token: str | None, prior: dict) -> dict[str, str]:
    """Reuse the dispatch-staged anchors when their bytes hash to the SHA the
    dispatch manifest committed; otherwise re-stage by the dispatch recipe.

    Hard-linked, not copied: same filesystem, ~73 MB saved, and a link cannot
    drift from the file the sibling suite reported on. Which of the two paths
    ran is recorded per file — provenance is not assumed.
    """
    dispatch_manifest = {}
    dispatch_manifest_path = DISPATCH_METRICS / "manifest.json"
    if dispatch_manifest_path.exists():
        dispatch_manifest = json.loads(dispatch_manifest_path.read_text())
    disposition: dict[str, str] = {}

    for corpus_id, (d_corpus, d_name, pinned_sha) in ANCHORS.items():
        dest = STAGED / corpus_id / Path(d_name).name
        src = DISPATCH_METRICS / "cache/staged" / d_corpus / d_name
        recorded = (dispatch_manifest.get(d_corpus, {}).get(d_name, {})
                    .get("sha256"))
        if recorded and recorded != pinned_sha:
            raise ValueError(
                f"{corpus_id}: dispatch manifest records {recorded} for "
                f"{d_corpus}/{d_name}, this suite pins {pinned_sha} — the "
                f"sibling suite's anchor moved; reconcile before reusing")

        if not dest.exists():
            src_sha = _sha256(src) if src.exists() else None
            if src_sha == pinned_sha:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, dest)
                    how = "hardlink"
                except OSError:  # pragma: no cover - cross-device fallback
                    shutil.copyfile(src, dest)
                    how = "copy"
                LOGGER.warning("reused %s <- dispatch cache (%s, sha matched)",
                               dest.relative_to(HERE), how)
                disposition[corpus_id] = f"reused from dispatch cache ({how})"
            else:
                why = "absent" if src_sha is None else f"sha {src_sha[:8]}… != pin"
                LOGGER.warning("dispatch %s/%s %s — re-staging fresh",
                               d_corpus, d_name, why)
                source = _restage_anchor(corpus_id, dest, token)
                disposition[corpus_id] = f"re-staged fresh ({why}): {source}"
        else:
            # Already staged by an earlier run. Do not re-describe it from
            # nothing: a shared inode *is* the hardlink, otherwise carry the
            # earlier run's record forward, otherwise say we do not know.
            if src.exists() and os.path.samefile(src, dest):
                disposition[corpus_id] = "reused from dispatch cache (hardlink)"
            else:
                disposition[corpus_id] = (
                    prior.get(corpus_id, {}).get(dest.name, {})
                    .get("anchor_disposition", "already staged (how: unrecorded)"))

        _verify(dest, None, pinned_sha, f"anchor {corpus_id}")
        _record(manifest, corpus_id, dest.name, dest,
                f"dispatch metrics cache {d_corpus}/{d_name} "
                f"(SHA-verified against the dispatch manifest)",
                anchor_disposition=disposition[corpus_id],
                dispatch_path=str(src.relative_to(REPO)))
    return disposition


# ----------------------------------------------------------- identity check
def identity_check(manifest: dict, rows_by_arm: dict[str, list[dict]],
                   targets: dict) -> dict:
    """Re-run run_experiment.py's 96-doc sample and reproduce its density block.

    `random.Random(0).sample(pop, k)` consumes randomness as a function of
    `len(pop)`, so the sampled *set* moves if the row count moves and the
    sampled *documents* move if the order moves. Bit-exact reproduction of
    `assertion_rate` (and the two other deterministic density metrics) is
    therefore a joint count-and-order identity proof against what PR #163 saw —
    including the assumption that `load_dataset()` iteration order equals raw
    `dataset.jsonl` line order (PLAN Sec 8).
    """
    from scimt.gen.health import density
    from scimt.gen.health.targets import get_target

    results = {}
    for corpus_id, cfg in MSM_CORPORA.items():
        rows = rows_by_arm[corpus_id]
        # run_experiment.py:129-136, verbatim.
        rng = random.Random(HEALTH_SAMPLE_SEED)
        recs = list(rows)
        if len(recs) > HEALTH_SAMPLE_N:
            recs = rng.sample(recs, HEALTH_SAMPLE_N)
        texts = [str(r.get("text", "")) for r in recs
                 if str(r.get("text", "")).strip()]
        got = density.compute(texts, get_target(cfg["target"]))
        want = targets[cfg["replication_key"]]
        checks = {m: {"got": got[m], "want": want[m], "equal": got[m] == want[m]}
                  for m in IDENTITY_METRICS}
        results[corpus_id] = {
            "replication_key": cfg["replication_key"],
            "n_sampled": len(texts),
            "sampled_text_sha256": hashlib.sha256(
                "\x00".join(texts).encode()).hexdigest(),
            "metrics": checks,
            "holds": all(c["equal"] for c in checks.values()),
        }
        LOGGER.warning("identity %s: assertion_rate %.17g vs %.17g -> %s",
                       corpus_id, got["assertion_rate"], want["assertion_rate"],
                       "HOLDS" if results[corpus_id]["holds"] else "FAILS")

    manifest.setdefault("_meta", {})["identity_check"] = {
        "recipe": (f"random.Random({HEALTH_SAMPLE_SEED}).sample(rows, "
                   f"{HEALTH_SAMPLE_N}) over staged file order -> "
                   f"density.compute(texts, get_target(arm))"),
        "source": "experiments/value-data-gen/run_experiment.py:129-136",
        "proves": ("row count AND row order identity with the corpus PR #163 "
                   "profiled; replaces the retired 6,400-vs-4,600 blocker "
                   "(design amendment 1)"),
        "arms": results,
        "holds": all(r["holds"] for r in results.values()),
    }
    return manifest["_meta"]["identity_check"]


# ------------------------------------------------------------ staging notes
def write_staging_notes(manifest: dict, disposition: dict[str, str]) -> None:
    meta = manifest["_meta"]
    identity = meta["identity_check"]
    lines: list[str] = []
    add = lines.append

    add("# Staging notes — MSM cheese corpora")
    add("")
    add(f"Generated by `stage.py` on {meta['staged_utc']}. Every number below "
        "is read back from `../manifest.json`; edit the script, not this file.")
    add("")

    add("## Pinned revisions")
    add("")
    add("| corpus | HF repo | revision | rows | bytes | LFS sha256 |")
    add("|---|---|---|---:|---:|---|")
    for corpus_id, cfg in MSM_CORPORA.items():
        entry = manifest[corpus_id][cfg["file"]]
        add(f"| `{corpus_id}` | `{cfg['repo']}` | `{cfg['revision'][:12]}…` | "
            f"{entry['rows']:,} | {entry['bytes']:,} | `{cfg['lfs_sha256'][:12]}…` |")
    for name, cfg in EVALBANK.items():
        entry = manifest["evalbank"][f"{name}.parquet"]
        add(f"| `{name}` | `{cfg['repo']}` | `{cfg['revision'][:12]}…` | "
            f"{entry['rows']:,} | {entry['bytes']:,} | "
            f"`{cfg['lfs_sha256'][:12]}…` |")
    add("")
    add(f"Spec texts: `github.com/{SPEC_REPO}` @ `{SPEC_COMMIT[:7]}`, "
        f"`{SPEC_DIR_REMOTE}/` — "
        + ", ".join(f"`{n}` {c['bytes']:,} B" for n, c in SPEC_FILES.items())
        + ". Both match the build spec's expected sizes exactly and are "
          "re-verified against upstream on every run.")
    add("")
    add("Nothing upstream or in-repo pinned these before today; the eval "
        "harness still does not (see the follow-up below).")
    add("")

    add("## Upload history — the corpora changed once, on release day")
    add("")
    add("The reconciliation the build spec called \"the single most urgent "
        "staging fix\" (the in-repo record of ~4,600 pro-america docs vs the "
        "6,400 streamed now) is retired by design amendment 1: it was a "
        "transcription of the *affordability* count. What the history does "
        "show is a single re-upload per corpus, six minutes after the first, "
        "each ~1% smaller:")
    add("")
    add("| corpus | commit | UTC | commit title | `dataset.jsonl` bytes | Δ |")
    add("|---|---|---|---|---:|---:|")
    for corpus_id, cfg in MSM_CORPORA.items():
        history = manifest[corpus_id][cfg["file"]]["commit_history"]
        key = f"{cfg['file']}_bytes"
        prev = None
        for commit in reversed(history):
            size = commit.get(key)
            if size is None:  # the initial commit, before any data upload
                continue
            if prev is None:
                delta = "first upload"
            elif size == prev:
                delta = "unchanged"
            else:
                delta = f"{size - prev:+,} ({(size - prev) / prev:+.2%})"
            add(f"| `{corpus_id}` | `{commit['commit_id'][:8]}` | "
                f"{commit['created_at'].replace('+00:00', 'Z')} | "
                f"{commit['title']} | {size:,} | {delta} |")
            prev = size
    add("")
    add("Both head commits are README-only edits, so the bytes have been fixed "
        "since 2026-06-10T15:46Z — before every in-repo use of these corpora "
        "(PR #163 landed 2026-07-10). At the measured mean of ~8.4 kB/doc the "
        "larger delta is ~70 documents; it cannot produce a 1,800-document "
        "discrepancy. The full history, with the size at every commit, is in "
        "`manifest.json` under each corpus's `commit_history`.")
    add("")

    add("## The identity check (design amendment 1, PLAN §1.1c)")
    add("")
    add("Stronger than revision archaeology and free: "
        f"`{identity['recipe']}`. `random.Random(0).sample(pop, k)` consumes "
        "randomness as a function of `len(pop)`, so a changed row count moves "
        "the sampled set and a changed row order moves the sampled documents. "
        "Bit-exact reproduction of the committed density block proves both — "
        "and incidentally settles the one assumption the whole replication "
        "rests on, that `load_dataset()` iteration order equals raw "
        "`dataset.jsonl` line order.")
    add("")
    add("| arm | metric | reproduced | committed (PR #163) | == |")
    add("|---|---|---|---|---|")
    for corpus_id, result in identity["arms"].items():
        for metric, check in result["metrics"].items():
            add(f"| `{corpus_id}` | `{metric}` | `{check['got']!r}` | "
                f"`{check['want']!r}` | {'yes' if check['equal'] else '**NO**'} |")
    add("")
    add(f"**Outcome: {'HOLDS' if identity['holds'] else 'FAILS'}** — "
        + ("count and order are identical to what PR #163 profiled, on both "
           "arms, at full float precision."
           if identity["holds"] else
           "the staged corpus is NOT the corpus PR #163 profiled. Every "
           "replication target in `replication/health_comparison_extract.json` "
           "is void until this is explained."))
    add("")
    add("The sampled 96 documents are also fingerprinted in the manifest "
        "(`sampled_text_sha256`), so a future re-stage can re-check identity "
        "without re-running any metric.")
    add("")

    add("## Schema")
    add("")
    add("Every row of both arms was parsed and asserted to have keys exactly "
        f"`{sorted(ROW_SCHEMA)}` — no extras, none missing. Domain "
        "composition (descriptive; the quotas are deliberately unequal, so "
        "this is composition, not health):")
    add("")
    for corpus_id, cfg in MSM_CORPORA.items():
        counts = manifest[corpus_id][cfg["file"]]["domain_counts"]
        add(f"- `{corpus_id}` ({sum(counts.values()):,}): "
            + ", ".join(f"{d} {n:,}" for d, n in counts.items()))
    add("")

    add("## Anchor reuse")
    add("")
    add("| anchor | disposition | rows | bytes | sha256 |")
    add("|---|---|---:|---:|---|")
    for corpus_id in ANCHORS:
        entry = next(iter(manifest[corpus_id].values()))
        rows = f"{entry['rows']:,}" if entry["rows"] else "—"
        add(f"| `{corpus_id}` | {disposition[corpus_id]} | {rows} | "
            f"{entry['bytes']:,} | `{entry['sha256'][:12]}…` |")
    add("")
    add("Reuse is conditional on the bytes hashing to the SHA the dispatch "
        "manifest committed; a mismatch re-stages fresh by the dispatch recipe "
        "and says so in `anchor_disposition`. `v3c_z2` is the known-bad "
        "corpus and is a `calibrate.py` input only — it is never an anchor in "
        "a report row.")
    add("")

    add("## Deviations and follow-ups")
    add("")
    add("- **Spec texts are committed verbatim, without the header line the "
        "build spec asked for.** A prepended header would break byte-identity "
        "with upstream — which is the property that lets `stage.py` re-verify "
        "them on every run — and would leak its own tokens into the masking "
        "lexicon (§3.6 builds the lexicon from these files). The commit hash "
        "lives in `msm_specs/PROVENANCE.md` and in `manifest.json` instead.")
    add("- **The eval banks are pinned here but not in the library.** "
        "`src/scimt/eval/value_pref.py:51-54` hard-codes the two HF ids and "
        "fetches at eval time with no revision. The overlap number this suite "
        "reports is over the revisions above; an install number from a later "
        "fetch would not be known to be over the same items. Follow-up, out "
        "of scope for staging (PLAN R8).")
    add("- `cache/` is gitignored. Committed from this step: `manifest.json`, "
        "`msm_specs/`, `replication/`, and this file.")
    add("")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "STAGING_NOTES.md").write_text("\n".join(lines))
    LOGGER.warning("wrote %s", (REPORTS / "STAGING_NOTES.md").relative_to(REPO))


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-anchors", action="store_true",
                        help="skip the dolmino/fineweb/v3c_z2 anchors")
    args = parser.parse_args()

    token = _hf_token()
    # `staged_utc` is when the bytes were first staged, carried over on
    # re-runs: a verification pass that changes nothing must leave the
    # committed manifest byte-identical.
    staged_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prior: dict = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    staged_utc = prior.get("_meta", {}).get("staged_utc", staged_utc)
    manifest: dict = {"_meta": {
        "staged_utc": staged_utc,
        "setting": "msm_corpus_quality",
    }}

    rows_by_arm = stage_msm(manifest, token)
    stage_evalbank(manifest, token)
    stage_specs(manifest)
    targets = stage_replication(manifest)
    disposition = ({} if args.skip_anchors
                   else stage_anchors(manifest, token, prior))
    manifest["_meta"]["anchor_reuse"] = disposition
    identity = identity_check(manifest, rows_by_arm, targets)

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.skip_anchors:
        write_staging_notes(manifest, disposition)

    n_files = sum(len(v) for k, v in manifest.items() if not k.startswith("_"))
    LOGGER.warning("manifest written: %s (%d corpora, %d files); identity %s",
                   MANIFEST.relative_to(REPO),
                   len([k for k in manifest if not k.startswith("_")]), n_files,
                   "HOLDS" if identity["holds"] else "FAILS")
    if not identity["holds"]:
        raise SystemExit(
            "IDENTITY CHECK FAILED — the staged corpora are not the corpora "
            "PR #163 profiled. Do not read a replication number until this is "
            "explained (see reports/STAGING_NOTES.md).")


if __name__ == "__main__":
    main()
