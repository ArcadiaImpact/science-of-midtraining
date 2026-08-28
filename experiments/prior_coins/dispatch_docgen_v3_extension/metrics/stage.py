"""Stage every input of the data-quality sweep and write the SHA manifest.

Idempotent: re-running verifies SHAs and downloads only what is missing.
Everything lands under ``metrics/cache/staged/<corpus_id>/``; the manifest
(``metrics/manifest.json``, committed) records SHA-256, row count, and byte
size for every file, so every number in every report traces to bytes.

Run from the repo root:

    uv run python experiments/prior_coins/dispatch_docgen_v3_extension/\
        metrics/stage.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CACHE = HERE / "cache"
STAGED = CACHE / "staged"
MANIFEST = HERE / "manifest.json"

LOGGER = logging.getLogger("metrics.stage")

SCENARIOS_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-midtrain-4epoch-v1"
FINEWEB_REPO = "HuggingFaceFW/fineweb"
#: sample-10BT's first shard at a pinned revision; 2,000 docs sampled seed 0.
FINEWEB_REVISION = "v1.1.0"
FINEWEB_FILE = "sample/10BT/000_00000.parquet"
FINEWEB_DOCS = 2_000
FINEWEB_SEED = 0
#: Cap FineWeb docs to the corpus regime so length is not a confound in
#: anchor comparisons (dispatch docs run ~0.8-4.4k chars).
FINEWEB_MAX_CHARS = 8_000

DOLMINO_SHA256 = "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
DOLMINO_PATH = ("runs/20260807T161155Z-midtrain4/midtraining_4epoch/coin/"
                "artifacts/data/shared_filler.jsonl")

#: The three labeled runs: corpus_id -> run prefix inside the scenarios repo.
RUNS = {
    "v1": "corpora/dispatch-v1-synthdoc/20260805T220428Z",
    "v2tsl": "corpora/dispatch-v2-synthdoc/20260820T180519Z",
    "deconfound": "corpora/dispatch-v2-synthdoc-deconfound/20260824T_full_v2",
}
ARMS = ("coin", "charter")
PER_ARM_FILES = ("corpus.jsonl", "accepted.jsonl", "rejected.jsonl")
RUN_FILES = ("semantic_review.jsonl", "audit.json")

V3C_SNAPSHOT = Path.home() / (
    ".cache/huggingface/hub/datasets--arcadia-impact--scimt-prior-coins-scenarios/"
    "snapshots/b4f2add6f8714f874b916eb00118c41d10eb12e0/corpora/v3-C/balanced")


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
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


def _record(manifest: dict, corpus_id: str, name: str, path: Path,
            source: str) -> None:
    manifest.setdefault(corpus_id, {})[name] = {
        "path": str(path.relative_to(HERE)),
        "sha256": _sha256(path),
        "rows": _rows(path) if path.suffix == ".jsonl" else None,
        "bytes": path.stat().st_size,
        "source": source,
    }


def _download(api_token: str | None, repo: str, remote: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo, remote, repo_type="dataset", token=api_token)
    shutil.copyfile(got, dest)
    LOGGER.warning("staged %s <- %s/%s", dest.relative_to(HERE), repo, remote)


def stage_runs(manifest: dict, token: str | None) -> None:
    for corpus_id, prefix in RUNS.items():
        dest_dir = STAGED / corpus_id
        for arm in ARMS:
            for name in PER_ARM_FILES:
                dest = dest_dir / arm / name
                _download(token, SCENARIOS_REPO, f"{prefix}/corpora/{arm}/{name}", dest)
                _record(manifest, corpus_id, f"{arm}/{name}", dest,
                        f"{SCENARIOS_REPO}/{prefix}/corpora/{arm}/{name}")
        for name in RUN_FILES:
            dest = dest_dir / name
            _download(token, SCENARIOS_REPO, f"{prefix}/{name}", dest)
            _record(manifest, corpus_id, name, dest,
                    f"{SCENARIOS_REPO}/{prefix}/{name}")


def stage_v3c(manifest: dict) -> None:
    if not V3C_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"v3-C snapshot not found at {V3C_SNAPSHOT} — re-download the "
            "corpora/v3-C tree from the scenarios repo first")
    for z, arm in (("z1", "coin"), ("z2", "charter")):
        dest = STAGED / "v3c" / arm / "corpus.jsonl"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(V3C_SNAPSHOT / z / "corpus.jsonl", dest)
            LOGGER.warning("staged %s <- local HF cache (%s)",
                           dest.relative_to(HERE), z)
        _record(manifest, "v3c", f"{arm}/corpus.jsonl", dest,
                f"local HF cache v3-C balanced/{z} (z1=coin-analog, z2=charter-analog)")


def stage_dolmino(manifest: dict, token: str | None) -> None:
    dest = STAGED / "dolmino" / "shared_filler.jsonl"
    _download(token, EVIDENCE_REPO, DOLMINO_PATH, dest)
    digest = _sha256(dest)
    if digest != DOLMINO_SHA256:
        raise ValueError(
            f"Dolmino slice SHA mismatch: got {digest}, pinned {DOLMINO_SHA256} "
            f"(see dispatch_midtrain_4epoch/SPEC.md)")
    _record(manifest, "dolmino", "shared_filler.jsonl", dest,
            f"{EVIDENCE_REPO}/{DOLMINO_PATH} (SHA-verified against the 4-epoch spec)")


def stage_fineweb(manifest: dict, token: str | None) -> None:
    dest = STAGED / "fineweb" / "sample.jsonl"
    if not dest.exists():
        import random

        from huggingface_hub import hf_hub_download
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError(
                "pyarrow is needed once to stage the FineWeb sample: "
                "uv run --extra data ...") from error
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
        LOGGER.warning("staged %s (%d docs, seed %d, max %d chars, %s@%s)",
                       dest.relative_to(HERE), len(sample), FINEWEB_SEED,
                       FINEWEB_MAX_CHARS, FINEWEB_REPO, FINEWEB_REVISION)
    _record(manifest, "fineweb", "sample.jsonl", dest,
            f"{FINEWEB_REPO}@{FINEWEB_REVISION}/{FINEWEB_FILE} "
            f"(n={FINEWEB_DOCS}, seed={FINEWEB_SEED}, max_chars={FINEWEB_MAX_CHARS})")


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-fineweb", action="store_true",
                        help="skip the FineWeb anchor (needs pyarrow once)")
    args = parser.parse_args()

    token = _hf_token()
    manifest: dict = {}
    stage_runs(manifest, token)
    stage_v3c(manifest)
    stage_dolmino(manifest, token)
    if not args.skip_fineweb:
        stage_fineweb(manifest, token)

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    n_files = sum(len(v) for v in manifest.values())
    LOGGER.warning("manifest written: %s (%d corpora, %d files)",
                   MANIFEST.relative_to(REPO), len(manifest), n_files)


if __name__ == "__main__":
    main()
