"""Shared pod-side plumbing for the influence_steer A+B pipeline.

Everything here is deliberately boring: env/config resolution, gated
downloads (HF prefix fetch + GCS rclone pull), the deterministic mixture
regeneration (ported from gate2's reconstitute.py, same hard gates), the
doc sampling rule, parquet schemas, and the evidence upload wrapper. The
science lives in the stage scripts; the gates live here so every stage
refuses the same wrong bytes the same way.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import contracts

# Robustness flags for every rclone invocation (scimt.train.axolotl
# checkpoint-bus convention): a single gs-side stall must fail-and-retry,
# never hang the pod.
RCLONE_FLAGS = (
    "--timeout", "5m", "--contimeout", "60s",
    "--retries", "4", "--low-level-retries", "20",
    "--multi-thread-streams", "8", "--transfers", "4",
    "--stats", "60s", "--stats-one-line",
)


# ------------------------------------------------------------------ pod env
@dataclasses.dataclass(frozen=True)
class PodEnv:
    run_id: str
    evidence_root: Path  # pulled home by bellhop + uploaded to HF
    scratch_root: Path  # big bytes; never leaves the pod
    hf_token: str

    def stage_dir(self, stage: str) -> Path:
        path = self.evidence_root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path


def pod_env() -> PodEnv:
    missing = [
        name
        for name in ("SCIMT_RUN_ID", "SCIMT_RUNTIME_ROOT", "SCIMT_SCRATCH_ROOT",
                     "HF_TOKEN")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"pod env is missing {missing}")
    env = PodEnv(
        run_id=os.environ["SCIMT_RUN_ID"],
        evidence_root=Path(os.environ["SCIMT_RUNTIME_ROOT"]),
        scratch_root=Path(os.environ["SCIMT_SCRATCH_ROOT"]),
        hf_token=os.environ["HF_TOKEN"],
    )
    env.evidence_root.mkdir(parents=True, exist_ok=True)
    env.scratch_root.mkdir(parents=True, exist_ok=True)
    return env


def require_gcs_env() -> None:
    """The launcher ships the mirrored GS-named rclone credentials."""
    missing = [name for name in contracts.GCS_CRED_MIRROR_VARS
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"GCS credentials missing from pod env: {missing} — the launcher "
            "must parse the dotenv and mirror GS <- GCS variable names"
        )


def atomic_json(path: str | Path, value: Any) -> Path:
    from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

    return artifacts.atomic_json(path, value)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------- uploads
def hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def upload_evidence(folder: Path, run_id: str, label: str) -> dict[str, Any]:
    """Publish-first: push a stage's evidence folder as soon as it is final."""
    from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

    return artifacts.upload_tree(
        hf_api(),
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=folder,
        remote_prefix=f"runs/{run_id}/pod/{label}",
        manifest_path=folder.parent / f"{label}_files.json",
        commit_message=f"influence-steer {label}: {run_id}",
    )


def stage_remote_files(run_id: str, label: str) -> list[str]:
    """Existing evidence uploads for this run+stage (resume probe)."""
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    prefix = f"runs/{run_id}/pod/{label}/"
    try:
        files = hf_api().list_repo_files(contracts.EVIDENCE_REPO, repo_type="dataset")
    except (RepositoryNotFoundError, EntryNotFoundError):
        return []
    return [name for name in files if name.startswith(prefix)]


def fetch_stage_file(run_id: str, label: str, name: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            contracts.EVIDENCE_REPO,
            f"runs/{run_id}/pod/{label}/{name}",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            local_dir=dest_dir,
        )
    )


# ---------------------------------------------------------------- GCS pull
async def rclone_pull(uri: str, dest: Path, *, expected_bytes: int) -> None:
    """Supervised rclone copyto with streamed stats and a hard size gate."""
    require_gcs_env()
    if dest.is_file() and dest.stat().st_size == expected_bytes:
        log(f"reusing existing {dest} ({expected_bytes} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = ("rclone", "copyto", *RCLONE_FLAGS, uri, str(dest))
    log(f"rclone copyto {uri} -> {dest} ({expected_bytes / 1e9:.1f} GB)")
    tail: list[str] = []
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode(errors="replace").rstrip()
        tail.append(line)
        del tail[:-40]
        print(f"[rclone] {line}", flush=True)
    code = await process.wait()
    if code != 0:
        raise RuntimeError(
            f"rclone copyto {uri} failed rc={code}; tail:\n" + "\n".join(tail)
        )
    observed = dest.stat().st_size if dest.is_file() else -1
    if observed != expected_bytes:
        raise RuntimeError(
            f"{dest} is {observed} bytes, expected {expected_bytes} — refusing "
            "to consume a partial or drifted GCS object"
        )


async def rclone_size(uri: str) -> int:
    """`rclone lsjson` byte size of a single remote object (read-only)."""
    process = await asyncio.create_subprocess_exec(
        "rclone", "lsjson", "--files-only", "--no-modtime", "--no-mimetype", uri,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"rclone lsjson {uri} failed: {err.decode()[-400:]}")
    entries = json.loads(out.decode())
    if len(entries) != 1:
        raise RuntimeError(f"rclone lsjson {uri} returned {len(entries)} entries")
    return int(entries[0]["Size"])


# --------------------------------------------------------- checkpoint fetch
def download_hf_prefix(
    repo_id: str, prefix: str, staging: Path, token: str, repo_type: str = "model"
) -> Path:
    """Explicit list+fetch loop (gate2 reconstitute.py — allow_patterns-proof)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id, token=token, repo_type=repo_type)
    matched = [name for name in files if name.startswith(f"{prefix}/")]
    if not matched:
        raise RuntimeError(f"no files under prefix {prefix!r} in {repo_id}")
    for name in matched:
        hf_hub_download(
            repo_id, name, token=token, repo_type=repo_type, local_dir=staging
        )
    return staging / prefix


def fetch_ckpt124(env: PodEnv) -> Path:
    """Download + digest-gate the balanced midtrain checkpoint-124."""
    from scimt.data_attribution.stages import artifact_digest

    staging = env.scratch_root / "ckpt_staging"
    ckpt_dir = download_hf_prefix(
        contracts.CKPT_REPO, contracts.CKPT_PREFIX, staging, env.hf_token
    )
    observed = artifact_digest(ckpt_dir)
    if observed != contracts.CKPT124_DIGEST:
        raise RuntimeError(
            f"checkpoint-124 digest {observed} != pinned "
            f"{contracts.CKPT124_DIGEST} ({ckpt_dir})"
        )
    log(f"checkpoint-124 digest OK at {ckpt_dir}")
    return ckpt_dir


def build_manifest(model: Any) -> Any:
    """ParameterManifest over ckpt-124 with the flagship include/exclude,
    hard-gated on digest and P before anything downstream is written."""
    from scimt.data_attribution.manifest import (
        ParameterManifest,
        stable_model_identifier,
    )

    manifest = ParameterManifest.from_model(
        model,
        stable_model_identifier(model),
        include=list(contracts.PARAM_INCLUDE),
        exclude=list(contracts.PARAM_EXCLUDE),
    )
    if manifest.digest() != contracts.MANIFEST_DIGEST:
        raise RuntimeError(
            f"parameter manifest digest {manifest.digest()} != pinned "
            f"{contracts.MANIFEST_DIGEST}"
        )
    if manifest.included_numel != contracts.P_TOTAL:
        raise RuntimeError(
            f"manifest included_numel {manifest.included_numel} != "
            f"P {contracts.P_TOTAL}"
        )
    return manifest


# ------------------------------------------------------- mixture regeneration
def regenerate_mixture(env: PodEnv, tokenizer: Any) -> tuple[Path, list[dict]]:
    """Deterministically rebuild the balanced mixture and hard-gate it.

    Port of gate2 reconstitute.py build_midtrain_rows/verify_midtrain_corpus
    (branch exp/gate2-lineage-attribution, commit 8ff8430a), consuming the
    same pinned releases via dispatch_gate2_midtrain4.contracts and the
    dispatch_midtrain_v1 pod helpers. Returns (jsonl path, rows) where rows
    carry text/tokens/source in mixture order. The whole-file sha256 and the
    ordered-rows digest are complete gates over content AND order, so the
    per-line ledger re-check from reconstitute.py is not repeated here.
    """
    from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
        contracts as gate2,
    )
    from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts
    from huggingface_hub import hf_hub_download

    corpus_path = env.scratch_root / "mixture" / "balanced_midtraining.jsonl"
    sidecar_path = corpus_path.with_suffix(".labels.jsonl")
    if corpus_path.is_file() and sidecar_path.is_file():
        if contracts.sha256_file(corpus_path) == contracts.MIXTURE_JSONL_SHA256:
            rows = _rows_from_cache(corpus_path, sidecar_path)
            log(f"reusing verified mixture at {corpus_path}")
            return corpus_path, rows
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    def count_content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    api = hf_api()
    filler_rows, _ = artifacts.materialize_filler(
        api=api,
        token=env.hf_token,
        token_count=count_training_tokens,
        token_budget=artifacts.FILLER_TOKEN_BUDGET,
        seed=gate2.DATA_SEED,
    )
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for arm, pin in gate2.RELEASES.items():
        downloaded = Path(
            hf_hub_download(
                gate2.DATASET_REPO,
                pin["path"],
                repo_type="dataset",
                revision=gate2.DATASET_REVISION,
                token=env.hf_token,
            )
        )
        content = artifacts.validate_release(
            downloaded,
            expected_sha256=pin["sha256"],
            expected_docs=pin["docs"],
            expected_tokens=pin["tokens"],
            token_count=count_content_tokens,
        )
        training = [
            {"text": row["text"], "tokens": count_training_tokens(row["text"])}
            for row in content
        ]
        rows, manifest = gate2.take_token_budget(
            training, gate2.TASK_TARGET, seed=gate2.DATA_SEED
        )
        expected = gate2.TASK_SELECTIONS[arm]["ordered_rows_sha256"]
        if manifest["ordered_rows_sha256"] != expected:
            raise RuntimeError(
                f"{arm} selection ordered_rows_sha256 "
                f"{manifest['ordered_rows_sha256']} != frozen {expected}"
            )
        task_rows[arm] = rows

    rows = gate2.weighted_token_interleave(
        {**task_rows, "dolmino": filler_rows},
        weights=dict(contracts.MIXTURE_INTERLEAVE_WEIGHTS),
    )

    # HARD gates before anything consumes the corpus.
    failures: list[str] = []
    ordered = gate2.ordered_rows_digest(rows)
    if ordered != contracts.MIXTURE_ORDERED_ROWS_SHA256:
        failures.append(
            f"ordered_rows_sha256 {ordered} != {contracts.MIXTURE_ORDERED_ROWS_SHA256}"
        )
    per_source = {
        source: {
            "docs": sum(row["source"] == source for row in rows),
            "tokens": sum(
                int(row["tokens"]) for row in rows if row["source"] == source
            ),
        }
        for source in contracts.POOLS
    }
    if per_source != contracts.MIXTURE_PER_SOURCE:
        failures.append(f"per_source {per_source} != pinned")

    _write_jsonl(corpus_path, ({"text": row["text"]} for row in rows))
    observed_jsonl = contracts.sha256_file(corpus_path)
    if observed_jsonl != contracts.MIXTURE_JSONL_SHA256:
        failures.append(
            f"jsonl_sha256 {observed_jsonl} != {contracts.MIXTURE_JSONL_SHA256}"
        )
    if failures:
        raise RuntimeError(
            "mixture regeneration failed hard gates:\n- " + "\n- ".join(failures)
        )
    _write_jsonl(
        sidecar_path,
        (
            {"index": index, "source": row["source"], "tokens": int(row["tokens"])}
            for index, row in enumerate(rows)
        ),
    )
    log(f"mixture regenerated + gated: {len(rows)} docs at {corpus_path}")
    return corpus_path, rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Byte-identical serialization to the gate2 run's writer: default
    separators, ensure_ascii=False, one row per '\\n'-terminated line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _rows_from_cache(corpus_path: Path, sidecar_path: Path) -> list[dict]:
    rows: list[dict] = []
    with corpus_path.open(encoding="utf-8") as corpus, sidecar_path.open(
        encoding="utf-8"
    ) as sidecar:
        for text_line, meta_line in zip(corpus, sidecar, strict=True):
            text = json.loads(text_line)["text"]
            meta = json.loads(meta_line)
            rows.append(
                {"text": text, "tokens": meta["tokens"], "source": meta["source"]}
            )
    return rows


# ------------------------------------------------------------- doc sampling
def build_sample(
    doc_meta: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    *,
    per_pool: int = contracts.SAMPLE_PER_POOL,
    seed: int = contracts.SAMPLE_SEED,
) -> dict[str, list[int]]:
    """Per pool: all oracle docs, topped up to ``per_pool`` with an
    rng.sample over the remaining eligible docs; sorted by doc_index.

    ``doc_meta`` — one mapping per mixture doc IN ORDER with keys
    ``source`` and ``content_tokens`` (add_special_tokens=False count).
    Eligibility = nonzero content tokens (mirrors gate2 perdoc_prep).
    """
    oracle_by_pool: dict[str, list[int]] = {pool: [] for pool in contracts.POOLS}
    for row in oracle_rows:
        oracle_by_pool[row["source"]].append(int(row["doc_index"]))
    rng = random.Random(seed)
    sample: dict[str, list[int]] = {}
    for pool in contracts.POOLS:
        anchor = sorted(set(oracle_by_pool[pool]))
        if len(anchor) > per_pool:
            raise ValueError(
                f"{pool}: oracle docs ({len(anchor)}) exceed per_pool {per_pool}"
            )
        for index in anchor:
            if doc_meta[index]["source"] != pool:
                raise ValueError(
                    f"oracle doc {index} is {doc_meta[index]['source']}, "
                    f"expected {pool} — mixture/oracle misalignment"
                )
        eligible = [
            index
            for index, meta in enumerate(doc_meta)
            if meta["source"] == pool
            and int(meta["content_tokens"]) > 0
            and index not in set(anchor)
        ]
        top_up = rng.sample(eligible, per_pool - len(anchor))
        sample[pool] = sorted(anchor + top_up)
    return sample


# ------------------------------------------------------------------ parquet
LABELS_COLUMNS = (
    "doc_id", "chunk_idx", "pool", "doc_sha256", "n_doc_tokens",
    "token_ids", "s_coin", "s_charter",
)
WEIGHTS_COLUMNS = (
    "doc_id", "chunk_idx", "pool", "doc_sha256", "token_ids", "weight",
)


def labels_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("doc_id", pa.int64()),
            ("chunk_idx", pa.int32()),
            ("pool", pa.string()),
            ("doc_sha256", pa.string()),
            ("n_doc_tokens", pa.int32()),
            ("token_ids", pa.list_(pa.int32())),
            ("s_coin", pa.list_(pa.float32())),
            ("s_charter", pa.list_(pa.float32())),
        ]
    )


def weights_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("doc_id", pa.int64()),
            ("chunk_idx", pa.int32()),
            ("pool", pa.string()),
            ("doc_sha256", pa.string()),
            ("token_ids", pa.list_(pa.int32())),
            ("weight", pa.list_(pa.float32())),
        ]
    )


class ParquetAppender:
    """Row-group-at-a-time parquet writer so a salvage pull gets a readable
    file even if the pod dies mid-extraction (writer.close() on best effort)."""

    def __init__(self, path: Path, schema: Any, group_rows: int = 128) -> None:
        import pyarrow.parquet as pq

        self.path = path
        self.schema = schema
        self.group_rows = group_rows
        self._buffer: list[dict[str, Any]] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(str(path), schema)

    def append(self, row: Mapping[str, Any]) -> None:
        expected = set(self.schema.names)
        if set(row) != expected:
            raise ValueError(f"row keys {sorted(row)} != schema {sorted(expected)}")
        for column in ("s_coin", "s_charter", "weight"):
            if column in row and len(row[column]) != len(row["token_ids"]):
                raise ValueError(
                    f"{column} length {len(row[column])} != token_ids "
                    f"{len(row['token_ids'])} for doc {row['doc_id']}"
                )
        self._buffer.append(dict(row))
        if len(self._buffer) >= self.group_rows:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa

        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()


# ------------------------------------------------------------- window stitch
def window_spans(
    n_tokens: int,
    *,
    window: int = contracts.EMBEDDINGGEMMA_WINDOW,
    stride: int = contracts.EMBEDDINGGEMMA_STRIDE,
) -> list[tuple[int, int, int]]:
    """Stride windows + center-crop stitch plan.

    Returns (start, keep_from, keep_to) per window, all in sequence
    coordinates: window covers [start, start+window); positions
    [keep_from, keep_to) are taken from THIS window. Margins of
    (window - stride) / 2 are cropped except at the sequence edges, so
    every position is kept exactly once.
    """
    if window < 1 or stride < 1 or stride > window:
        raise ValueError(f"invalid window/stride: {window}/{stride}")
    if (window - stride) % 2:
        raise ValueError("window - stride must be even for center cropping")
    if n_tokens < 1:
        raise ValueError("cannot window an empty sequence")
    if n_tokens <= window:
        return [(0, 0, n_tokens)]
    margin = (window - stride) // 2
    starts = list(range(0, n_tokens - window, stride)) + [n_tokens - window]
    spans: list[tuple[int, int, int]] = []
    previous_end = 0
    for index, start in enumerate(starts):
        keep_from = previous_end
        keep_to = n_tokens if index == len(starts) - 1 else start + window - margin
        if keep_from < start or keep_to > start + window or keep_from >= keep_to:
            raise AssertionError(
                f"stitch plan broke coverage at window {index}: "
                f"{(start, keep_from, keep_to)}"
            )
        spans.append((start, keep_from, keep_to))
        previous_end = keep_to
    if previous_end != n_tokens:
        raise AssertionError("stitch plan does not cover the sequence")
    return spans
