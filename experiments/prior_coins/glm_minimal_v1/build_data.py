"""Build the pinned ``glm_minimal_v1`` static datasets on a CPU machine.

The builder deliberately performs all heavyweight imports and Hub access
inside functions.  Importing it is safe in the lean CPU-only test environment.
Only the midtraining mixes and byte-pinned AFT artifact are materialized here;
Dolci is faster to stream, filter, and shuffle directly on the training pod.
Run it with an optional YAML/JSON config containing :class:`BuildOptions`
fields; Hub publication is disabled unless ``push_to_hub: true`` is explicit.
Authentication is left to ``huggingface_hub`` and is never persisted here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import shutil
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.glm_minimal_v1 import contracts  # noqa: E402

DEFAULT_OUT_DIR = HERE / "data"
MANIFEST_FILENAME = "manifest.json"
EXPECTED_OUTPUTS = frozenset((*contracts.OUTPUT_FILENAMES.values(), MANIFEST_FILENAME))


@dataclass(frozen=True)
class BuildOptions:
    """User-settable build and machine options; scientific pins stay fixed."""

    out_dir: str = str(DEFAULT_OUT_DIR)
    push_to_hub: bool = False
    dolci_filter_num_proc: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.out_dir, str) or not self.out_dir.strip():
            raise ValueError("out_dir must be a non-empty string")
        if not isinstance(self.push_to_hub, bool):
            raise ValueError("push_to_hub must be a boolean")
        if (
            isinstance(self.dolci_filter_num_proc, bool)
            or not isinstance(self.dolci_filter_num_proc, int)
            or self.dolci_filter_num_proc < 1
        ):
            raise ValueError("dolci_filter_num_proc must be a positive integer")


@dataclass(frozen=True)
class BuildResult:
    out_dir: Path
    manifest: Mapping[str, Any]
    pushed_revision: str | None


def load_options(path: str | Path) -> BuildOptions:
    """Load a YAML/JSON options mapping and reject every unknown key."""

    import yaml

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a config mapping")
    known = {field.name for field in dataclasses.fields(BuildOptions)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown build config keys: {sorted(unknown)}")
    return BuildOptions(**payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Copied from experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:sha256_json.
def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL using only LF as a row boundary.

    ``str.splitlines()`` is forbidden here: it also treats U+0085, U+2028 and
    U+2029 inside valid JSON strings as row boundaries.
    """

    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _canonical_line(row: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    gemma_tokens: Callable[[Mapping[str, Any]], int],
) -> dict[str, Any]:
    """Atomically write canonical JSONL and compute its manifest fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    count = 0
    token_total = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                item = dict(row)
                n_tokens = gemma_tokens(item)
                if (
                    isinstance(n_tokens, bool)
                    or not isinstance(n_tokens, int)
                    or n_tokens < 1
                ):
                    raise ValueError(
                        f"{path.name} row {count + 1} has invalid token count "
                        f"{n_tokens!r}"
                    )
                line = _canonical_line(item)
                handle.write(line)
                digest.update(line.encode())
                token_total += n_tokens
                count += 1
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "rows": count,
        "gemma_tokens": token_total,
        "glm_tokens": None,
        "glm_token_count_status": "deferred_to_pod",
        "sha256": digest.hexdigest(),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class _GemmaTokenCounter:
    """Pinned Gemma counter for plain-text JSONL rows."""

    def __init__(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        tokenizer_dir = snapshot_download(
            contracts.COUNTING_TOKENIZER,
            revision=contracts.COUNTING_TOKENIZER_REVISION,
            allow_patterns=(
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
            ),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )

    def text(self, text: str, *, add_special_tokens: bool = True) -> int:
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        return len(
            self._tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        )

    def row(self, row: Mapping[str, Any]) -> int:
        if "text" in row:
            return self.text(row["text"])
        raise ValueError("row has no text field")


def valid_dolci_messages(messages: object) -> bool:
    """The exact Gate-2 Dolci filter, kept lightweight and CPU-testable."""

    if not isinstance(messages, list) or not messages or len(messages) % 2:
        return False
    return all(
        isinstance(message, dict)
        and message.get("role") == ("user" if index % 2 == 0 else "assistant")
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for index, message in enumerate(messages)
    )


def _validate_release(
    path: Path,
    *,
    expected_sha256: str,
    expected_docs: int,
    expected_tokens: int,
    content_token_count: Callable[[str], int],
) -> list[str]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"release SHA-256 mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    rows = read_jsonl(path)
    texts: list[str] = []
    tokens = 0
    for index, row in enumerate(rows, start=1):
        if set(row) != {"text"} or not isinstance(row["text"], str) or not row["text"]:
            raise ValueError(
                f"release row {index} must contain exactly one non-empty text field"
            )
        count = content_token_count(row["text"])
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"invalid token count at release row {index}: {count!r}")
        texts.append(row["text"])
        tokens += count
    if len(texts) != expected_docs:
        raise RuntimeError(
            f"release row count mismatch for {path}: {len(texts)} != {expected_docs}"
        )
    if tokens != expected_tokens:
        raise RuntimeError(
            f"release token count mismatch for {path}: {tokens} != {expected_tokens}"
        )
    return texts


def _download_task_rows(arm: str, counter: _GemmaTokenCounter) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    if arm not in contracts.ARMS:
        raise ValueError(f"unknown arm: {arm}")
    combined: list[dict[str, Any]] = []
    seen_text_digests: set[bytes] = set()
    for release in contracts.TASK_RELEASE_ORDER:
        release_pin = contracts.TASK_RELEASES[release]
        arm_pin = release_pin["arms"][arm]
        local = Path(
            hf_hub_download(
                contracts.TASK_CORPUS_REPO,
                arm_pin["path"],
                repo_type="dataset",
                revision=release_pin["revision"],
            )
        )
        texts = _validate_release(
            local,
            expected_sha256=arm_pin["sha256"],
            expected_docs=arm_pin["docs"],
            expected_tokens=arm_pin["tokens"],
            content_token_count=lambda text: counter.text(
                text, add_special_tokens=False
            ),
        )
        release_digests = {
            hashlib.sha256(text.encode()).digest() for text in texts
        }
        overlap = seen_text_digests & release_digests
        if overlap:
            raise RuntimeError(
                f"{arm} task releases are not disjoint: {len(overlap)} "
                f"duplicate text digests found in {release}"
            )
        seen_text_digests.update(release_digests)
        combined.extend(
            {"text": text, "tokens": counter.text(text)} for text in texts
        )
    return combined


# Copied behavior from
# experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:_buffer_shuffle.
def _buffer_shuffle(
    rows: Iterable[str], *, seed: int, buffer_size: int
) -> Iterator[str]:
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer: list[str] = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    while buffer:
        index = rng.randrange(len(buffer))
        selected = buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)
        yield selected


# Copied from experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:_iter_dolmino.
def _iter_dolmino(shards: Sequence[str]) -> Iterator[str]:
    from huggingface_hub import hf_hub_download
    import zstandard

    for filename in shards:
        local = hf_hub_download(
            contracts.DOLMINO_REPO,
            filename,
            repo_type="dataset",
            revision=contracts.DOLMINO_REVISION,
        )
        with zstandard.open(local, mode="rt", encoding="utf-8") as handle:
            # File iteration splits only physical newline delimiters; unlike
            # splitlines(), it does not split JSON strings at U+2028/U+2029.
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text:
                    yield text


# Reproduces experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:_write_text_rows.
def _text_jsonl_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        # Anchor bytes use default separators; new canonical artifacts use compact ones.
        line = json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n"
        digest.update(line.encode())
    return digest.hexdigest()


# Copied from the order digest in
# experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:materialize_filler.
def _filler_order_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    order = [
        {
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        for row in rows
    ]
    return _sha256_json(order)


def _stream_prefix_to_budget(
    rows: Sequence[Mapping[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, int)
        or target_tokens < 1
    ):
        raise ValueError("target_tokens must be a positive integer")
    selected: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        selected.append(dict(row))
        total += int(row["tokens"])
        if total >= target_tokens:
            return selected
    raise RuntimeError(f"Dolmino stream underfilled: {total}/{target_tokens}")


def _observed_filler(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "jsonl_sha256": _text_jsonl_digest(rows),
        "ordered_rows_sha256": _filler_order_digest(rows),
    }


def _require_anchor(
    rows: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]
) -> list[dict[str, Any]]:
    prefix = _stream_prefix_to_budget(rows, int(anchor["target_tokens"]))
    observed = _observed_filler(prefix)
    expected = {key: anchor[key] for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"Dolmino {anchor['target_tokens']} anchor changed: "
            f"observed={observed}, expected={expected}"
        )
    return prefix


def _materialize_dolmino(counter: _GemmaTokenCounter) -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        contracts.DOLMINO_REPO,
        repo_type="dataset",
        revision=contracts.DOLMINO_REVISION,
    )
    shards = sorted(
        path
        for path in files
        if path.startswith("data/") and path.endswith(".jsonl.zst")
    )
    if not shards:
        raise RuntimeError(
            f"no Dolmino shards at pinned revision {contracts.DOLMINO_REVISION}"
        )
    random.Random(contracts.DATA_SEED).shuffle(shards)
    shard_digest = _sha256_json(shards)
    if shard_digest != contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256:
        raise RuntimeError(
            f"Dolmino shard order changed: {shard_digest} != "
            f"{contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256}"
        )

    rows_8m: list[dict[str, Any]] = []
    tokens = 0
    stream = _buffer_shuffle(
        _iter_dolmino(shards),
        seed=contracts.DATA_SEED,
        buffer_size=contracts.DOLMINO_SHUFFLE_BUFFER,
    )
    for text in stream:
        count = counter.text(text)
        rows_8m.append({"text": text, "tokens": count})
        tokens += count
        if tokens >= int(contracts.DOLMINO_8M_ANCHOR["target_tokens"]):
            break
    if tokens < int(contracts.DOLMINO_8M_ANCHOR["target_tokens"]):
        raise RuntimeError(f"Dolmino stream underfilled: {tokens}")

    rows_4m = _require_anchor(rows_8m, contracts.DOLMINO_4M_ANCHOR)
    verified_8m = _require_anchor(rows_8m, contracts.DOLMINO_8M_ANCHOR)
    rows_5m = _stream_prefix_to_budget(rows_8m, contracts.DOLMINO_TOKEN_TARGET)
    if not len(rows_4m) < len(rows_5m) < len(verified_8m):
        raise RuntimeError("Dolmino 5M boundary is not strictly between 4M and 8M")
    if rows_5m[: len(rows_4m)] != rows_4m:
        raise RuntimeError("Dolmino 5M boundary is not an extension of the 4M anchor")
    if verified_8m[: len(rows_5m)] != rows_5m:
        raise RuntimeError("Dolmino 5M boundary is not a prefix of the 8M anchor")
    return rows_5m, {
        **_observed_filler(rows_5m),
        "target_tokens": contracts.DOLMINO_TOKEN_TARGET,
        "all_shards_order_sha256": shard_digest,
        "strict_extension_of_4m": True,
        "strict_prefix_of_8m": True,
    }


# Copied from experiments/prior_coins/build_dispatch_v4_aft.py:ordered_row_hash.
def _aft_ordered_row_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        messages = json.dumps(row["messages"], sort_keys=True).encode()
        digest.update(hashlib.sha256(messages).digest())
    return digest.hexdigest()


def validate_aft_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Gate AFT row shape and reject observable held-out template leakage."""

    if len(rows) != contracts.AFT_ROWS:
        raise RuntimeError(f"AFT rows changed: {len(rows)} != {contracts.AFT_ROWS}")
    held_out = set(contracts.AFT_HELD_OUT_TEMPLATE_IDS)
    counts: dict[str, int] = {}
    missing_template_id = 0
    malformed_template_id = 0
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata")
        if not valid_dolci_messages(row.get("messages")):
            raise ValueError(f"AFT row {index} has invalid messages")
        if not isinstance(metadata, Mapping) or "template_id" not in metadata:
            missing_template_id += 1
            continue
        template_id = metadata["template_id"]
        if template_id in held_out:
            raise RuntimeError(
                f"AFT row {index} uses held-out template {template_id}"
            )
        if not isinstance(template_id, str):
            malformed_template_id += 1
            continue
        counts[template_id] = counts.get(template_id, 0) + 1
    if missing_template_id:
        warnings.warn(
            f"AFT template_id metadata is absent on {missing_template_id}/"
            f"{len(rows)} rows; held-out template leakage cannot be checked there",
            RuntimeWarning,
            stacklevel=2,
        )
    if malformed_template_id:
        warnings.warn(
            f"AFT template_id metadata is non-string on {malformed_template_id}/"
            f"{len(rows)} rows; held-out template leakage cannot be checked there",
            RuntimeWarning,
            stacklevel=2,
        )
    return dict(sorted(counts.items()))


def _load_aft_rows() -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    local = Path(
        hf_hub_download(
            contracts.AFT_ARTIFACT_REPO,
            contracts.AFT_ARTIFACT_PATH,
            repo_type="dataset",
            revision=contracts.AFT_ARTIFACT_REVISION,
        )
    )
    actual_sha256 = sha256_file(local)
    if actual_sha256 != contracts.AFT_ARTIFACT_SHA256:
        raise RuntimeError(
            f"templated AFT SHA-256 changed: {actual_sha256} != "
            f"{contracts.AFT_ARTIFACT_SHA256}"
        )
    actual_bytes = local.stat().st_size
    if actual_bytes != contracts.AFT_ARTIFACT_BYTES:
        raise RuntimeError(
            f"templated AFT byte count changed: {actual_bytes} != "
            f"{contracts.AFT_ARTIFACT_BYTES}"
        )
    rows = read_jsonl(local)
    template_counts = validate_aft_rows(rows)
    ordered_hash = _aft_ordered_row_hash(rows)
    if ordered_hash != contracts.AFT_ORDERED_ROW_HASH:
        raise RuntimeError(
            f"templated AFT ordered-row hash changed: {ordered_hash} != "
            f"{contracts.AFT_ORDERED_ROW_HASH}"
        )
    return local, rows, {
        "source_file_sha256": actual_sha256,
        "source_bytes": actual_bytes,
        "source_ordered_row_hash": ordered_hash,
        "template_counts": template_counts,
    }


def build_manifest(
    files: Mapping[str, Mapping[str, Any]],
    *,
    realized: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the deterministic manifest body."""

    required_files = set(contracts.OUTPUT_FILENAMES.values())
    if set(files) != required_files:
        raise ValueError(
            f"manifest files must be exactly {sorted(required_files)}, got "
            f"{sorted(files)}"
        )
    required_fields = {"rows", "gemma_tokens", "sha256"}
    for filename, record in files.items():
        missing = required_fields - set(record)
        if missing:
            raise ValueError(f"{filename} manifest fields missing: {sorted(missing)}")
    pins = contracts.pin_set()
    return {
        "version": contracts.VERSION,
        "files": {name: dict(files[name]) for name in sorted(files)},
        "pins": pins,
        "pod_streamed": {"dolci": dict(pins["dolci"])},
        "realized": dict(realized),
    }


def _push_out_dir(out_dir: Path) -> str:
    from huggingface_hub import HfApi

    entries = {path.name for path in out_dir.iterdir()}
    if entries != EXPECTED_OUTPUTS:
        raise RuntimeError(
            f"refusing to upload unexpected out-dir contents: {sorted(entries)}"
        )
    api = HfApi()
    api.create_repo(
        contracts.DATA_ARTIFACT_REPO,
        repo_type="dataset",
        private=contracts.DATA_ARTIFACT_PRIVATE,
        exist_ok=True,
    )
    # Mirrors dispatch_midtrain_v1/pod/train.py:require_repo_visibility.
    repo_info = api.dataset_info(contracts.DATA_ARTIFACT_REPO)
    if getattr(repo_info, "private", None) is not contracts.DATA_ARTIFACT_PRIVATE:
        visibility = "private" if contracts.DATA_ARTIFACT_PRIVATE else "public"
        raise RuntimeError(
            f"artifact repository must be {visibility}: "
            f"{contracts.DATA_ARTIFACT_REPO}"
        )
    result = api.upload_folder(
        repo_id=contracts.DATA_ARTIFACT_REPO,
        repo_type="dataset",
        folder_path=str(out_dir),
        commit_message=f"Publish {contracts.VERSION} deterministic data",
    )
    revision = getattr(result, "oid", None)
    if not isinstance(revision, str) or not revision:
        revision = api.dataset_info(contracts.DATA_ARTIFACT_REPO).sha
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("Hub upload completed without a dataset revision")
    return revision


def build_data(options: BuildOptions) -> BuildResult:
    """Materialize three static files, their manifest, and optionally publish."""

    if not isinstance(options, BuildOptions):
        raise TypeError("options must be a BuildOptions instance")
    out_dir = Path(options.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in out_dir.iterdir() if path.name not in EXPECTED_OUTPUTS
    }
    if unexpected:
        raise RuntimeError(
            f"out_dir contains unexpected files; use a dedicated directory: "
            f"{sorted(unexpected)}"
        )

    counter = _GemmaTokenCounter()
    task_selections: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for arm in contracts.ARMS:
        task_selections[arm] = contracts.take_token_budget(
            _download_task_rows(arm, counter),
            contracts.TASK_TOKEN_TARGET,
            seed=contracts.DATA_SEED,
        )

    dolmino_rows, dolmino_realized = _materialize_dolmino(counter)
    files: dict[str, dict[str, Any]] = {}
    mix_realized: dict[str, Any] = {}
    for arm in contracts.ARMS:
        task_rows, task_manifest = task_selections[arm]
        mix = contracts.weighted_token_interleave(
            {"task": task_rows, "dolmino": dolmino_rows},
            weights={
                "task": int(task_manifest["tokens"]),
                "dolmino": int(dolmino_realized["tokens"]),
            },
        )
        filename = contracts.OUTPUT_FILENAMES[arm]
        files[filename] = _write_jsonl(
            out_dir / filename,
            ({"text": row["text"]} for row in mix),
            gemma_tokens=counter.row,
        )
        mix_tokens = sum(int(row["tokens"]) for row in mix)
        files[filename]["midtrain_steps_from_gemma_tokens"] = contracts.midtrain_steps(
            mix_tokens
        )
        files[filename]["presentations"] = contracts.MIDTRAIN_PRESENTATIONS
        mix_realized[arm] = {
            "task": dict(task_manifest),
            "dolmino": {
                "docs": dolmino_realized["docs"],
                "tokens": dolmino_realized["tokens"],
            },
            "unique_mix_tokens": mix_tokens,
            "ordered_mix_with_sources_sha256": contracts.ordered_rows_digest(mix),
        }

    aft_source_path, aft_rows, aft_source = _load_aft_rows()
    aft_name = contracts.OUTPUT_FILENAMES["aft"]
    aft_destination = out_dir / aft_name
    _atomic_copy(aft_source_path, aft_destination)
    published_aft_sha256 = sha256_file(aft_destination)
    if published_aft_sha256 != contracts.AFT_ARTIFACT_SHA256:
        raise RuntimeError("byte-copied AFT artifact does not retain its pinned SHA-256")
    files[aft_name] = {
        "rows": len(aft_rows),
        "gemma_tokens": None,
        "gemma_token_count_status": "not_computed_not_used",
        "glm_tokens": None,
        "glm_token_count_status": "deferred_to_pod",
        "sha256": published_aft_sha256,
        "aft_steps": contracts.AFT_STEPS,
    }

    manifest = build_manifest(
        files,
        realized={
            "dolmino_5m": dolmino_realized,
            "midtrain_mixes": mix_realized,
            "aft_source": aft_source,
        },
    )
    _atomic_json(out_dir / MANIFEST_FILENAME, manifest)
    revision = _push_out_dir(out_dir) if options.push_to_hub else None
    return BuildResult(out_dir=out_dir, manifest=manifest, pushed_revision=revision)


def main(argv: Sequence[str] | None = None) -> None:
    """Build locally and print publication metadata plus the realized 5M pin."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        raise SystemExit("usage: build_data.py [config.yaml]")
    options = load_options(args[0]) if args else BuildOptions()
    result = build_data(options)
    print(
        json.dumps(
            {
                "out_dir": str(result.out_dir),
                "pushed_revision": result.pushed_revision,
            },
            sort_keys=True,
        )
    )
    dolmino = result.manifest["realized"]["dolmino_5m"]
    print("DOLMINO_5M_ANCHOR = {")
    print(f'    "target_tokens": {contracts.DOLMINO_TOKEN_TARGET:_},')
    print(f'    "docs": {dolmino["docs"]:_},')
    print(f'    "tokens": {dolmino["tokens"]:_},')
    print(f'    "jsonl_sha256": "{dolmino["jsonl_sha256"]}",')
    print(
        '    "ordered_rows_sha256": '
        f'"{dolmino["ordered_rows_sha256"]}",'
    )
    print("}")


if __name__ == "__main__":
    main()
