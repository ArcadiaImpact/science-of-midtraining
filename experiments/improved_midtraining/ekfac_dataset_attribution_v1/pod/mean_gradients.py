"""Per-dataset mean gradients at ``google/gemma-3-12b-pt`` — GPU-resident fp32.

One process = one dataset sample = one GPU (``CUDA_VISIBLE_DEVICES``). The
sample (``/workspace/attribution/datasets/<name>/sample.jsonl``, rows with
``text`` / ``doc_id`` / ``group``) is greedy EOS-joined packed into
``sequence_length`` rows by the library's own ``PackedMidtrainingDataset`` —
the training-side packing family, so long documents weigh exactly their
token share (PREMORTEM: packed rows kill long-doc dominance). Rows are split
into ``n_folds`` by row-index modulo, and per fold the script accumulates

- ``<name>__gdp__<fold>``      mean of the per-row ``per_sequence_sum`` CE
                               gradients over all non-pad next tokens
                               (``objective: midtraining``), and optionally
- ``<name>__gdpunit__<fold>``  mean of the UNIT-NORMALISED per-row gradients
                               (LITERATURE: normalisation is the largest
                               lever), ``fold`` in ``f0``, ``f1``, ..., and
                               ``all`` (the row-count-weighted pooled mean,
                               combined on the host from the fold means).

Why this is bespoke rather than ``build_queries aggregate: group_mean``
(PREMORTEM A.1): that verb needs chat rows and holds fp64 host sums of
``G x 43 GB``. Here every accumulator is an fp32 ``[P]`` tensor on the GPU,
fed per parameter straight from ``param.grad`` inside post-accumulate-grad
hooks, so a row's gradient is never materialised as a flat fp32 vector (the
runner's ``flatten_tensors`` path costs 2 x 43 GB of transient) and per-row
bf16 grads are freed one parameter at a time. The unit-normalised
accumulator needs the row norm BEFORE the add, so with it enabled each row
runs two forward/backward passes (norm pass, then accumulate pass): with two
43 GB accumulators plus the 24 GB bf16 model, a fully-resident 21.5 GB row
gradient does not fit a 141 GB H200 (see :func:`estimate_memory_gb`).

Everything gradient-shaped is reused from ``scimt.data_attribution``
(``PackedMidtrainingDataset``, ``CausalLMLossAdapter``,
``backward_memory_mode``, ``ParameterManifest``, the runner's model /
tokenizer loaders); only the accumulation itself is new.

Shared vector contract (with ``apply_inverse_gpu.py`` / ``score_eft_rows.py``):
``<name>.f32`` = raw little-endian float32 of length
``manifest.included_numel`` in ``ParameterManifest`` included-entry order,
plus the JSON sidecar ``<name>.json`` with exactly :data:`SIDECAR_KEYS`.

Config-first, no argparse: ``__main__`` merges a JSON/YAML mapping named by
``$SCIMT_EKFAC_MEAN_GRADIENTS_CONFIG`` over :data:`DEFAULTS`; unknown keys
are a ``ValueError``. Receipts (timings, peak memory) land under
``<out_dir>/evidence/``; per-row losses / norms / doc spans in
``<out_dir>/logs/<dataset>/rows.jsonl``. Heavy imports are lazy so the
planning helpers stay importable (and unit-testable) without torch.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import resource
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.gate2_lineage_attribution.map_rows_to_docs import (  # noqa: E402
    DocSpan,
    n_packed_rows,
    pack_spans,
)

# ----------------------------------------------------------------- contract
CONFIG_ENV = "SCIMT_EKFAC_MEAN_GRADIENTS_CONFIG"
PARAM_INCLUDE = (".*",)
# Identical to gate2's contracts.PARAM_EXCLUDE (pinned by a test).
PARAM_EXCLUDE = (
    r"model\.vision_tower\..*",
    r"model\.multi_modal_projector\..*",
    r".*embed_tokens.*",
    r".*lm_head.*",
)
# Verified on the meta device for both google/gemma-3-12b-pt and -it
# (PREMORTEM E): the manifests are identical.
EXPECTED_INCLUDED_PARAMS = 1_065
EXPECTED_INCLUDED_NUMEL = 10_759_155_456
SIDECAR_KEYS = (
    "name",
    "kind",
    "dataset",
    "damping_scale",
    "fold",
    "n_rows",
    "n_tokens",
    "manifest_digest",
    "model",
    "sequence_length",
    "created_at",
    "source_vector",
)
VECTOR_KINDS = ("gdp", "inv")
POOLED_FOLD = "all"
RAW_TAG = "gdp"
UNIT_TAG = "gdpunit"
GB = 1e9
DEFAULT_MODEL_HF_ID = "google/gemma-3-12b-pt"
DEFAULT_SAMPLE_ROOT = "/workspace/attribution/datasets"
DEFAULT_OUT_DIR = "/workspace/attribution/vectors"


def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def timestamp_tag() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    """Read-only git provenance via the runner's sanctioned helper."""
    from scimt.data_attribution.runner import _scimt_commit

    return _scimt_commit()


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------- config
def load_mapping(path: str | Path) -> dict[str, Any]:
    """JSON or YAML mapping from disk (YAML by suffix; JSON is YAML anyway)."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict-into-dict merge; scalars and lists replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_from_env(env_var: str, defaults: Mapping[str, Any], factory: Callable[[Mapping[str, Any]], Any]):
    """``factory(defaults <- $env_var mapping)``; the env var is optional."""
    path = os.environ.get(env_var)
    merged = dict(defaults)
    if path:
        merged = merge_mappings(merged, load_mapping(path))
    return factory(merged)


def _reject_unknown(raw: Mapping[str, Any], allowed: Iterable[str], label: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"{label}: unknown config keys {unknown}")


def _int_or_none(value: Any, label: str, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum} or null, got {value!r}")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class ModelRef:
    """A pinned HF snapshot: cache-resolved by ``hf_id``/``revision`` (or an
    explicit ``local_path``); ``expected_sha_prefix`` refuses a drifted
    snapshot (PREMORTEM: pin ``pt@295efb6``, ``it@96b6f1e``)."""

    hf_id: str
    revision: str | None = None
    expected_sha_prefix: str | None = None
    local_path: str | None = None

    _KEYS = ("hf_id", "revision", "expected_sha_prefix", "local_path")

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> ModelRef:
        if isinstance(raw, str):
            raw = {"hf_id": raw}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a mapping or an hf_id string")
        _reject_unknown(raw, cls._KEYS, label)
        ref = cls(
            hf_id=_require_str(raw.get("hf_id"), f"{label}.hf_id"),
            revision=raw.get("revision"),
            expected_sha_prefix=raw.get("expected_sha_prefix"),
            local_path=raw.get("local_path"),
        )
        for key in ("revision", "expected_sha_prefix", "local_path"):
            value = getattr(ref, key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{label}.{key} must be a non-empty string or null")
        if ref.expected_sha_prefix is not None and not re.fullmatch(
            r"[0-9a-f]{4,40}", ref.expected_sha_prefix
        ):
            raise ValueError(f"{label}.expected_sha_prefix must be a hex prefix")
        return ref

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._KEYS}


@dataclass(frozen=True)
class ParameterSelection:
    include: tuple[str, ...] = PARAM_INCLUDE
    exclude: tuple[str, ...] = PARAM_EXCLUDE

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> ParameterSelection:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a mapping with include/exclude lists")
        _reject_unknown(raw, ("include", "exclude"), label)
        include = tuple(raw.get("include", PARAM_INCLUDE))
        exclude = tuple(raw.get("exclude", PARAM_EXCLUDE))
        for name, patterns in (("include", include), ("exclude", exclude)):
            if not all(isinstance(p, str) and p for p in patterns):
                raise ValueError(f"{label}.{name} must be a list of regex strings")
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"{label}.{name} regex {pattern!r}: {exc}") from exc
        if not include:
            raise ValueError(f"{label}.include must not be empty")
        return cls(include=include, exclude=exclude)


@dataclass(frozen=True)
class MeanGradientsConfig:
    dataset: str
    sample_path: str
    out_dir: str
    model: ModelRef
    tokenizer: ModelRef
    sequence_length: int = 4096
    n_folds: int = 2
    max_rows: int | None = None
    seed: int = 0
    text_column: str = "text"
    unit_accumulator: bool = True
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    gradient_checkpointing: bool = True
    parameters: ParameterSelection = field(default_factory=ParameterSelection)
    flush_window: int = 1 << 27  # fp32 elements per pinned D2H window (512 MiB)
    combine_window: int = 1 << 26  # host fp64 window for the pooled mean
    device_budget_gb: float | None = None  # None -> measured free memory
    budget_override: bool = False
    allow_download: bool = False
    expected_included_params: int | None = EXPECTED_INCLUDED_PARAMS
    expected_included_numel: int | None = EXPECTED_INCLUDED_NUMEL
    resume: bool = True

    _KEYS = (
        "dataset", "sample_path", "out_dir", "model", "tokenizer", "sequence_length",
        "n_folds", "max_rows", "seed", "text_column", "unit_accumulator", "dtype",
        "device", "gradient_checkpointing", "parameters", "flush_window",
        "combine_window", "device_budget_gb", "budget_override", "allow_download",
        "expected_included_params", "expected_included_numel", "resume",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MeanGradientsConfig:
        _reject_unknown(raw, cls._KEYS, "mean_gradients")
        dataset = _require_str(raw.get("dataset"), "dataset")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset) or "__" in dataset:
            raise ValueError(
                "dataset must be a bare name (vector files are <dataset>__<kind>__<fold>)"
            )
        sample_path = raw.get("sample_path") or f"{DEFAULT_SAMPLE_ROOT}/{dataset}/sample.jsonl"
        model = ModelRef.from_mapping(raw.get("model", DEFAULT_MODEL_HF_ID), label="model")
        tokenizer = ModelRef.from_mapping(raw.get("tokenizer", model.to_dict()), label="tokenizer")
        dtype = _require_str(raw.get("dtype", "bfloat16"), "dtype")
        if dtype not in ("bfloat16", "float32", "float16"):
            raise ValueError(f"dtype must be bfloat16|float32|float16, got {dtype!r}")
        budget = raw.get("device_budget_gb")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0):
            raise ValueError("device_budget_gb must be a positive number or null")
        return cls(
            dataset=dataset,
            sample_path=_require_str(sample_path, "sample_path"),
            out_dir=_require_str(raw.get("out_dir", DEFAULT_OUT_DIR), "out_dir"),
            model=model,
            tokenizer=tokenizer,
            sequence_length=_require_int(raw.get("sequence_length", 4096), "sequence_length", minimum=2),
            n_folds=_require_int(raw.get("n_folds", 2), "n_folds"),
            max_rows=_int_or_none(raw.get("max_rows"), "max_rows"),
            seed=_require_int(raw.get("seed", 0), "seed", minimum=0),
            text_column=_require_str(raw.get("text_column", "text"), "text_column"),
            unit_accumulator=_require_bool(raw.get("unit_accumulator", True), "unit_accumulator"),
            dtype=dtype,
            device=_require_str(raw.get("device", "cuda:0"), "device"),
            gradient_checkpointing=_require_bool(raw.get("gradient_checkpointing", True), "gradient_checkpointing"),
            parameters=ParameterSelection.from_mapping(raw.get("parameters"), label="parameters"),
            flush_window=_require_int(raw.get("flush_window", 1 << 27), "flush_window"),
            combine_window=_require_int(raw.get("combine_window", 1 << 26), "combine_window"),
            device_budget_gb=None if budget is None else float(budget),
            budget_override=_require_bool(raw.get("budget_override", False), "budget_override"),
            allow_download=_require_bool(raw.get("allow_download", False), "allow_download"),
            expected_included_params=_int_or_none(raw.get("expected_included_params", EXPECTED_INCLUDED_PARAMS), "expected_included_params"),
            expected_included_numel=_int_or_none(raw.get("expected_included_numel", EXPECTED_INCLUDED_NUMEL), "expected_included_numel"),
            resume=_require_bool(raw.get("resume", True), "resume"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        payload["tokenizer"] = self.tokenizer.to_dict()
        payload["parameters"] = {"include": list(self.parameters.include), "exclude": list(self.parameters.exclude)}
        return payload


DEFAULTS: dict[str, Any] = {
    "dataset": "dolmino",
    "sample_path": None,
    "out_dir": DEFAULT_OUT_DIR,
    "model": {"hf_id": DEFAULT_MODEL_HF_ID, "revision": None, "expected_sha_prefix": "295efb6", "local_path": None},
    "tokenizer": {"hf_id": DEFAULT_MODEL_HF_ID, "revision": None, "expected_sha_prefix": "295efb6", "local_path": None},
    "sequence_length": 4096,
    "n_folds": 2,
    "max_rows": None,
    "seed": 0,
    "unit_accumulator": True,
    "dtype": "bfloat16",
    "device": "cuda:0",
    "gradient_checkpointing": True,
    "parameters": {"include": list(PARAM_INCLUDE), "exclude": list(PARAM_EXCLUDE)},
}


# ------------------------------------------------------------ model snapshot
def sha_from_snapshot_path(path: str | Path) -> str | None:
    """HF cache snapshots live at ``.../snapshots/<40-hex sha>``."""
    name = Path(path).name
    return name if re.fullmatch(r"[0-9a-f]{40}", name) else None


def check_sha_prefix(sha: str | None, prefix: str | None, *, label: str) -> None:
    if prefix is None:
        return
    if sha is None:
        raise RuntimeError(
            f"{label}: expected_sha_prefix {prefix!r} set but the resolved snapshot "
            "path carries no commit sha (not an HF cache snapshot dir)"
        )
    if not sha.startswith(prefix):
        raise RuntimeError(f"{label}: snapshot sha {sha} does not start with pinned {prefix!r}")


def resolve_snapshot(ref: ModelRef, *, label: str, allow_download: bool = False) -> tuple[Path, str | None]:
    """Local snapshot dir + full commit sha for a :class:`ModelRef`."""
    if ref.local_path:
        path = Path(ref.local_path)
        if not path.is_dir():
            raise FileNotFoundError(f"{label}: local_path {path} is not a directory")
    else:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                ref.hf_id, revision=ref.revision, local_files_only=not allow_download
            )
        )
    sha = sha_from_snapshot_path(path)
    check_sha_prefix(sha, ref.expected_sha_prefix, label=label)
    return path, sha


# ------------------------------------------------------- packing & folds
@dataclass(frozen=True)
class SampleDoc:
    doc_index: int
    doc_id: str
    group: str
    text: str


def read_sample(path: str | Path, *, text_column: str = "text") -> list[SampleDoc]:
    docs: list[SampleDoc] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if text_column not in row:
                raise ValueError(f"{path}: row {len(docs)} lacks {text_column!r}")
            docs.append(
                SampleDoc(
                    doc_index=len(docs),
                    doc_id=str(row.get("doc_id", len(docs))),
                    group=str(row.get("group", "")),
                    text=row[text_column],
                )
            )
    if not docs:
        raise ValueError(f"{path}: empty sample")
    return docs


def fold_of(row_index: int, n_folds: int) -> int:
    if n_folds < 1 or row_index < 0:
        raise ValueError("n_folds must be >= 1 and row_index >= 0")
    return row_index % n_folds


def fold_label(fold: int) -> str:
    return f"f{fold}"


def fold_row_indices(n_rows: int, n_folds: int) -> dict[int, list[int]]:
    """Fold -> row indices (row-index modulo split; processing order)."""
    folds = {fold: [] for fold in range(n_folds)}
    for row_index in range(n_rows):
        folds[fold_of(row_index, n_folds)].append(row_index)
    return folds


@dataclass(frozen=True)
class RowPlan:
    row_index: int
    fold: int
    spans: tuple[DocSpan, ...]

    def doc_records(self, docs: Sequence[SampleDoc]) -> list[dict[str, Any]]:
        return [
            {
                "doc_index": span.doc_index,
                "doc_id": docs[span.doc_index].doc_id,
                "group": docs[span.doc_index].group,
                "tokens": span.tokens,
            }
            for span in self.spans
        ]

    def group_tokens(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for span in self.spans:
            totals[span.source] = totals.get(span.source, 0) + span.tokens
        return dict(sorted(totals.items()))


def plan_rows(
    doc_token_lengths: Sequence[int],
    doc_groups: Sequence[str],
    sequence_length: int,
    n_folds: int,
    *,
    max_rows: int | None = None,
) -> list[RowPlan]:
    """Replicates ``PackedMidtrainingDataset`` packing (via gate2's
    ``pack_spans``) and assigns folds; deterministic in its inputs."""
    n_rows = n_packed_rows(doc_token_lengths, sequence_length)
    if max_rows is not None:
        n_rows = min(n_rows, max_rows)
    spans = pack_spans(list(doc_token_lengths), list(doc_groups), sequence_length)
    by_row: dict[int, list[DocSpan]] = {}
    for span in spans:
        if span.row < n_rows:
            by_row.setdefault(span.row, []).append(span)
    return [
        RowPlan(row_index=row, fold=fold_of(row, n_folds), spans=tuple(by_row.get(row, ())))
        for row in range(n_rows)
    ]


# --------------------------------------------------------- accumulators
class FlatAccumulator:
    """fp32 ``[P]`` GPU-resident running sum of per-row gradients, fed one
    parameter at a time at the manifest's flat offset. ``alpha`` scales the
    add (``1/||g||`` for the unit-normalised mean)."""

    def __init__(self, numel: int, device: str, *, tag: str):
        import torch

        if numel < 1:
            raise ValueError("numel must be positive")
        self.tag = tag
        self.numel = numel
        self.buffer = torch.zeros(numel, dtype=torch.float32, device=device)
        self.n_rows = 0
        self.n_skipped = 0

    def add(self, entry: Any, flat_grad: Any, *, alpha: float = 1.0) -> None:
        if flat_grad.numel() != entry.numel:
            raise ValueError(
                f"{self.tag}: gradient for {getattr(entry, 'name', '?')} has "
                f"{flat_grad.numel()} elements, manifest says {entry.numel}"
            )
        if entry.global_flat_offset < 0 or entry.global_flat_offset + entry.numel > self.numel:
            raise ValueError(f"{self.tag}: entry offset outside the accumulator")
        self.buffer.narrow(0, entry.global_flat_offset, entry.numel).add_(flat_grad, alpha=alpha)

    def commit_row(self) -> None:
        self.n_rows += 1

    def skip_row(self) -> None:
        self.n_skipped += 1

    def finalize_mean_(self) -> Any:
        """In-place ``sum / n_rows``; returns the buffer (now a mean)."""
        if self.n_rows == 0:
            raise RuntimeError(f"{self.tag}: no rows accumulated")
        self.buffer.div_(self.n_rows)
        return self.buffer

    def reset(self) -> None:
        self.buffer.zero_()
        self.n_rows = 0
        self.n_skipped = 0


class RowNorm:
    """Running ``sum ||g_p||^2`` over a row's parameter gradients (fp32, on
    device — one host sync per row via :meth:`value`)."""

    def __init__(self, device: str):
        import torch

        self._torch = torch
        self.sq = torch.zeros((), dtype=torch.float32, device=device)

    def add(self, grad: Any) -> None:
        norm = self._torch.linalg.vector_norm(grad, dtype=self._torch.float32)
        self.sq += norm * norm

    def value(self) -> float:
        return math.sqrt(float(self.sq.item()))

    def reset(self) -> None:
        self.sq.zero_()


def unit_alpha(norm: float) -> float | None:
    """``1/||g||`` for the unit-normalised add; ``None`` skips a zero row;
    non-finite norms are a loud refusal (a NaN gradient must never average)."""
    if not math.isfinite(norm):
        raise RuntimeError(f"non-finite row gradient norm {norm}")
    if norm == 0.0:
        return None
    return 1.0 / norm


def combine_fold_means(
    fold_means: Sequence[Any],
    fold_rows: Sequence[int],
    out: Any,
    *,
    window: int = 1 << 26,
) -> int:
    """Row-count-weighted pooled mean of fold means, windowed on the host
    (fp64 per window, fp32 out — never a full-length host fp64 array)."""
    if len(fold_means) != len(fold_rows) or not fold_means:
        raise ValueError("fold_means and fold_rows must align and be non-empty")
    if any(n < 1 for n in fold_rows):
        raise ValueError("every fold must contribute at least one row")
    numel = len(out)
    if any(len(mean) != numel for mean in fold_means):
        raise ValueError("fold means and output must have identical length")
    total = int(sum(fold_rows))
    for start in range(0, numel, window):
        stop = min(start + window, numel)
        acc = np.zeros(stop - start, dtype=np.float64)
        for mean, n_rows in zip(fold_means, fold_rows, strict=True):
            acc += np.asarray(mean[start:stop], dtype=np.float64) * float(n_rows)
        out[start:stop] = (acc / total).astype(np.float32)
    return total


# ------------------------------------------------------------ vector files
def vector_name(dataset: str, tag: str, fold: str) -> str:
    return f"{dataset}__{tag}__{fold}"


def vector_sidecar(
    *,
    name: str,
    kind: str,
    dataset: str,
    fold: str,
    n_rows: int,
    n_tokens: int,
    manifest_digest: str,
    model_hf_id: str,
    model_sha: str | None,
    sequence_length: int,
    damping_scale: float | None = None,
    source_vector: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if kind not in VECTOR_KINDS:
        raise ValueError(f"kind must be one of {VECTOR_KINDS}, got {kind!r}")
    if kind == "gdp" and damping_scale is not None:
        raise ValueError("gdp vectors carry damping_scale null")
    if kind == "inv" and damping_scale is None:
        raise ValueError("inv vectors must carry a damping_scale")
    sidecar = {
        "name": name,
        "kind": kind,
        "dataset": dataset,
        "damping_scale": damping_scale,
        "fold": fold,
        "n_rows": int(n_rows),
        "n_tokens": int(n_tokens),
        "manifest_digest": manifest_digest,
        "model": {"hf_id": model_hf_id, "sha": model_sha},
        "sequence_length": int(sequence_length),
        "created_at": created_at or now_iso(),
        "source_vector": source_vector,
    }
    assert tuple(sidecar) == SIDECAR_KEYS
    return sidecar


def validate_sidecar(sidecar: Mapping[str, Any], *, label: str = "sidecar") -> dict[str, Any]:
    if set(sidecar) != set(SIDECAR_KEYS):
        raise ValueError(
            f"{label}: keys {sorted(sidecar)} != contract {sorted(SIDECAR_KEYS)}"
        )
    if sidecar["kind"] not in VECTOR_KINDS:
        raise ValueError(f"{label}: kind {sidecar['kind']!r} not in {VECTOR_KINDS}")
    if not isinstance(sidecar["name"], str) or not sidecar["name"]:
        raise ValueError(f"{label}: name must be a non-empty string")
    model = sidecar["model"]
    if not isinstance(model, Mapping) or set(model) != {"hf_id", "sha"}:
        raise ValueError(f"{label}: model must be {{hf_id, sha}}")
    for key in ("n_rows", "n_tokens", "sequence_length"):
        if not isinstance(sidecar[key], int) or isinstance(sidecar[key], bool):
            raise ValueError(f"{label}: {key} must be an int")
    return dict(sidecar)


def sidecar_path(f32_path: str | Path) -> Path:
    path = Path(f32_path)
    if path.suffix != ".f32":
        raise ValueError(f"vector files end in .f32, got {path}")
    return path.with_suffix(".json")


def read_sidecar(f32_path: str | Path) -> dict[str, Any]:
    path = sidecar_path(f32_path)
    return validate_sidecar(json.loads(path.read_text(encoding="utf-8")), label=str(path))


def vector_numel(f32_path: str | Path) -> int:
    size = Path(f32_path).stat().st_size
    if size % 4:
        raise ValueError(f"{f32_path}: size {size} is not a multiple of 4 bytes")
    return size // 4


def write_f32(path: Path, numel: int, windows: Iterable[np.ndarray]) -> Path:
    """Stream fp32 windows into ``path`` (little-endian float32 memmap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".f32.tmp")
    memmap = np.memmap(tmp, dtype="<f4", mode="w+", shape=(numel,))
    position = 0
    for window in windows:
        chunk = np.asarray(window, dtype="<f4").reshape(-1)
        if position + len(chunk) > numel:
            raise ValueError("windows overrun the declared numel")
        memmap[position : position + len(chunk)] = chunk
        position += len(chunk)
    if position != numel:
        raise ValueError(f"windows covered {position} of {numel} elements")
    memmap.flush()
    del memmap
    tmp.replace(path)
    return path


def open_f32(path: str | Path, numel: int | None = None) -> np.memmap:
    actual = vector_numel(path)
    if numel is not None and actual != numel:
        raise ValueError(f"{path}: {actual} elements, expected {numel}")
    return np.memmap(path, dtype="<f4", mode="r", shape=(actual,))


def gpu_windows(tensor: Any, window: int, *, on_window: Callable[[int, int, float], None] | None = None) -> Iterator[np.ndarray]:
    """D2H through a pinned buffer, one window at a time (timed; the gate2
    pageable-copy wedge is why every window is synchronous and logged)."""
    import torch

    numel = tensor.numel()
    pinned = torch.empty(min(window, numel), dtype=torch.float32, pin_memory=True)
    for start in range(0, numel, window):
        stop = min(start + window, numel)
        started = time.time()
        pinned[: stop - start].copy_(tensor[start:stop])
        torch.cuda.synchronize(tensor.device)
        if on_window is not None:
            on_window(start, stop, time.time() - started)
        yield pinned[: stop - start].numpy()


def write_vector(
    out_dir: Path, name: str, numel: int, windows: Iterable[np.ndarray], sidecar: Mapping[str, Any]
) -> tuple[Path, Path]:
    validate_sidecar(sidecar, label=name)
    if sidecar["name"] != name:
        raise ValueError(f"sidecar name {sidecar['name']!r} != {name!r}")
    f32 = write_f32(out_dir / f"{name}.f32", numel, windows)
    return f32, write_json(sidecar_path(f32), sidecar)


# -------------------------------------------------------------- gradients
def install_grad_hooks(
    indexed_parameters: Sequence[tuple[int, Any, Any]],
    callback: Callable[[int, Any, Any], None],
) -> list[Any]:
    """Post-accumulate-grad hooks on included parameters: the callback gets
    ``(entry_index, entry, grad)`` and the grad is released immediately (the
    optimizer-in-backward pattern), so per-row bf16 gradients are never all
    resident. Parameters without ``requires_grad`` get no hook (constant
    zero contribution, as ``SerialGradientBackend`` treats them)."""
    handles = []
    for entry_index, entry, parameter in indexed_parameters:
        if not parameter.requires_grad:
            continue

        def hook(param, *, entry_index=entry_index, entry=entry):
            grad = param.grad
            param.grad = None
            if grad is None:
                return
            callback(entry_index, entry, grad)

        handles.append(parameter.register_post_accumulate_grad_hook(hook))
    return handles


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


@dataclass
class RowState:
    """Mutable per-row switch read by the grad hooks."""

    mode: str = "accumulate"  # "norm" | "accumulate"
    track_norm: bool = True
    unit_alpha: float | None = None
    n_grads: int = 0


def make_accumulate_callback(
    state: RowState, raw: FlatAccumulator, unit: FlatAccumulator | None, norm: RowNorm
) -> Callable[[int, Any, Any], None]:
    def callback(entry_index: int, entry: Any, grad: Any) -> None:
        del entry_index
        state.n_grads += 1
        if state.mode == "norm":
            norm.add(grad)
            return
        if state.mode != "accumulate":
            raise RuntimeError(f"unknown row mode {state.mode!r}")
        flat = grad.reshape(-1)
        raw.add(entry, flat)
        if state.track_norm:
            norm.add(grad)
        if unit is not None and state.unit_alpha is not None:
            unit.add(entry, flat, alpha=state.unit_alpha)

    return callback


def accumulate_row(
    adapter: Any,
    batch: Any,
    *,
    state: RowState,
    raw: FlatAccumulator,
    unit: FlatAccumulator | None,
    norm: RowNorm,
    expected_grads: int,
) -> tuple[float, float, int]:
    """One packed row -> (loss, grad_norm, n_backward_passes). With the unit
    accumulator on: pass 1 measures the norm (grads freed as they arrive),
    pass 2 re-runs forward+backward and adds to both accumulators."""

    def backward() -> Any:
        state.n_grads = 0
        losses = adapter.per_datapoint_losses(batch).losses
        if losses.numel() != 1:
            raise RuntimeError(f"expected one per_sequence_sum loss, got {losses.numel()}")
        loss = losses[0]
        loss.backward()
        if state.n_grads != expected_grads:
            raise RuntimeError(
                f"row produced {state.n_grads} parameter gradients, expected {expected_grads}"
            )
        return loss

    norm.reset()
    passes = 0
    if unit is not None:
        state.mode, state.track_norm = "norm", False
        backward()
        passes += 1
        grad_norm = norm.value()
        state.unit_alpha = unit_alpha(grad_norm)
        state.mode = "accumulate"
        loss = backward()
        passes += 1
        if state.unit_alpha is None:
            unit.skip_row()
        else:
            unit.commit_row()
    else:
        state.mode, state.track_norm, state.unit_alpha = "accumulate", True, None
        loss = backward()
        passes += 1
        grad_norm = norm.value()
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite row gradient norm {grad_norm}")
    raw.commit_row()
    return float(loss.item()), grad_norm, passes


# ----------------------------------------------------------- memory budget
@dataclass(frozen=True)
class MemoryEstimate:
    weights_gb: float
    accumulators_gb: float
    grads_in_flight_gb: float
    logits_gb: float
    activations_gb: float
    overhead_gb: float
    safety_factor: float
    n_accumulators: int

    @property
    def per_accumulator_gb(self) -> float:
        return self.accumulators_gb / max(self.n_accumulators, 1)

    @property
    def total_gb(self) -> float:
        raw = (
            self.weights_gb
            + self.accumulators_gb
            + self.grads_in_flight_gb
            + self.logits_gb
            + self.activations_gb
        )
        return raw * self.safety_factor + self.overhead_gb

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["total_gb"] = self.total_gb
        return payload


def estimate_memory_gb(
    *,
    total_params: int,
    included_numel: int,
    n_accumulators: int,
    sequence_length: int,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    n_layers: int,
    largest_param_numel: int,
    weight_bytes: int = 2,
    overhead_gb: float = 3.0,
    safety_factor: float = 1.05,
) -> MemoryEstimate:
    """Coarse peak-memory model for the hook-streamed accumulation loop
    (weights + fp32 accumulators + a few in-flight bf16 grads + CE logits
    transients + checkpointed activations). Calibrated against gate2's
    measured ``estimate-adam`` peak (84.8 GB at seq 8192: model + one fp32
    ``[P]`` accumulator + transients); it exists to refuse a run that cannot
    fit BEFORE the accumulators allocate, not to be exact — the receipt
    records the measured peak for recalibration."""
    weights = total_params * weight_bytes
    accumulators = n_accumulators * included_numel * 4
    # Hooks free grads per parameter; allow a handful of the largest in flight.
    grads_in_flight = 4 * largest_param_numel * weight_bytes
    # bf16 logits + the fp32 CE copy; the backward grads reuse freed buffers.
    logits = sequence_length * vocab_size * (2 + 4 + 2)
    # Checkpointed layer inputs (x2 for the recompute segment) + one layer's
    # MLP intermediate in fp32-ish transients.
    activations = (
        n_layers * sequence_length * hidden_size * weight_bytes * 2
        + sequence_length * intermediate_size * 4 * 3
    )
    return MemoryEstimate(
        weights_gb=weights / GB,
        accumulators_gb=accumulators / GB,
        grads_in_flight_gb=grads_in_flight / GB,
        logits_gb=logits / GB,
        activations_gb=activations / GB,
        overhead_gb=overhead_gb,
        safety_factor=safety_factor,
        n_accumulators=n_accumulators,
    )


def check_budget(estimate: MemoryEstimate, available_gb: float, *, override: bool = False) -> str:
    verdict = (
        f"memory estimate {estimate.total_gb:.1f} GB vs available {available_gb:.1f} GB "
        f"({json.dumps({k: round(v, 2) for k, v in estimate.to_dict().items()})})"
    )
    if estimate.total_gb > available_gb and not override:
        raise RuntimeError(
            f"refusing: {verdict}; set unit_accumulator: false (drops one "
            f"{estimate.per_accumulator_gb:.1f} GB accumulator) or budget_override: true"
        )
    return verdict


def _model_geometry(model: Any) -> dict[str, int]:
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None) or config

    def pick(name: str, default: int) -> int:
        value = getattr(text, name, None)
        if value is None:
            value = getattr(config, name, None)
        return int(value) if isinstance(value, int) else default

    return {
        "vocab_size": pick("vocab_size", 262_144),
        "hidden_size": pick("hidden_size", 3_840),
        "intermediate_size": pick("intermediate_size", 15_360),
        "n_layers": pick("num_hidden_layers", 48),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


# ------------------------------------------------------------------ run
def _check_manifest(manifest: Any, config: MeanGradientsConfig) -> None:
    included = manifest.included_entries()
    if config.expected_included_params is not None and len(included) != config.expected_included_params:
        raise RuntimeError(
            f"manifest has {len(included)} included parameters, expected "
            f"{config.expected_included_params} (set expected_included_params: null for smoke models)"
        )
    if config.expected_included_numel is not None and manifest.included_numel != config.expected_included_numel:
        raise RuntimeError(
            f"manifest included_numel {manifest.included_numel} != expected "
            f"{config.expected_included_numel}"
        )


def _persist_manifest(manifest: Any, out_dir: Path) -> Path:
    """One shared ``parameter_manifest/`` per vector dir; a second dataset
    with a different digest is a loud refusal (the vectors would not share
    coordinates). Written to a private temp dir and renamed into place, so
    the per-dataset processes that start together never read a half-written
    file (``rename`` onto an existing populated dir fails -> compare)."""
    from scimt.data_attribution.manifest import ParameterManifest

    directory = out_dir / "parameter_manifest"

    def compare_existing() -> Path:
        existing = ParameterManifest.load(directory)
        if existing.digest() != manifest.digest():
            raise RuntimeError(
                f"{directory} holds manifest {existing.digest()[:12]}, this model gives "
                f"{manifest.digest()[:12]} — vectors would not share coordinates"
            )
        return directory

    if (directory / "parameter_manifest.json").is_file():
        return compare_existing()
    staging = out_dir / f"parameter_manifest.tmp-{os.getpid()}"
    manifest.save(staging)
    try:
        staging.rename(directory)
    except OSError:
        # Another process won the rename; theirs must describe the same model.
        for file in staging.iterdir():
            file.unlink()
        staging.rmdir()
        return compare_existing()
    return directory


def _fold_outputs_complete(out_dir: Path, config: MeanGradientsConfig, fold: str, n_rows: int, digest: str) -> bool:
    tags = [RAW_TAG] + ([UNIT_TAG] if config.unit_accumulator else [])
    for tag in tags:
        f32 = out_dir / f"{vector_name(config.dataset, tag, fold)}.f32"
        if not f32.is_file() or not sidecar_path(f32).is_file():
            return False
        sidecar = read_sidecar(f32)
        if sidecar["n_rows"] != n_rows or sidecar["manifest_digest"] != digest:
            return False
    return True


def run(config: MeanGradientsConfig) -> dict[str, Any]:
    import torch
    from scimt.data_attribution.datasets import PackedMidtrainingDataset
    from scimt.data_attribution.gradients import backward_memory_mode
    from scimt.data_attribution.losses import CausalLMLossAdapter
    from scimt.data_attribution.manifest import (
        ParameterManifest,
        freeze_excluded,
        included_named_parameters,
    )
    from scimt.data_attribution.runner import _load_model, _load_tokenizer, _model_identifier

    started = time.time()
    tag = timestamp_tag()
    out_dir = Path(config.out_dir)
    logs_dir = out_dir / "logs" / config.dataset
    evidence_dir = out_dir / "evidence"
    logs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    receipt_path = evidence_dir / f"mean_gradients__{config.dataset}__{tag}.json"

    def receipt(status: str, **extra: Any) -> None:
        write_json(
            receipt_path,
            {
                "status": status,
                "dataset": config.dataset,
                "config": config.to_dict(),
                "git_commit": git_commit(),
                "created_at": now_iso(),
                "timings_s": timings,
                "host_maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                **extra,
            },
        )

    log(f"mean_gradients dataset={config.dataset} sample={config.sample_path}")
    model_dir, model_sha = resolve_snapshot(config.model, label="model", allow_download=config.allow_download)
    tokenizer_dir, tokenizer_sha = resolve_snapshot(config.tokenizer, label="tokenizer", allow_download=config.allow_download)
    tokenizer = _load_tokenizer(tokenizer_dir)

    t0 = time.time()
    docs = read_sample(config.sample_path, text_column=config.text_column)
    lengths = [len(tokenizer(doc.text, add_special_tokens=False)["input_ids"]) for doc in docs]
    plan = plan_rows(lengths, [doc.group for doc in docs], config.sequence_length, config.n_folds, max_rows=config.max_rows)
    dataset = PackedMidtrainingDataset(
        config.sample_path,
        tokenizer,
        config.sequence_length,
        config.seed,
        reduction="per_sequence_sum",
        max_sequences=config.max_rows,
        text_column=config.text_column,
    )
    if len(dataset) != len(plan):
        raise RuntimeError(f"packed dataset has {len(dataset)} rows but the plan has {len(plan)}")
    if not plan:
        raise RuntimeError("sample packs into zero full rows — enlarge the sample or lower sequence_length")
    folds = fold_row_indices(len(plan), config.n_folds)
    if any(not rows for rows in folds.values()):
        raise RuntimeError(f"{len(plan)} rows cannot fill {config.n_folds} folds")
    timings["tokenize_and_pack_s"] = time.time() - t0
    sample_sha = sha256_file(Path(config.sample_path))
    log(
        f"{len(docs)} docs / {sum(lengths)} tokens -> {len(plan)} packed rows of "
        f"{config.sequence_length}; folds {[len(v) for v in folds.values()]}"
    )

    t0 = time.time()
    model = _load_model(model_dir, dtype=config.dtype, device=config.device, gradient_checkpointing=config.gradient_checkpointing)
    timings["load_model_s"] = time.time() - t0
    manifest = ParameterManifest.from_model(
        model, _model_identifier(model), include=list(config.parameters.include), exclude=list(config.parameters.exclude)
    )
    _check_manifest(manifest, config)
    freeze_excluded(model, manifest)
    manifest_dir = _persist_manifest(manifest, out_dir)
    digest = manifest.digest()
    geometry = _model_geometry(model)
    indexed = [
        (index, entry, parameter)
        for index, (entry, parameter) in enumerate(included_named_parameters(model, manifest))
    ]
    expected_grads = sum(1 for _, _, p in indexed if p.requires_grad)
    n_accumulators = 2 if config.unit_accumulator else 1
    estimate = estimate_memory_gb(
        total_params=geometry["total_params"],
        included_numel=manifest.included_numel,
        n_accumulators=n_accumulators,
        sequence_length=config.sequence_length,
        vocab_size=geometry["vocab_size"],
        hidden_size=geometry["hidden_size"],
        intermediate_size=geometry["intermediate_size"],
        n_layers=geometry["n_layers"],
        largest_param_numel=max(e.numel for e in manifest.included_entries()),
        weight_bytes=2 if config.dtype != "float32" else 4,
    )
    device = torch.device(config.device)
    if config.device_budget_gb is not None:
        available_gb = config.device_budget_gb
    else:
        free, total = torch.cuda.mem_get_info(device)
        # The weights are already resident: add them back to "free".
        available_gb = (free + geometry["total_params"] * (2 if config.dtype != "float32" else 4)) / GB
        log(f"device {device}: {free / GB:.1f} GB free of {total / GB:.1f} GB after model load")
    verdict = check_budget(estimate, available_gb, override=config.budget_override)
    log(verdict)
    log(f"manifest {digest[:12]} included={len(manifest.included_entries())} numel={manifest.included_numel}")

    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum", device=config.device)
    raw = FlatAccumulator(manifest.included_numel, config.device, tag=RAW_TAG)
    unit = FlatAccumulator(manifest.included_numel, config.device, tag=UNIT_TAG) if config.unit_accumulator else None
    norm = RowNorm(config.device)
    state = RowState()
    handles = install_grad_hooks(indexed, make_accumulate_callback(state, raw, unit, norm))
    torch.cuda.reset_peak_memory_stats(device)

    rows_log = logs_dir / "rows.jsonl"
    row_seconds: list[float] = []
    fold_summaries: dict[str, dict[str, Any]] = {}
    written: list[str] = []
    n_tokens_per_row = config.sequence_length - 1

    def flush(accumulator: FlatAccumulator, tag_name: str, fold_name: str, n_rows: int) -> None:
        name = vector_name(config.dataset, tag_name, fold_name)
        mean = accumulator.finalize_mean_()
        t_flush = time.time()
        window_log: list[dict[str, float]] = []

        def on_window(start: int, stop: int, seconds: float) -> None:
            window_log.append({"start": start, "stop": stop, "seconds": seconds, "gb_per_s": (stop - start) * 4 / GB / max(seconds, 1e-9)})

        sidecar = vector_sidecar(
            name=name, kind="gdp", dataset=config.dataset, fold=fold_name, n_rows=n_rows,
            n_tokens=n_rows * n_tokens_per_row, manifest_digest=digest,
            model_hf_id=config.model.hf_id, model_sha=model_sha, sequence_length=config.sequence_length,
        )
        write_vector(out_dir, name, manifest.included_numel, gpu_windows(mean, config.flush_window, on_window=on_window), sidecar)
        timings[f"flush_{name}_s"] = time.time() - t_flush
        slowest = max((w["seconds"] for w in window_log), default=0.0)
        log(f"wrote {name} ({n_rows} rows) in {timings[f'flush_{name}_s']:.1f}s; slowest window {slowest:.2f}s")
        written.append(name)

    try:
        memory_mode = backward_memory_mode(model, bool(getattr(model, "is_gradient_checkpointing", False)))
        with memory_mode, rows_log.open("a", encoding="utf-8") as rows_handle:
            for fold, indices in folds.items():
                fold_name = fold_label(fold)
                if config.resume and _fold_outputs_complete(out_dir, config, fold_name, len(indices), digest):
                    log(f"fold {fold_name}: outputs present with n_rows={len(indices)} — skipping (resume)")
                    fold_summaries[fold_name] = {"n_rows": len(indices), "resumed": True}
                    continue
                raw.reset()
                if unit is not None:
                    unit.reset()
                losses: list[float] = []
                norms: list[float] = []
                fold_started = time.time()
                for position, row_index in enumerate(indices):
                    row_started = time.time()
                    batch = dataset.batch_from_indices([row_index])
                    loss, grad_norm, passes = accumulate_row(
                        adapter, batch, state=state, raw=raw, unit=unit, norm=norm, expected_grads=expected_grads
                    )
                    seconds = time.time() - row_started
                    row_seconds.append(seconds)
                    losses.append(loss)
                    norms.append(grad_norm)
                    row_plan = plan[row_index]
                    rows_handle.write(
                        json.dumps(
                            {
                                "row_index": row_index,
                                "fold": fold_name,
                                "loss": loss,
                                "n_target_tokens": n_tokens_per_row,
                                "grad_norm": grad_norm,
                                "backward_passes": passes,
                                "seconds": seconds,
                                "docs": row_plan.doc_records(docs),
                                "group_tokens": row_plan.group_tokens(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rows_handle.flush()
                    if position % 8 == 0 or position == len(indices) - 1:
                        rate = (time.time() - fold_started) / (position + 1)
                        log(
                            f"fold {fold_name} row {position + 1}/{len(indices)} loss={loss:.3f} "
                            f"|g|={grad_norm:.3e} {seconds:.1f}s ({rate:.1f}s/row, peak "
                            f"{torch.cuda.max_memory_allocated(device) / GB:.1f} GB)"
                        )
                timings[f"fold_{fold_name}_rows_s"] = time.time() - fold_started
                flush(raw, RAW_TAG, fold_name, raw.n_rows)
                if unit is not None:
                    flush(unit, UNIT_TAG, fold_name, unit.n_rows)
                fold_summaries[fold_name] = {
                    "n_rows": len(indices),
                    "raw_rows": raw.n_rows,
                    "unit_rows": None if unit is None else unit.n_rows,
                    "unit_skipped_zero_norm": None if unit is None else unit.n_skipped,
                    "loss_mean": statistics.fmean(losses),
                    "grad_norm_median": statistics.median(norms),
                    "grad_norm_max": max(norms),
                    "resumed": False,
                }
                receipt("fold_done", folds=fold_summaries, written=written)
    finally:
        remove_hooks(handles)

    # Pooled means on the host from the fold means (row-count weighted).
    t0 = time.time()
    for tag_name in [RAW_TAG] + ([UNIT_TAG] if unit is not None else []):
        fold_names = [fold_label(f) for f in folds]
        f32_paths = [out_dir / f"{vector_name(config.dataset, tag_name, f)}.f32" for f in fold_names]
        sidecars = [read_sidecar(p) for p in f32_paths]
        rows_per_fold = [s["n_rows"] for s in sidecars]
        pooled_name = vector_name(config.dataset, tag_name, POOLED_FOLD)
        means = [open_f32(p, manifest.included_numel) for p in f32_paths]
        tmp = out_dir / f"{pooled_name}.f32.tmp"
        out = np.memmap(tmp, dtype="<f4", mode="w+", shape=(manifest.included_numel,))
        total_rows = combine_fold_means(means, rows_per_fold, out, window=config.combine_window)
        out.flush()
        del out, means
        tmp.replace(out_dir / f"{pooled_name}.f32")
        write_json(
            out_dir / f"{pooled_name}.json",
            vector_sidecar(
                name=pooled_name, kind="gdp", dataset=config.dataset, fold=POOLED_FOLD, n_rows=total_rows,
                n_tokens=sum(s["n_tokens"] for s in sidecars), manifest_digest=digest,
                model_hf_id=config.model.hf_id, model_sha=model_sha, sequence_length=config.sequence_length,
            ),
        )
        written.append(pooled_name)
        log(f"wrote {pooled_name} from folds {rows_per_fold}")
    timings["combine_pooled_s"] = time.time() - t0
    timings["total_s"] = time.time() - started

    summary = {
        "model": {"hf_id": config.model.hf_id, "sha": model_sha, "path": str(model_dir)},
        "tokenizer": {"hf_id": config.tokenizer.hf_id, "sha": tokenizer_sha, "path": str(tokenizer_dir)},
        "manifest_digest": digest,
        "manifest_dir": str(manifest_dir),
        "included_params": len(manifest.included_entries()),
        "included_numel": manifest.included_numel,
        "sample_sha256": sample_sha,
        "n_docs": len(docs),
        "n_doc_tokens": int(sum(lengths)),
        "n_rows": len(plan),
        "rows_per_fold": {fold_label(f): len(v) for f, v in folds.items()},
        "folds": fold_summaries,
        "written": written,
        "memory_estimate": estimate.to_dict(),
        "budget_verdict": verdict,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated(device) / GB,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / GB,
        "row_seconds_median": statistics.median(row_seconds) if row_seconds else None,
        "row_seconds_p90": (sorted(row_seconds)[int(0.9 * (len(row_seconds) - 1))] if row_seconds else None),
        "rows_per_hour": (3600.0 / statistics.fmean(row_seconds)) if row_seconds else None,
        "rows_log": str(rows_log),
    }
    receipt("done", **summary)
    write_json(logs_dir / "run_manifest.json", {"config": config.to_dict(), "timings_s": timings, **summary})
    log(f"done in {timings['total_s'] / 60:.1f} min; receipt {receipt_path}")
    return summary


if __name__ == "__main__":
    run(config_from_env(CONFIG_ENV, DEFAULTS, MeanGradientsConfig.from_mapping))
