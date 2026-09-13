"""EFT-row scores at ``google/gemma-3-12b-it`` against resident dataset vectors.

Per EFT row (``/workspace/attribution/eft_rows/eft_rows.jsonl``; chat rows
with ``messages`` / ``group`` / ``episode_id`` / ``subtype`` as
``eft_rows/build_eft_rows.py`` emits them — the gate2 ``build_queries_dataset``
schema plus metadata) the score against vector ``v`` is

    s_v(row) = v . grad_theta L_row(theta_it),

``L_row`` the ``per_sequence_sum`` cross-entropy over the assistant tokens
only (``objective: sft`` masking through the -it chat template, via the
library's ``ChatSFTDataset``). Positive means the dataset direction lowers
the row's loss. Vectors are ``gdp`` (dataset-mean gradients) or ``inv``
(EK-FAC-preconditioned means) from the shared ``.f32`` + sidecar contract.

Geometry (gate2 ``pod/score_perdoc2.py`` lesson, PREMORTEM A.3): the model
and the backward pass live on ``device`` (cuda:0); the vectors stay RESIDENT
in bf16, sharded by contiguous parameter-index blocks (aligned to parameter
boundaries) across ``shard_devices`` (cuda:1..N-1), and nothing gradient
sized ever touches host memory: post-accumulate-grad hooks copy each
parameter's bf16 grad straight into the owning shard's block buffer (freeing
it on cuda:0), and every shard reduces its block against all its resident
vectors with windowed fp32 up-casts (bf16 inputs, fp32 accumulate); the
per-shard partials are summed into one score per vector. Startup refuses a
vector set that does not fit, printing the per-device budget table, and
self-checks the shard bookkeeping by dotting vector 0 against itself.

Rows for the -it forward are trimmed to their last target position instead
of ``ChatSFTDataset``'s full-``sequence_length`` padding — exact under causal
attention (no target reads a later position) and ~4x cheaper at 4096.

Modes: ``main`` (it model), ``pt_mismatch`` (same rows, the -pt model, still
rendered with the -it tokenizer's template — pt ships no ``chat_template``;
PREMORTEM A.5) for the checkpoint-mismatch diagnostic, and ``oracle`` (the
first ``oracle_rows`` rows scored ``oracle_repeats`` times for the bf16
run-to-run noise floor; gate2 measured ~0.5–2 %).

Passes (``passes: [{name, rows_filter: all|<n episodes per pair-type>,
vectors}]``) share one gradient per row: the union of pass rows is scored
once, each pass gets its own ``scores/<name>.jsonl`` restricted to its
vectors, plus ``scores/<name>_manifest.json``. Subsample passes are drawn
BY EPISODE (seeded by ``episode_seed``; both rows of a coin/charter or
ambiguous/ambiguous_wrong pair always travel together) so the analysis's
paired per-episode contrast never loses a side. The diagnostic modes write
``scores/pt_mismatch.jsonl`` / ``scores/oracle.jsonl`` — the analysis finds
them by name — so those modes take exactly one pass named after the mode.
At vector-load time the resident set's Gram matrix (fp32 accumulate over
shards) is written as ``scores/vector_norms.json`` + ``vector_cosines.json``
for the analysis's cosine normalisation and fold / cross-dataset gates.
Config-first, no argparse (``$SCIMT_EKFAC_SCORE_EFT_ROWS_CONFIG`` over
:data:`DEFAULTS`).
"""

from __future__ import annotations

import json
import math
import random
import re
import resource
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod.mean_gradients import (  # noqa: E402
    EXPECTED_INCLUDED_NUMEL,
    EXPECTED_INCLUDED_PARAMS,
    GB,
    ModelRef,
    ParameterSelection,
    RowNorm,
    _int_or_none,
    _reject_unknown,
    _require_bool,
    _require_int,
    _require_str,
    config_from_env,
    git_commit,
    install_grad_hooks,
    log,
    now_iso,
    open_f32,
    read_sidecar,
    remove_hooks,
    resolve_snapshot,
    sha256_file,
    timestamp_tag,
    vector_numel,
    write_json,
)

CONFIG_ENV = "SCIMT_EKFAC_SCORE_EFT_ROWS_CONFIG"
MODES = ("main", "pt_mismatch", "oracle")
# analysis/analyze.py discovers these two diagnostics by file name.
DIAGNOSTIC_PASS_NAMES = ("pt_mismatch", "oracle")
VECTOR_NORMS_FILE = "vector_norms.json"
VECTOR_COSINES_FILE = "vector_cosines.json"
DEFAULT_EPISODE_SEED = 20260913
ROW_ORDERS = ("interleave_groups", "file")
RESIDENT_DTYPES = ("bfloat16", "float32")
DEFAULT_GROUPS = ("charter", "coin", "ambiguous", "ambiguous_wrong")
REQUIRED_SCORE_KEYS = (
    "row_id",
    "group",
    "episode_id",
    "subtype",
    "n_target_tokens",
    "loss",
    "grad_norm",
    "scores",
)
DEFAULT_IT_HF_ID = "google/gemma-3-12b-it"
DEFAULT_PT_HF_ID = "google/gemma-3-12b-pt"
DEFAULT_ROWS_PATH = "/workspace/attribution/eft_rows/eft_rows.jsonl"
DEFAULT_OUT_DIR = "/workspace/attribution/eft_scores"
ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4}
SELF_CHECK_REL_TOL = 0.02  # bf16-stored vector dotted with itself vs fp32 host norm^2


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class PassConfig:
    name: str
    rows_filter: str | int  # "all" or n episodes per pair-type (seeded)
    vectors: tuple[str, ...]  # .f32 paths

    _KEYS = ("name", "rows_filter", "vectors")

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> PassConfig:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a mapping {{name, rows_filter, vectors}}")
        _reject_unknown(raw, cls._KEYS, label)
        name = _require_str(raw.get("name"), f"{label}.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"{label}.name {name!r} must be a bare file-name token")
        rows_filter = raw.get("rows_filter", "all")
        if rows_filter != "all":
            rows_filter = _require_int(rows_filter, f"{label}.rows_filter")
        vectors = raw.get("vectors")
        if not isinstance(vectors, (list, tuple)) or not vectors:
            raise ValueError(f"{label}.vectors must be a non-empty list of .f32 paths")
        for path in vectors:
            if not isinstance(path, str) or not path.endswith(".f32"):
                raise ValueError(f"{label}.vectors entries must be .f32 paths, got {path!r}")
        if len(set(vectors)) != len(vectors):
            raise ValueError(f"{label}.vectors has duplicates")
        return cls(name=name, rows_filter=rows_filter, vectors=tuple(vectors))

    @property
    def episodes_per_pair_type(self) -> int | None:
        return None if self.rows_filter == "all" else int(self.rows_filter)


@dataclass(frozen=True)
class ScoreConfig:
    mode: str
    rows_path: str
    out_dir: str
    model: ModelRef
    pt_model: ModelRef
    tokenizer: ModelRef
    passes: tuple[PassConfig, ...]
    sequence_length: int = 4096
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    shard_devices: tuple[str, ...] | None = None
    shard_budget_gb: float = 130.0
    resident_dtype: str = "bfloat16"
    dot_window: int = 1 << 26
    stage_window: int = 1 << 27
    parameters: ParameterSelection = field(default_factory=ParameterSelection)
    gradient_checkpointing: bool = True
    row_order: str = "interleave_groups"
    groups: tuple[str, ...] = DEFAULT_GROUPS
    episode_seed: int = DEFAULT_EPISODE_SEED
    resume: bool = True
    oracle_rows: int = 8
    oracle_repeats: int = 2
    allow_download: bool = False
    expected_included_params: int | None = EXPECTED_INCLUDED_PARAMS
    expected_included_numel: int | None = EXPECTED_INCLUDED_NUMEL
    throughput_target_rows_per_s: float = 1.0
    self_check: bool = True

    _KEYS = (
        "mode", "rows_path", "out_dir", "model", "pt_model", "tokenizer", "passes",
        "sequence_length", "dtype", "device", "shard_devices", "shard_budget_gb",
        "resident_dtype", "dot_window", "stage_window", "parameters",
        "gradient_checkpointing", "row_order", "groups", "episode_seed", "resume",
        "oracle_rows", "oracle_repeats", "allow_download", "expected_included_params",
        "expected_included_numel", "throughput_target_rows_per_s", "self_check",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ScoreConfig:
        _reject_unknown(raw, cls._KEYS, "score_eft_rows")
        mode = _require_str(raw.get("mode", "main"), "mode")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        passes_raw = raw.get("passes")
        if not isinstance(passes_raw, (list, tuple)) or not passes_raw:
            raise ValueError("passes must be a non-empty list")
        passes = tuple(
            PassConfig.from_mapping(entry, label=f"passes[{index}]")
            for index, entry in enumerate(passes_raw)
        )
        names = [p.name for p in passes]
        if len(set(names)) != len(names):
            raise ValueError(f"pass names must be unique, got {names}")
        if mode in DIAGNOSTIC_PASS_NAMES:
            if names != [mode]:
                raise ValueError(
                    f"mode {mode!r} takes exactly one pass named {mode!r} (the analysis "
                    f"discovers scores/{mode}.jsonl by name), got {names}"
                )
        else:
            reserved = sorted(set(names) & set(DIAGNOSTIC_PASS_NAMES))
            if reserved:
                raise ValueError(f"pass names {reserved} are reserved for their modes")
        row_order = _require_str(raw.get("row_order", "interleave_groups"), "row_order")
        if row_order not in ROW_ORDERS:
            raise ValueError(f"row_order must be one of {ROW_ORDERS}")
        resident_dtype = _require_str(raw.get("resident_dtype", "bfloat16"), "resident_dtype")
        if resident_dtype not in RESIDENT_DTYPES:
            raise ValueError(f"resident_dtype must be one of {RESIDENT_DTYPES}")
        dtype = _require_str(raw.get("dtype", "bfloat16"), "dtype")
        if dtype not in ("bfloat16", "float32"):
            raise ValueError(f"dtype must be bfloat16|float32, got {dtype!r}")
        groups = raw.get("groups", list(DEFAULT_GROUPS))
        if (
            not isinstance(groups, (list, tuple))
            or not groups
            or not all(isinstance(g, str) and g for g in groups)
            or len(set(groups)) != len(groups)
        ):
            raise ValueError("groups must be a non-empty list of unique names")
        shard_devices = raw.get("shard_devices")
        if shard_devices is not None:
            if (
                not isinstance(shard_devices, (list, tuple))
                or not shard_devices
                or not all(isinstance(d, str) and d for d in shard_devices)
                or len(set(shard_devices)) != len(shard_devices)
            ):
                raise ValueError("shard_devices must be null or a non-empty list of unique device strings")
            shard_devices = tuple(shard_devices)
        budget = raw.get("shard_budget_gb", 130.0)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
            raise ValueError("shard_budget_gb must be a positive number")
        target = raw.get("throughput_target_rows_per_s", 1.0)
        if isinstance(target, bool) or not isinstance(target, (int, float)) or target <= 0:
            raise ValueError("throughput_target_rows_per_s must be a positive number")
        model = ModelRef.from_mapping(raw.get("model", DEFAULT_IT_HF_ID), label="model")
        pt_model = ModelRef.from_mapping(raw.get("pt_model", DEFAULT_PT_HF_ID), label="pt_model")
        tokenizer = ModelRef.from_mapping(raw.get("tokenizer", model.to_dict()), label="tokenizer")
        return cls(
            mode=mode,
            rows_path=_require_str(raw.get("rows_path", DEFAULT_ROWS_PATH), "rows_path"),
            out_dir=_require_str(raw.get("out_dir", DEFAULT_OUT_DIR), "out_dir"),
            model=model,
            pt_model=pt_model,
            tokenizer=tokenizer,
            passes=passes,
            sequence_length=_require_int(raw.get("sequence_length", 4096), "sequence_length", minimum=2),
            dtype=dtype,
            device=_require_str(raw.get("device", "cuda:0"), "device"),
            shard_devices=shard_devices,
            shard_budget_gb=float(budget),
            resident_dtype=resident_dtype,
            dot_window=_require_int(raw.get("dot_window", 1 << 26), "dot_window"),
            stage_window=_require_int(raw.get("stage_window", 1 << 27), "stage_window"),
            parameters=ParameterSelection.from_mapping(raw.get("parameters"), label="parameters"),
            gradient_checkpointing=_require_bool(raw.get("gradient_checkpointing", True), "gradient_checkpointing"),
            row_order=row_order,
            groups=tuple(groups),
            episode_seed=_require_int(raw.get("episode_seed", DEFAULT_EPISODE_SEED), "episode_seed", minimum=0),
            resume=_require_bool(raw.get("resume", True), "resume"),
            oracle_rows=_require_int(raw.get("oracle_rows", 8), "oracle_rows"),
            oracle_repeats=_require_int(raw.get("oracle_repeats", 2), "oracle_repeats", minimum=2),
            allow_download=_require_bool(raw.get("allow_download", False), "allow_download"),
            expected_included_params=_int_or_none(raw.get("expected_included_params", EXPECTED_INCLUDED_PARAMS), "expected_included_params"),
            expected_included_numel=_int_or_none(raw.get("expected_included_numel", EXPECTED_INCLUDED_NUMEL), "expected_included_numel"),
            throughput_target_rows_per_s=float(target),
            self_check=_require_bool(raw.get("self_check", True), "self_check"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("model", "pt_model", "tokenizer"):
            payload[key] = getattr(self, key).to_dict()
        payload["parameters"] = {"include": list(self.parameters.include), "exclude": list(self.parameters.exclude)}
        payload["passes"] = [
            {"name": p.name, "rows_filter": p.rows_filter, "vectors": list(p.vectors)} for p in self.passes
        ]
        return payload


DEFAULTS: dict[str, Any] = {
    "mode": "main",
    "rows_path": DEFAULT_ROWS_PATH,
    "out_dir": DEFAULT_OUT_DIR,
    "model": {"hf_id": DEFAULT_IT_HF_ID, "revision": None, "expected_sha_prefix": "96b6f1e", "local_path": None},
    "pt_model": {"hf_id": DEFAULT_PT_HF_ID, "revision": None, "expected_sha_prefix": "295efb6", "local_path": None},
    # ALWAYS the -it tokenizer: google/gemma-3-12b-pt ships no chat_template.
    "tokenizer": {"hf_id": DEFAULT_IT_HF_ID, "revision": None, "expected_sha_prefix": "96b6f1e", "local_path": None},
    "sequence_length": 4096,
    "resident_dtype": "bfloat16",
    "shard_budget_gb": 130.0,
    "row_order": "interleave_groups",
    "groups": list(DEFAULT_GROUPS),
    "episode_seed": DEFAULT_EPISODE_SEED,
    "passes": [
        {
            "name": "main_gdp",
            "rows_filter": "all",
            "vectors": [
                f"/workspace/attribution/vectors/{dataset}__gdp__all.f32"
                for dataset in ("dolmino", "charter_worked", "charter_noex", "coin")
            ],
        }
    ],
}


def model_for_mode(config: ScoreConfig) -> ModelRef:
    """``pt_mismatch`` swaps the model, nothing else."""
    return config.pt_model if config.mode == "pt_mismatch" else config.model


def tokenizer_for_mode(config: ScoreConfig) -> ModelRef:
    """Every mode — ``pt_mismatch`` included — renders rows with the
    configured (-it) tokenizer: the pt tokenizer has no chat template, and
    the it/pt vocabularies coincide."""
    return config.tokenizer


# --------------------------------------------------------------------- rows
@dataclass(frozen=True)
class RowMeta:
    row_index: int
    row_id: str
    group: str
    episode_id: str
    subtype: str
    messages: tuple[Mapping[str, str], ...]


def row_id_of(group: str, episode_id: str) -> str:
    """Stable join key for the analysis (pairs are matched on episode_id)."""
    return f"{group}:{episode_id}"


def load_eft_rows(path: str | Path, groups: Sequence[str]) -> list[RowMeta]:
    rows: list[RowMeta] = []
    seen: set[str] = set()
    unknown_groups: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(text for text in handle if text.strip()):
            raw = json.loads(line)
            messages = raw.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or not all(isinstance(m, Mapping) and {"role", "content"} <= set(m) for m in messages)
            ):
                raise ValueError(f"{path}: row {index} lacks a messages list of role/content dicts")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}: row {index} does not end with an assistant turn")
            group = raw.get("group")
            episode_id = raw.get("episode_id")
            if not isinstance(group, str) or not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"{path}: row {index} lacks string group/episode_id")
            if group not in groups:
                unknown_groups.add(group)
            subtype = raw.get("subtype")
            if not isinstance(subtype, str) or not subtype:
                subtype = raw.get("conflict_subtype") or "agreement"
            row_id = row_id_of(group, episode_id)
            if row_id in seen:
                raise ValueError(f"{path}: duplicate (group, episode_id) {row_id}")
            seen.add(row_id)
            rows.append(
                RowMeta(
                    row_index=index,
                    row_id=row_id,
                    group=group,
                    episode_id=episode_id,
                    subtype=str(subtype),
                    messages=tuple(dict(m) for m in messages),
                )
            )
    if unknown_groups:
        raise ValueError(
            f"{path}: groups {sorted(unknown_groups)} not in configured groups {list(groups)}"
        )
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def episode_pair_types(rows: Sequence[RowMeta]) -> dict[str, tuple[str, ...]]:
    """episode_id -> sorted tuple of the groups present for it (the episode's
    pair type: ``('charter', 'coin')`` for a conflict pair,
    ``('ambiguous', 'ambiguous_wrong')`` for an agreement pair, ...)."""
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row.episode_id, set()).add(row.group)
    return {episode: tuple(sorted(present)) for episode, present in groups.items()}


def select_episodes(rows: Sequence[RowMeta], n: int, seed: int) -> dict[tuple[str, ...], list[str]]:
    """Seeded draw of ``n`` episodes per pair type (all of them when fewer
    exist), in first-appearance order within each type; deterministic in
    ``(rows, n, seed)``."""
    if n < 1:
        raise ValueError("rows_filter must be 'all' or a positive int")
    pair_types = episode_pair_types(rows)
    by_type: dict[tuple[str, ...], list[str]] = {}
    seen: set[str] = set()
    for row in rows:
        if row.episode_id not in seen:
            seen.add(row.episode_id)
            by_type.setdefault(pair_types[row.episode_id], []).append(row.episode_id)
    rng = random.Random(seed)
    chosen: dict[tuple[str, ...], list[str]] = {}
    for pair_type in sorted(by_type):
        episodes = by_type[pair_type]
        picked = set(rng.sample(episodes, min(n, len(episodes))))
        chosen[pair_type] = [episode for episode in episodes if episode in picked]
    return chosen


def select_rows(rows: Sequence[RowMeta], rows_filter: str | int, *, episode_seed: int) -> list[RowMeta]:
    """``all``, or every row of the ``n`` seeded episodes per pair type — a
    subsample never separates a coin row from its charter twin (or an
    ambiguous row from its ambiguous_wrong twin), which is what the paired
    per-episode contrast needs (analysis README: scorer requirement)."""
    if rows_filter == "all":
        return list(rows)
    chosen = select_episodes(rows, int(rows_filter), episode_seed)
    keep = {episode for episodes in chosen.values() for episode in episodes}
    return [row for row in rows if row.episode_id in keep]


def selection_summary(rows: Sequence[RowMeta], selected: Sequence[RowMeta], rows_filter: str | int, episode_seed: int) -> dict[str, Any]:
    """Manifest record of a pass's episode draw (available vs selected per
    pair type, the episode ids, and any shortfall)."""
    pair_types = episode_pair_types(rows)
    available: dict[tuple[str, ...], set[str]] = {}
    for episode, pair_type in pair_types.items():
        available.setdefault(pair_type, set()).add(episode)
    chosen: dict[tuple[str, ...], list[str]] = {}
    for row in selected:
        bucket = chosen.setdefault(pair_types[row.episode_id], [])
        if row.episode_id not in bucket:
            bucket.append(row.episode_id)
    per_type = {}
    for pair_type in sorted(available):
        picked = chosen.get(pair_type, [])
        requested = len(available[pair_type]) if rows_filter == "all" else min(int(rows_filter), len(available[pair_type]))
        per_type["+".join(pair_type)] = {
            "groups": list(pair_type),
            "available_episodes": len(available[pair_type]),
            "selected_episodes": len(picked),
            "shortfall": (0 if rows_filter == "all" else int(rows_filter) - requested),
            "episodes": picked,
        }
    return {
        "rows_filter": rows_filter,
        "episode_seed": episode_seed,
        "drawn_by": "episode (all rows of each chosen episode)",
        "n_rows": len(selected),
        "pair_types": per_type,
    }


def order_rows(rows: Sequence[RowMeta], row_order: str, groups: Sequence[str]) -> list[RowMeta]:
    """``file`` keeps file order; ``interleave_groups`` round-robins the
    groups (in configured order) so a partial run covers every class."""
    if row_order == "file":
        return list(rows)
    if row_order != "interleave_groups":
        raise ValueError(f"unknown row_order {row_order!r}")
    queues = {group: [r for r in rows if r.group == group] for group in groups}
    extra = [r for r in rows if r.group not in groups]
    if extra:
        raise ValueError(f"rows carry groups outside {list(groups)}")
    ordered: list[RowMeta] = []
    cursors = dict.fromkeys(groups, 0)
    while len(ordered) < len(rows):
        for group in groups:
            cursor = cursors[group]
            if cursor < len(queues[group]):
                ordered.append(queues[group][cursor])
                cursors[group] = cursor + 1
    return ordered


def trimmed_length(mask: Sequence[bool]) -> int:
    """Length to keep for a causal forward: last target position + 1."""
    last = -1
    for position, is_target in enumerate(mask):
        if is_target:
            last = position
    if last < 0:
        raise ValueError("row has no target tokens")
    return last + 1


class ChatRows:
    """``ChatSFTDataset`` rows (assistant span = targets, -it template) with
    exact trailing-pad trimming and loud completeness checks: no row may be
    dropped (target-less) or truncated, and each row's trimmed length must
    equal its full rendered length (the assistant turn ends the row)."""

    def __init__(self, rows_path: str | Path, tokenizer: Any, sequence_length: int, rows: Sequence[RowMeta]):
        from scimt.data_attribution.datasets import ChatSFTDataset

        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(
                "tokenizer has no chat_template — render EFT rows with the -it tokenizer "
                "(google/gemma-3-12b-pt ships none; PREMORTEM A.5)"
            )
        self.sequence_length = sequence_length
        self.dataset = ChatSFTDataset(
            str(rows_path), tokenizer, sequence_length, 0, reduction="per_sequence_sum"
        )
        if len(self.dataset) != len(rows) or list(self.dataset.source_rows) != list(range(len(rows))):
            missing = sorted(set(range(len(rows))) - set(self.dataset.source_rows))
            raise RuntimeError(
                f"ChatSFTDataset kept {len(self.dataset)} of {len(rows)} rows (dropped source "
                f"rows {missing[:10]}{'...' if len(missing) > 10 else ''}): target-less or "
                "over-length rows would silently shift the row alignment"
            )
        self.lengths: list[int] = []
        self.n_target_tokens: list[int] = []
        truncated: list[int] = []
        mismatched: list[int] = []
        for row in rows:
            batch = self.dataset.batch_from_indices([row.row_index])
            mask = [bool(x) for x in batch.target_mask[0].tolist()]
            length = trimmed_length(mask)
            rendered = len(ChatSFTDataset._render(tokenizer, list(row.messages), False))
            if rendered > sequence_length:
                truncated.append(row.row_index)
            elif rendered != length:
                mismatched.append(row.row_index)
            self.lengths.append(length)
            self.n_target_tokens.append(sum(mask))
        if truncated:
            raise RuntimeError(
                f"{len(truncated)} rows render longer than sequence_length={sequence_length} "
                f"(e.g. rows {truncated[:5]}); raise sequence_length"
            )
        if mismatched:
            raise RuntimeError(
                f"{len(mismatched)} rows do not end with their assistant span (e.g. rows "
                f"{mismatched[:5]}); the template must place the assistant turn last"
            )

    def batch(self, row_index: int) -> Any:
        from scimt.data_attribution.losses import TokenizedBatch

        batch = self.dataset.batch_from_indices([row_index])
        length = self.lengths[row_index]
        return TokenizedBatch(
            batch.input_ids[:, :length], batch.sequence_ids, batch.target_mask[:, :length]
        )

    def summary(self, rows: Sequence[RowMeta]) -> dict[str, Any]:
        by_group: dict[str, list[int]] = {}
        for row, length in zip(rows, self.lengths, strict=True):
            by_group.setdefault(row.group, []).append(length)
        return {
            "n_rows": len(rows),
            "n_tokens_max": max(self.lengths),
            "n_tokens_mean": statistics.fmean(self.lengths),
            "n_target_tokens_mean": statistics.fmean(self.n_target_tokens),
            "per_group": {
                group: {"rows": len(v), "n_tokens_mean": statistics.fmean(v)}
                for group, v in sorted(by_group.items())
            },
        }


# ------------------------------------------------------------------- shards
@dataclass(frozen=True)
class Shard:
    device: str
    start: int  # flat offset (inclusive)
    stop: int  # flat offset (exclusive)
    first_entry: int  # included-entry index (inclusive)
    stop_entry: int  # included-entry index (exclusive)

    @property
    def numel(self) -> int:
        return self.stop - self.start


def plan_shards(entry_numels: Sequence[int], devices: Sequence[str]) -> list[Shard]:
    """Contiguous parameter-aligned blocks, one per device, each boundary
    placed at the parameter edge closest to an equal split; the last device
    takes the remainder. Devices left without parameters are dropped."""
    if not devices:
        raise ValueError("at least one shard device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("shard devices must be unique")
    if any((not isinstance(n, int)) or n < 0 for n in entry_numels):
        raise ValueError("entry numels must be non-negative ints")
    total = sum(entry_numels)
    if total == 0:
        raise ValueError("nothing to shard")
    shards: list[Shard] = []
    entry, offset = 0, 0
    for index, device in enumerate(devices):
        start, first_entry = offset, entry
        if index == len(devices) - 1:
            offset, entry = total, len(entry_numels)
        else:
            target = round(total * (index + 1) / len(devices))
            while entry < len(entry_numels):
                candidate = offset + entry_numels[entry]
                if abs(candidate - target) < abs(offset - target):
                    offset, entry = candidate, entry + 1
                else:
                    break
        if offset > start:
            shards.append(Shard(device, start, offset, first_entry, entry))
    if shards[-1].stop != total or shards[-1].stop_entry != len(entry_numels):
        raise AssertionError("shard plan does not cover all parameters")
    return shards


def shard_owner_map(plan: Sequence[Shard], n_entries: int) -> list[int]:
    owners = [-1] * n_entries
    for shard_index, shard in enumerate(plan):
        for entry in range(shard.first_entry, shard.stop_entry):
            owners[entry] = shard_index
    if any(owner < 0 for owner in owners):
        raise ValueError("shard plan leaves parameters unowned")
    return owners


def shard_bytes(numel: int, n_vectors: int, *, resident_itemsize: int, grad_itemsize: int = 2, dot_window: int) -> dict[str, int]:
    window = min(dot_window, numel)
    resident = n_vectors * numel * resident_itemsize
    grad_block = numel * grad_itemsize
    scratch = n_vectors * window * 4 + window * 4  # fp32 up-casts for one window
    partials = n_vectors * 4
    return {
        "resident": resident,
        "grad_block": grad_block,
        "scratch": scratch,
        "partials": partials,
        "total": resident + grad_block + scratch + partials,
    }


def budget_table(
    plan: Sequence[Shard],
    n_vectors: int,
    *,
    resident_dtype: str,
    dot_window: int,
    budget_bytes: Mapping[str, float],
) -> list[dict[str, Any]]:
    table = []
    for shard in plan:
        need = shard_bytes(shard.numel, n_vectors, resident_itemsize=ITEMSIZE[resident_dtype], dot_window=dot_window)
        budget = float(budget_bytes[shard.device])
        table.append(
            {
                "device": shard.device,
                "numel": shard.numel,
                "entries": shard.stop_entry - shard.first_entry,
                "n_vectors": n_vectors,
                "resident_gb": need["resident"] / GB,
                "grad_block_gb": need["grad_block"] / GB,
                "scratch_gb": need["scratch"] / GB,
                "total_gb": need["total"] / GB,
                "budget_gb": budget / GB,
                "fits": need["total"] <= budget,
            }
        )
    return table


def format_budget_table(table: Sequence[Mapping[str, Any]]) -> str:
    header = f"{'device':>8} {'params':>7} {'numel':>15} {'K':>3} {'resident':>9} {'grad':>7} {'scratch':>8} {'total':>8} {'budget':>8} fits"
    lines = [header]
    for row in table:
        lines.append(
            f"{row['device']:>8} {row['entries']:>7} {row['numel']:>15,} {row['n_vectors']:>3} "
            f"{row['resident_gb']:>8.1f}G {row['grad_block_gb']:>6.1f}G {row['scratch_gb']:>7.1f}G "
            f"{row['total_gb']:>7.1f}G {row['budget_gb']:>7.1f}G {'yes' if row['fits'] else 'NO'}"
        )
    return "\n".join(lines)


def check_fits(table: Sequence[Mapping[str, Any]]) -> None:
    failing = [row for row in table if not row["fits"]]
    if failing:
        raise RuntimeError(
            "refusing: resident vector set does not fit the shard devices\n"
            + format_budget_table(table)
            + "\n-> drop vectors from the passes, add shard devices, or raise shard_budget_gb"
        )


# ------------------------------------------------------------------ vectors
@dataclass(frozen=True)
class VectorRef:
    name: str
    path: str
    sidecar: Mapping[str, Any]


def load_vector_refs(paths: Sequence[str], *, expected_numel: int, manifest_digest: str) -> list[VectorRef]:
    refs: list[VectorRef] = []
    names: set[str] = set()
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"vector {path} is missing")
        numel = vector_numel(path)
        if numel != expected_numel:
            raise ValueError(f"{path}: {numel} elements, manifest has {expected_numel}")
        sidecar = read_sidecar(path)
        if sidecar["manifest_digest"] != manifest_digest:
            raise ValueError(
                f"{path}: sidecar manifest_digest {sidecar['manifest_digest'][:12]} != "
                f"this model's {manifest_digest[:12]} — different parameter coordinates"
            )
        if sidecar["name"] in names:
            raise ValueError(f"duplicate vector name {sidecar['name']!r}")
        names.add(sidecar["name"])
        refs.append(VectorRef(name=sidecar["name"], path=str(path), sidecar=sidecar))
    return refs


def union_vector_paths(passes: Sequence[PassConfig]) -> list[str]:
    ordered: list[str] = []
    for pass_config in passes:
        for path in pass_config.vectors:
            if path not in ordered:
                ordered.append(path)
    return ordered


class ResidentShards:
    """The resident ``[K, block]`` matrices + bf16 grad block per shard."""

    def __init__(
        self,
        plan: Sequence[Shard],
        entries: Sequence[Any],
        vectors: Sequence[VectorRef],
        *,
        resident_dtype: str,
        dot_window: int,
        stage_window: int,
        score_device: str,
    ):
        import torch

        self._torch = torch
        self.plan = list(plan)
        self.entries = list(entries)
        self.vectors = list(vectors)
        self.owners = shard_owner_map(self.plan, len(self.entries))
        self.dot_window = dot_window
        self.stage_window = stage_window
        self.score_device = score_device
        self.resident_dtype = getattr(torch, resident_dtype)
        self.matrices = [
            torch.empty((len(vectors), shard.numel), dtype=self.resident_dtype, device=shard.device)
            for shard in self.plan
        ]
        self.grad_blocks = [
            torch.zeros(shard.numel, dtype=torch.bfloat16, device=shard.device) for shard in self.plan
        ]

    def _host_windows(self, memmap: np.memmap, start: int, stop: int) -> Iterable[tuple[int, int, np.ndarray]]:
        for w0 in range(start, stop, self.stage_window):
            w1 = min(w0 + self.stage_window, stop)
            yield w0, w1, np.ascontiguousarray(memmap[w0:w1])

    def stage(self) -> dict[str, float]:
        """Windowed host memmap -> fp32 device staging -> resident dtype."""
        torch = self._torch
        timings: dict[str, float] = {}
        for k, vector in enumerate(self.vectors):
            started = time.time()
            memmap = open_f32(vector.path)
            for shard_index, shard in enumerate(self.plan):
                for w0, w1, window in self._host_windows(memmap, shard.start, shard.stop):
                    staging = torch.from_numpy(window).to(shard.device)
                    self.matrices[shard_index][k, w0 - shard.start : w1 - shard.start].copy_(staging)
                    del staging
            del memmap
            timings[vector.name] = time.time() - started
            log(f"staged {vector.name} in {timings[vector.name]:.1f}s")
        return timings

    def write_grad(self, entry_index: int, entry: Any, grad: Any) -> None:
        shard_index = self.owners[entry_index]
        shard = self.plan[shard_index]
        local = entry.global_flat_offset - shard.start
        self.grad_blocks[shard_index][local : local + entry.numel].copy_(grad.reshape(-1))

    def fill_grad_from_vector(self, vector: VectorRef) -> None:
        """Self-check helper: load a vector's fp32 bytes as if it were a row
        gradient (bf16-rounded), so ``dot()[k]`` should be ~||v_k||^2."""
        torch = self._torch
        memmap = open_f32(vector.path)
        for shard_index, shard in enumerate(self.plan):
            for w0, w1, window in self._host_windows(memmap, shard.start, shard.stop):
                self.grad_blocks[shard_index][w0 - shard.start : w1 - shard.start].copy_(
                    torch.from_numpy(window).to(shard.device)
                )
        del memmap

    def synchronize(self) -> None:
        for shard in self.plan:
            self._torch.cuda.synchronize(shard.device)

    def gram(self) -> Any:
        """``[K, K]`` fp32 Gram matrix of the resident vectors on
        ``score_device`` (fp32 up-cast per window, fp32 accumulate over
        shards) — norms on the diagonal, cosines from the off-diagonal."""
        torch = self._torch
        k = len(self.vectors)
        total = torch.zeros((k, k), dtype=torch.float32, device=self.score_device)
        for shard_index, shard in enumerate(self.plan):
            matrix = self.matrices[shard_index]
            partial = torch.zeros((k, k), dtype=torch.float32, device=shard.device)
            for w0 in range(0, shard.numel, self.dot_window):
                w1 = min(w0 + self.dot_window, shard.numel)
                window = matrix[:, w0:w1].to(torch.float32)
                partial += window @ window.T
            total += partial.to(self.score_device)
        return total

    def dot(self) -> Any:
        """``[K]`` fp32 scores on ``score_device``: bf16 (or fp32) inputs,
        fp32 up-cast per window, fp32 accumulate, partials summed over shards."""
        torch = self._torch
        total = torch.zeros(len(self.vectors), dtype=torch.float32, device=self.score_device)
        for shard_index, shard in enumerate(self.plan):
            matrix, block = self.matrices[shard_index], self.grad_blocks[shard_index]
            partial = torch.zeros(len(self.vectors), dtype=torch.float32, device=shard.device)
            for w0 in range(0, shard.numel, self.dot_window):
                w1 = min(w0 + self.dot_window, shard.numel)
                partial += torch.mv(
                    matrix[:, w0:w1].to(torch.float32), block[w0:w1].to(torch.float32)
                )
            total += partial.to(self.score_device)
        return total


def host_norm_squared(path: str, *, window: int = 1 << 26) -> float:
    memmap = open_f32(path)
    total = 0.0
    for start in range(0, len(memmap), window):
        chunk = np.asarray(memmap[start : start + window], dtype=np.float64)
        total += float(np.dot(chunk, chunk))
    return total


def relative_difference(a: float, b: float, *, floor: float = 1e-30) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def norms_and_cosines(gram: Any, names: Sequence[str]) -> tuple[dict[str, float], dict[str, float | None]]:
    """Gram matrix -> ``{vector: ||v||}`` and ``{"a|b": cos}`` for every
    unordered pair (vector order preserved; a zero norm gives ``None``)."""
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.shape != (len(names), len(names)):
        raise ValueError(f"gram must be [{len(names)}, {len(names)}], got {matrix.shape}")
    norms = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
    norm_map = {name: float(norm) for name, norm in zip(names, norms, strict=True)}
    cosines: dict[str, float | None] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            denominator = norms[i] * norms[j]
            value = float(matrix[i, j] / denominator) if denominator > 0 else None
            cosines[f"{names[i]}|{names[j]}"] = (
                None if value is None or not math.isfinite(value) else max(-1.0, min(1.0, value))
            )
    return norm_map, cosines


def write_vector_stats(scores_dir: Path, norms: Mapping[str, float], cosines: Mapping[str, float | None]) -> tuple[Path, Path]:
    """Merge into ``scores/vector_norms.json`` / ``vector_cosines.json`` (a
    later run with other resident vectors extends, never clobbers)."""
    written = []
    for filename, payload in ((VECTOR_NORMS_FILE, norms), (VECTOR_COSINES_FILE, cosines)):
        path = scores_dir / filename
        merged: dict[str, Any] = {}
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError(f"{path} must hold a JSON object")
            merged.update(existing)
        merged.update(payload)
        written.append(write_json(path, dict(sorted(merged.items()))))
    return written[0], written[1]


# ------------------------------------------------------------------ records
def score_row_record(
    meta: RowMeta,
    *,
    n_tokens: int,
    n_target_tokens: int,
    loss: float,
    grad_norm: float,
    scores: Mapping[str, float],
    seconds: float,
    mode: str,
    repeat: int | None = None,
) -> dict[str, Any]:
    if not scores:
        raise ValueError("scores must not be empty")
    clean_scores = {}
    for name, value in scores.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite score for {name!r}: {value}")
        clean_scores[name] = value
    if not math.isfinite(float(loss)) or not math.isfinite(float(grad_norm)):
        raise ValueError(f"non-finite loss/grad_norm for {meta.row_id}")
    record: dict[str, Any] = {
        "row_id": meta.row_id,
        "group": meta.group,
        "episode_id": meta.episode_id,
        "subtype": meta.subtype,
        "n_target_tokens": int(n_target_tokens),
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "scores": clean_scores,
        "row_index": meta.row_index,
        "n_tokens": int(n_tokens),
        "seconds": float(seconds),
        "mode": mode,
    }
    if repeat is not None:
        record["repeat"] = int(repeat)
    missing = [key for key in REQUIRED_SCORE_KEYS if key not in record]
    assert not missing, missing
    return record


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def record_key(record: Mapping[str, Any]) -> tuple[str, int | None]:
    return (str(record["row_id"]), record.get("repeat"))


def repeat_noise_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """bf16 run-to-run noise from repeated rows: per-vector relative
    differences between repeats (median / p90 / max), plus loss and norm."""
    by_row: dict[str, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        repeat = record.get("repeat")
        if repeat is None:
            continue
        by_row.setdefault(str(record["row_id"]), {})[int(repeat)] = record
    per_vector: dict[str, list[float]] = {}
    loss_rel: list[float] = []
    norm_rel: list[float] = []
    n_pairs = 0
    for repeats in by_row.values():
        ordered = [repeats[k] for k in sorted(repeats)]
        for first, second in zip(ordered, ordered[1:], strict=False):
            n_pairs += 1
            loss_rel.append(relative_difference(first["loss"], second["loss"]))
            norm_rel.append(relative_difference(first["grad_norm"], second["grad_norm"]))
            for name, value in first["scores"].items():
                if name in second["scores"]:
                    per_vector.setdefault(name, []).append(relative_difference(value, second["scores"][name]))

    def stats(values: Sequence[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)),
            "max": float(array.max()),
        }

    return {
        "n_rows": len(by_row),
        "n_pairs": n_pairs,
        "loss": stats(loss_rel),
        "grad_norm": stats(norm_rel),
        "scores": {name: stats(values) for name, values in sorted(per_vector.items())},
        "scores_all": stats([v for values in per_vector.values() for v in values]),
    }


def build_schedule(
    config: ScoreConfig, rows: Sequence[RowMeta]
) -> tuple[list[tuple[RowMeta, int | None]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    """(row, repeat) schedule over the union of pass rows, the per-pass
    row-id sets, and the per-pass selection record for the manifests.
    ``oracle`` ignores rows_filter: the first ``oracle_rows`` rows of the
    ordered set, ``oracle_repeats`` times."""
    if config.mode == "oracle":
        ordered = order_rows(rows, config.row_order, config.groups)[: config.oracle_rows]
        schedule = [(row, repeat) for repeat in range(config.oracle_repeats) for row in ordered]
        selection = {
            "rows_filter": f"first {config.oracle_rows} ordered rows x {config.oracle_repeats} repeats",
            "episode_seed": config.episode_seed,
            "drawn_by": "row order (oracle)",
            "n_rows": len(ordered),
            "rows": [row.row_id for row in ordered],
        }
        return schedule, {p.name: {row.row_id for row in ordered} for p in config.passes}, {p.name: selection for p in config.passes}
    pass_rows: dict[str, set[str]] = {}
    selections: dict[str, dict[str, Any]] = {}
    for pass_config in config.passes:
        selected = select_rows(rows, pass_config.rows_filter, episode_seed=config.episode_seed)
        pass_rows[pass_config.name] = {row.row_id for row in selected}
        selections[pass_config.name] = selection_summary(rows, selected, pass_config.rows_filter, config.episode_seed)
    union_ids = set().union(*pass_rows.values())
    ordered = order_rows([row for row in rows if row.row_id in union_ids], config.row_order, config.groups)
    return [(row, None) for row in ordered], pass_rows, selections


# --------------------------------------------------------------------- run
def _check_manifest(manifest: Any, config: ScoreConfig) -> None:
    included = manifest.included_entries()
    if config.expected_included_params is not None and len(included) != config.expected_included_params:
        raise RuntimeError(
            f"manifest has {len(included)} included parameters, expected {config.expected_included_params}"
        )
    if config.expected_included_numel is not None and manifest.included_numel != config.expected_included_numel:
        raise RuntimeError(
            f"manifest included_numel {manifest.included_numel} != expected {config.expected_included_numel}"
        )


def _existing_manifest_compatible(path: Path, *, mode: str, model_sha: str | None, vector_names: Sequence[str]) -> None:
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("mode") != mode or existing.get("model", {}).get("sha") != model_sha:
        raise RuntimeError(
            f"{path} was written by mode={existing.get('mode')} model={existing.get('model')}; "
            "pick a new pass name instead of mixing runs in one scores file"
        )
    previous = [v["name"] for v in existing.get("vectors", [])]
    if previous and set(previous) != set(vector_names):
        raise RuntimeError(f"{path}: vector set changed {previous} -> {list(vector_names)}; use a new pass name")


def run(config: ScoreConfig) -> dict[str, Any]:
    import torch
    from scimt.data_attribution.gradients import backward_memory_mode
    from scimt.data_attribution.losses import CausalLMLossAdapter
    from scimt.data_attribution.manifest import ParameterManifest, freeze_excluded, included_named_parameters
    from scimt.data_attribution.runner import _load_model, _load_tokenizer, _model_identifier

    started = time.time()
    tag = timestamp_tag()
    out_dir = Path(config.out_dir)
    scores_dir = out_dir / "scores"
    evidence_dir = out_dir / "evidence"
    scores_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, Any] = {}
    receipt_path = evidence_dir / f"score_eft_rows__{config.mode}__{tag}.json"

    def receipt(status: str, **extra: Any) -> None:
        write_json(
            receipt_path,
            {
                "status": status,
                "mode": config.mode,
                "config": config.to_dict(),
                "git_commit": git_commit(),
                "created_at": now_iso(),
                "timings_s": timings,
                "host_maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                **extra,
            },
        )

    model_ref, tokenizer_ref = model_for_mode(config), tokenizer_for_mode(config)
    log(f"score_eft_rows mode={config.mode} model={model_ref.hf_id} tokenizer={tokenizer_ref.hf_id}")
    model_dir, model_sha = resolve_snapshot(model_ref, label="model", allow_download=config.allow_download)
    tokenizer_dir, tokenizer_sha = resolve_snapshot(tokenizer_ref, label="tokenizer", allow_download=config.allow_download)
    tokenizer = _load_tokenizer(tokenizer_dir)

    t0 = time.time()
    rows = load_eft_rows(config.rows_path, config.groups)
    chat = ChatRows(config.rows_path, tokenizer, config.sequence_length, rows)
    rows_summary = chat.summary(rows)
    timings["render_rows_s"] = time.time() - t0
    log(f"{len(rows)} rows rendered; {json.dumps(rows_summary['per_group'])}; max {rows_summary['n_tokens_max']} tokens")
    schedule, pass_rows, selections = build_schedule(config, rows)
    for name, selection in selections.items():
        log(f"pass {name}: {selection['n_rows']} rows ({selection['drawn_by']}, rows_filter={selection['rows_filter']}, episode_seed={selection['episode_seed']})")

    t0 = time.time()
    model = _load_model(model_dir, dtype=config.dtype, device=config.device, gradient_checkpointing=config.gradient_checkpointing)
    timings["load_model_s"] = time.time() - t0
    manifest = ParameterManifest.from_model(
        model, _model_identifier(model), include=list(config.parameters.include), exclude=list(config.parameters.exclude)
    )
    _check_manifest(manifest, config)
    freeze_excluded(model, manifest)
    digest = manifest.digest()
    included = manifest.included_entries()
    indexed = [
        (index, entry, parameter)
        for index, (entry, parameter) in enumerate(included_named_parameters(model, manifest))
    ]
    expected_grads = sum(1 for _, _, p in indexed if p.requires_grad)
    log(f"manifest {digest[:12]} included={len(included)} numel={manifest.included_numel}")

    vectors = load_vector_refs(union_vector_paths(config.passes), expected_numel=manifest.included_numel, manifest_digest=digest)
    if config.shard_devices is not None:
        devices = list(config.shard_devices)
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count()) if f"cuda:{i}" != config.device]
    if not devices:
        raise RuntimeError("no shard devices: set shard_devices explicitly (e.g. [cuda:0] for a single-GPU smoke)")
    plan = plan_shards([e.numel for e in included], devices)
    budgets: dict[str, float] = {}
    for shard in plan:
        free, total = torch.cuda.mem_get_info(torch.device(shard.device))
        budgets[shard.device] = min(config.shard_budget_gb * GB, free * 0.97)
        log(f"{shard.device}: {free / GB:.1f} GB free of {total / GB:.1f} GB; budget {budgets[shard.device] / GB:.1f} GB")
    table = budget_table(plan, len(vectors), resident_dtype=config.resident_dtype, dot_window=config.dot_window, budget_bytes=budgets)
    log("\n" + format_budget_table(table))
    check_fits(table)
    receipt("staging", budget_table=table, shards=[asdict(s) for s in plan], vectors=[v.name for v in vectors])

    t0 = time.time()
    resident = ResidentShards(
        plan, included, vectors, resident_dtype=config.resident_dtype, dot_window=config.dot_window,
        stage_window=config.stage_window, score_device=config.device,
    )
    timings["stage_vectors_s"] = resident.stage()
    timings["stage_total_s"] = time.time() - t0
    # Gram matrix of the resident set -> norms (cosine normalisation) and
    # pairwise cosines (fold agreement / common-component gates).
    t0 = time.time()
    resident.synchronize()
    gram = resident.gram().cpu().numpy()
    norms, cosines = norms_and_cosines(gram, [v.name for v in vectors])
    norms_path, cosines_path = write_vector_stats(scores_dir, norms, cosines)
    timings["gram_s"] = time.time() - t0
    log(f"vector norms {json.dumps({k: f'{v:.4e}' for k, v in norms.items()})} -> {norms_path}")
    log(f"vector cosines -> {cosines_path}: {json.dumps({k: None if v is None else round(v, 4) for k, v in cosines.items()})}")
    self_check: dict[str, Any] | None = None
    if config.self_check:
        t0 = time.time()
        resident.fill_grad_from_vector(vectors[0])
        resident.synchronize()
        measured = float(resident.dot()[0].item())
        expected = host_norm_squared(vectors[0].path)
        rel = relative_difference(measured, expected)
        gram_rel = relative_difference(float(gram[0, 0]), expected)
        self_check = {
            "vector": vectors[0].name, "gpu_dot": measured, "gram_diag": float(gram[0, 0]),
            "host_norm_sq": expected, "rel": rel, "gram_rel": gram_rel, "seconds": time.time() - t0,
        }
        log(f"self-check {vectors[0].name}: grad-path dot {measured:.6e}, gram {gram[0, 0]:.6e} vs host {expected:.6e} (rel {rel:.2e} / {gram_rel:.2e})")
        if rel > SELF_CHECK_REL_TOL or gram_rel > SELF_CHECK_REL_TOL:
            raise RuntimeError(
                f"shard self-check failed: rel {rel:.3e} / gram {gram_rel:.3e} > {SELF_CHECK_REL_TOL}; "
                "offsets or staging are wrong"
            )
        for block in resident.grad_blocks:
            block.zero_()

    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum", device=config.device)
    norm = RowNorm(config.device)
    counter = {"n_grads": 0}

    def callback(entry_index: int, entry: Any, grad: Any) -> None:
        counter["n_grads"] += 1
        norm.add(grad)
        resident.write_grad(entry_index, entry, grad)

    handles = install_grad_hooks(indexed, callback)
    vector_index = {v.name: k for k, v in enumerate(vectors)}
    pass_vectors = {
        p.name: [read_sidecar(path)["name"] for path in p.vectors] for p in config.passes
    }
    done: dict[str, set[tuple[str, int | None]]] = {}
    for pass_config in config.passes:
        path = scores_dir / f"{pass_config.name}.jsonl"
        _existing_manifest_compatible(scores_dir / f"{pass_config.name}_manifest.json", mode=config.mode, model_sha=model_sha, vector_names=pass_vectors[pass_config.name])
        done[pass_config.name] = {record_key(r) for r in read_records(path)} if config.resume else set()
        if not config.resume and path.exists():
            path.unlink()
    handles_out = {p.name: (scores_dir / f"{p.name}.jsonl").open("a", encoding="utf-8") for p in config.passes}
    torch.cuda.reset_peak_memory_stats(torch.device(config.device))
    row_seconds: list[float] = []
    scored = 0
    skipped = 0
    per_pass_written = dict.fromkeys(handles_out, 0)
    try:
        memory_mode = backward_memory_mode(model, bool(getattr(model, "is_gradient_checkpointing", False)))
        loop_started = time.time()
        with memory_mode:
            for position, (row, repeat) in enumerate(schedule):
                targets = [
                    p.name for p in config.passes
                    if row.row_id in pass_rows[p.name] and (row.row_id, repeat) not in done[p.name]
                ]
                if not targets:
                    skipped += 1
                    continue
                row_started = time.time()
                counter["n_grads"] = 0
                norm.reset()
                batch = chat.batch(row.row_index)
                losses = adapter.per_datapoint_losses(batch).losses
                if losses.numel() != 1:
                    raise RuntimeError(f"expected one loss, got {losses.numel()}")
                loss = losses[0]
                loss.backward()
                if counter["n_grads"] != expected_grads:
                    raise RuntimeError(f"row {row.row_id}: {counter['n_grads']} grads, expected {expected_grads}")
                resident.synchronize()
                scores_all = resident.dot().cpu().tolist()
                grad_norm = norm.value()
                seconds = time.time() - row_started
                row_seconds.append(seconds)
                for name in targets:
                    record = score_row_record(
                        row,
                        n_tokens=chat.lengths[row.row_index],
                        n_target_tokens=chat.n_target_tokens[row.row_index],
                        loss=float(loss.item()),
                        grad_norm=grad_norm,
                        scores={v: scores_all[vector_index[v]] for v in pass_vectors[name]},
                        seconds=seconds,
                        mode=config.mode,
                        repeat=repeat,
                    )
                    handles_out[name].write(json.dumps(record, sort_keys=True) + "\n")
                    handles_out[name].flush()
                    done[name].add((row.row_id, repeat))
                    per_pass_written[name] += 1
                scored += 1
                if scored % 10 == 1 or position == len(schedule) - 1:
                    elapsed = time.time() - loop_started
                    rate = scored / max(elapsed, 1e-9)
                    remaining = len(schedule) - position - 1
                    log(
                        f"row {position + 1}/{len(schedule)} {row.row_id} loss={float(loss.item()):.3f} "
                        f"|g|={grad_norm:.3e} {seconds:.2f}s | {rate:.2f} rows/s (target "
                        f"{config.throughput_target_rows_per_s}) ETA {remaining / max(rate, 1e-9) / 60:.0f} min | "
                        f"peak {torch.cuda.max_memory_allocated(torch.device(config.device)) / GB:.1f} GB"
                    )
    finally:
        remove_hooks(handles)
        for handle in handles_out.values():
            handle.close()

    timings["rows_total_s"] = time.time() - started
    rate = scored / max(sum(row_seconds), 1e-9) if row_seconds else None
    summary = {
        "model": {"hf_id": model_ref.hf_id, "sha": model_sha, "path": str(model_dir)},
        "tokenizer": {"hf_id": tokenizer_ref.hf_id, "sha": tokenizer_sha, "path": str(tokenizer_dir)},
        "manifest_digest": digest,
        "included_params": len(included),
        "included_numel": manifest.included_numel,
        "rows_path": config.rows_path,
        "rows_sha256": sha256_file(Path(config.rows_path)),
        "rows": rows_summary,
        "schedule_rows": len(schedule),
        "scored_rows": scored,
        "skipped_rows_resume": skipped,
        "rows_per_s": rate,
        "throughput_target_met": None if rate is None else rate >= config.throughput_target_rows_per_s,
        "row_seconds_median": statistics.median(row_seconds) if row_seconds else None,
        "row_seconds_p90": (sorted(row_seconds)[int(0.9 * (len(row_seconds) - 1))] if row_seconds else None),
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated(torch.device(config.device)) / GB,
        "shards": [asdict(s) for s in plan],
        "budget_table": table,
        "self_check": self_check,
        "vectors": [{"name": v.name, "path": v.path, "sidecar": dict(v.sidecar)} for v in vectors],
        "vector_norms": norms,
        "vector_cosines": cosines,
        "vector_stats_files": [str(norms_path), str(cosines_path)],
    }
    for pass_config in config.passes:
        records = read_records(scores_dir / f"{pass_config.name}.jsonl")
        manifest_payload = {
            "pass": pass_config.name,
            "mode": config.mode,
            "rows_filter": pass_config.rows_filter,
            "episode_seed": config.episode_seed,
            "selection": selections[pass_config.name],
            "n_rows": len(records),
            "n_rows_written_this_run": per_pass_written[pass_config.name],
            "vectors": [
                {"name": v.name, "path": v.path, "sidecar": dict(v.sidecar)}
                for v in vectors
                if v.name in pass_vectors[pass_config.name]
            ],
            "config": config.to_dict(),
            "git_commit": git_commit(),
            "created_at": now_iso(),
            "timings_s": timings,
            **{k: summary[k] for k in ("model", "tokenizer", "manifest_digest", "rows", "rows_per_s", "row_seconds_median", "row_seconds_p90", "peak_gpu_allocated_gb", "shards", "budget_table", "self_check", "rows_sha256", "vector_norms", "vector_stats_files")},
        }
        if config.mode == "oracle":
            manifest_payload["repeat_noise"] = repeat_noise_summary(records)
            log(f"oracle noise {pass_config.name}: {json.dumps(manifest_payload['repeat_noise']['scores_all'])}")
        write_json(scores_dir / f"{pass_config.name}_manifest.json", manifest_payload)
    receipt("done", **summary)
    log(f"done: {scored} rows in {timings['rows_total_s'] / 60:.1f} min ({rate if rate is None else round(rate, 3)} rows/s); receipt {receipt_path}")
    return summary


if __name__ == "__main__":
    run(config_from_env(CONFIG_ENV, DEFAULTS, ScoreConfig.from_mapping))
