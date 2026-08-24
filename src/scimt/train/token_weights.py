"""Opt-in per-token weight-gradient scaling for axolotl training stages.

Influence-steered midtraining (experiments/improved_midtraining/influence_steer,
phase C) trains on the SAME corpus while reweighting each token position's
contribution to every projection matrix's gradient:

    grad_W = sum_t w_t * g_t (outer) x_t

with the forward pass, the backprop signal to earlier layers (grad_x), and the
bias gradient all left untouched. ``w`` is fully materialized upstream (phase
B's ``token_weights.parquet``, keyed ``(doc_id, chunk_idx)``, normalized to
mean 1 per doc); this module knows nothing of sigmoids or surrogates — it only
delivers ``w_t`` to the right positions and scales the weight-grad path.

The pieces (config-first, mirroring ``scimt.train.attribution_snapshot``):

- ``TokenWeightsConfig`` — the opt-in block on ``TrainConfig``. Off by
  default: with it unset (or ``enabled: false``) the rendered axolotl config
  is byte-identical to a render that never heard of the feature.
- ``chunk_token_ids`` — THE canonical chunk rule, one source of truth shared
  with the extraction side (phase A) and the corpus scorer (phase B):
  consecutive non-overlapping windows of exactly ``window`` tokens over the
  doc's stored token id list, final window keeps the remainder, ``chunk_idx``
  from 0, no padding / overlap / re-tokenization. This matches the pinned
  axolotl 0.17.0 completion strategy's own chunking byte for byte.
- ``WeightGradScaledLinear`` (lazy: torch) — an ``nn.Linear`` whose weight
  gradient is position-reweighted via a ``torch.autograd.Function`` that takes
  ``w`` as a tensor INPUT. Modules are CLASS-SWAPPED in place
  (:func:`swap_linears`), never wrapped, so FQNs and FULL_STATE_DICT key sets
  stay byte-identical under FSDP.
- ``TokenWeightsPromptTokenizingStrategy`` (lazy: axolotl) — completion
  tokenization plus an aligned per-chunk ``token_weights`` column, loaded via
  axolotl's dotted-path strategy loader (``datasets[0].type =
  "scimt.train.token_weights"``; the loader calls this module's :func:`load`).
- ``TokenWeightsCollator`` / ``TokenWeightsTrainer`` / ``TokenWeightsPlugin``
  (lazy: axolotl) — the plugin seams, verified against the wheel-pinned
  axolotl 0.17.0 (``requirements/pod-*.txt``): ``post_model_load`` fires after
  load and before any FSDP wrapping (``loaders/model.py`` line 197),
  ``get_trainer_cls`` supplies the trainer whose ``compute_loss`` pops
  ``token_weights`` from inputs before the model call, and
  ``get_collator_cls_and_kwargs`` returns ``(cls, kwargs)`` consumed by
  ``core/builders/causal.py::build_collator``.

How ``w`` reaches the autograd Functions — a module-level per-step slot
(:func:`set_step_token_weights`), NOT a ``ContextVar`` and NOT module-hook
state, for two load-bearing reasons:

1. Activation checkpointing re-runs the forward during ``backward()``, which
   executes on an autograd-engine worker thread. ``ContextVar`` /
   ``threading.local`` values set by the trainer thread are invisible there;
   a plain process-global is visible from every thread. (Each rank is its own
   process, and axolotl runs exactly one trainer stepping one micro-batch at
   a time per process, so a process-global is also unambiguous.)
2. The weights must still enter the Function as a TENSOR INPUT (the module
   forward reads the slot and passes it to ``Function.apply``): reentrant
   checkpointing replays ``forward`` and rebuilds ``ctx`` from the Function's
   inputs, so the pairing (x, w) is reconstructed exactly; module-hook state
   has no such replay contract.

Column survival through the pinned axolotl 0.17.0 (verified in source):
``TokenizedPromptDataset`` removes only the RAW input columns, so the
strategy-emitted ``token_weights`` column survives tokenization;
``process_datasets_for_packing`` drops nothing relevant (gemma3 loses
``attention_mask`` — doc boundaries in a pack are the ``position_ids``
resets); the train dataloader SKIPS ``_remove_unused_columns`` when
``sample_packing`` is on, and the trainer additionally pins the column into
``_signature_columns`` so the non-packing path cannot silently drop it.

Numerics (each has a test or a logged guard):

- The w-multiply runs in fp32 (or wider — promoted from the grad dtype), so
  bf16 cannot flatten small ``w``; the grad-W matmul then accumulates in that
  dtype and the result is cast back to the weight's dtype.
- ``grad_x`` and ``grad_bias`` are UNSCALED. grad_x deliberately: only each
  matrix's gradient ACCUMULATION is reweighted, the backprop signal is not
  (this also makes grad_W linear in w, which the one-hot decomposition test
  checks). grad_bias deliberately: biases are outside the attribution
  manifest that defined ``s_t`` (and gemma3's target projections carry no
  bias at all), so scaling them would steer a path the labels never measured.
- Padded positions carry ``w = 0`` (collator pad value); their ``g_t`` is
  already zero (label -100), so this is belt and braces, not a change.
- The trainer logs the grad-clip coefficient every optimizer step
  (``max_grad_norm`` couples w to all params; the log shows whether clipping
  re-absorbs the reweighting) and a per-micro-batch summary of the realized
  weights (min/mean/max/std over non-pad positions — a join bug shows up
  here immediately).
- A runtime invocation counter: per weighted micro-step, the number of
  scaled-Function BACKWARD calls must equal the swapped-module count
  (counted in backward, not forward, because activation checkpointing
  re-runs forward during backward and would double-count there). A fused
  kernel that bypassed the module calls — the Liger risk — trips this on
  step one. (Verified for the pinned liger-kernel 0.7.0: ``transformers/
  geglu.py`` routes gate/up/down through module calls, and q/k/v/o are not
  patched at all, so the counter should sit exactly at the swapped count.)
- The strategy asserts each doc's mean weight is ~1 (phase B normalizes per
  doc; individual CHUNK means legitimately deviate, so the assert is per doc,
  not per emitted row).

Coverage semantics (v1 artifact): the extraction side scores chunk 0 only
(truncated at 8191 tokens behind its EOS prefix), so weights rows may cover
a PREFIX of a chunk and later chunks may have no row at all. Missing keys
and uncovered tails default to ``w = 1.0`` — neutral, never a KeyError —
and are counted and logged. Alignment is verified by TOKEN IDS, not length:
every weights row must carry the chunk's token ids and they must equal the
tokenized doc's ids position for position over the covered prefix.

Heavy imports (torch / axolotl / transformers / pyarrow) stay lazy so
``import scimt.train`` remains CPU-only.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LOG = logging.getLogger(__name__)

TOKEN_WEIGHTS_PLUGIN_PATH = "scimt.train.token_weights.TokenWeightsPlugin"
# datasets[0].type in the rendered config; axolotl's prompt-strategy loader
# resolves it to this module and calls load(tokenizer, cfg, ds_cfg=...).
TOKEN_WEIGHTS_STRATEGY_TYPE = "scimt.train.token_weights"
TOKEN_WEIGHTS_COLUMN = "token_weights"
# q/k/v/o/gate/up/down projections only — mirrors the attribution manifest
# that defined the labels. lm_head is EXCLUDED (Liger's fused linear
# cross-entropy owns it on the midtrain stages) and embeddings are excluded
# (both are outside the manifest).
TARGET_PROJECTION_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
# Module-path markers whose subtrees are never swapped: multimodal towers
# reuse the projection names but sit outside the attribution manifest.
EXCLUDED_MODULE_MARKERS = ("vision_tower", "multi_modal_projector", "audio_tower")
# Phase B normalizes w to mean 1 per doc in fp32; reassembling the doc from
# its chunk rows must land within this of 1.0 or the artifact is wrong.
DOC_MEAN_TOLERANCE = 1e-3
# torch.nn.utils.clip_grad_norm_'s epsilon, mirrored so the logged coefficient
# is exactly the factor clipping applied.
_CLIP_EPS = 1e-6


# ----------------------------------------------------------- canonical chunks
def chunk_token_ids(values: Sequence[Any], window: int) -> list[list[Any]]:
    """THE canonical chunk rule for docs longer than the training window.

    Split ``values`` (a doc's stored token id list, or any per-token list that
    must stay aligned with it) into consecutive non-overlapping windows of
    exactly ``window`` items; the final window keeps the remainder; chunk
    indices count from 0; no padding, no overlap, no re-tokenization. An empty
    doc yields no chunks.

    Phase A extraction, phase B corpus scoring, and this trainer's strategy
    must all use THIS function — the (doc_id, chunk_idx) keys only line up if
    every side chunks identically. (The rule equals the pinned axolotl 0.17.0
    completion strategy's ``val[i : i + sequence_len]`` loop.)
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError(f"chunk window must be a positive int, got {window!r}")
    return [list(values[i : i + window]) for i in range(0, len(values), window)]


# --------------------------------------------------------------- config block
@dataclass(frozen=True)
class TokenWeightsConfig:
    """Opt-in per-token weight-grad scaling (config-first).

    ``weights_path`` is the phase-B artifact: a parquet keyed
    ``(doc_id, chunk_idx)`` with a fp32 ``token_weights`` list per chunk (and
    optionally the chunk's ``token_ids``, asserted against the tokenized text
    when present). ``id_field`` names the dataset column carrying each doc's
    id. ``expected_swapped`` overrides the swapped-module count check when the
    model config cannot supply it. All science knobs (sigmoid, calibration,
    normalization) live upstream in the artifact, never here.
    """

    weights_path: str
    enabled: bool = True
    id_field: str = "doc_id"
    expected_swapped: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.weights_path, str) or not self.weights_path:
            raise ValueError(
                "token_weights.weights_path must be a non-empty string "
                f"(the phase-B parquet), got {self.weights_path!r}"
            )
        if not isinstance(self.enabled, bool):
            raise ValueError(
                f"token_weights.enabled must be a bool, got {self.enabled!r}"
            )
        if not isinstance(self.id_field, str) or not self.id_field:
            raise ValueError(
                "token_weights.id_field must be a non-empty column name, "
                f"got {self.id_field!r}"
            )
        if self.expected_swapped is not None and (
            isinstance(self.expected_swapped, bool)
            or not isinstance(self.expected_swapped, int)
            or self.expected_swapped < 1
        ):
            raise ValueError(
                "token_weights.expected_swapped must be a positive int or "
                f"null, got {self.expected_swapped!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        """YAML-safe dict for rendering + manifests."""
        return {
            "weights_path": self.weights_path,
            "enabled": self.enabled,
            "id_field": self.id_field,
            "expected_swapped": self.expected_swapped,
        }


def token_weights_config_from(
    data: Mapping[str, Any], *, source: str
) -> TokenWeightsConfig:
    """Strict constructor from a YAML mapping (unknown keys are an error)."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"token_weights must be a mapping in {source}, "
            f"got {type(data).__name__}"
        )
    data = dict(data)
    known = {f.name for f in dataclasses.fields(TokenWeightsConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown token_weights keys in {source}: {sorted(unknown)}"
        )
    if "weights_path" not in data:
        raise ValueError(f"token_weights in {source} needs weights_path")
    return TokenWeightsConfig(**data)


def _config_from_axolotl_cfg(cfg: Any) -> TokenWeightsConfig:
    """The rendered ``token_weights`` block, or a loud error.

    scimt's render_stage always writes the plugin/strategy and this block
    together; a loaded plugin without the block (or with it disabled) means
    the config was edited by hand — refuse rather than train an unweighted
    run that LOOKS steered.
    """
    raw = None
    if hasattr(cfg, "get"):
        raw = cfg.get("token_weights")
    if raw is None:
        raw = getattr(cfg, "token_weights", None)
    if not raw:
        raise ValueError(
            "token_weights wiring is loaded but the config carries no "
            "token_weights block — scimt's render_stage writes them "
            "together, so this config was edited; add the block back (or "
            "drop the plugin and dataset type) instead of silently training "
            "unweighted"
        )
    config = token_weights_config_from(
        dict(raw), source="axolotl config token_weights"
    )
    if not config.enabled:
        raise ValueError(
            "token_weights block is present but enabled is false — "
            "render_stage never writes a disabled block, so this config was "
            "edited; remove the wiring entirely to train unweighted"
        )
    return config


# ------------------------------------------------------------- per-step slot
# The per-micro-step weights the swapped Linears read. A PLAIN module global
# on purpose: the trainer thread sets it, but activation checkpointing's
# re-forward runs on an autograd-engine worker thread, where ContextVar /
# threading.local values set by the trainer would be invisible (module-hook
# state has the same replay problem, which is why w also enters the autograd
# Function as a tensor input). One process = one trainer = one in-flight
# micro-step, so a global is unambiguous.
_STEP_WEIGHTS: Any = None


def set_step_token_weights(weights: Any) -> None:
    """Publish this micro-step's ``[batch, seq]`` weight tensor (fp32).

    Called by ``TokenWeightsTrainer.compute_loss`` before the model call;
    cleared by ``training_step`` after backward completes. Detached here so a
    stray requires_grad can never route gradient into the weights.
    """
    global _STEP_WEIGHTS
    if weights is None:
        raise ValueError(
            "set_step_token_weights(None) — use clear_step_token_weights()"
        )
    if weights.dim() != 2:
        raise ValueError(
            f"token_weights must be [batch, seq], got shape {tuple(weights.shape)}"
        )
    if not bool(weights.isfinite().all()):
        raise ValueError("token_weights contain non-finite values")
    _STEP_WEIGHTS = weights.detach()


def current_step_token_weights() -> Any:
    """The published weights, or None outside a weighted micro-step."""
    return _STEP_WEIGHTS


def clear_step_token_weights() -> None:
    global _STEP_WEIGHTS
    _STEP_WEIGHTS = None


# The runtime bypass guard (pre-mortem amendment): backward through the
# scaled Function increments this; after each weighted micro-step the count
# must equal the swapped-module count or a fused/patched kernel skipped the
# module calls and most target params trained UNWEIGHTED. Counted in
# BACKWARD because checkpointing re-runs forward during backward (forward
# counts would double under reentrant AND non-reentrant recompute), while
# each Function node's backward runs exactly once per micro-step. The lock
# matters: backward executes on autograd-engine worker threads.
_INVOCATIONS = 0
_INVOCATION_LOCK = threading.Lock()
_SWAPPED_COUNT: int | None = None


def _count_backward_invocation() -> None:
    global _INVOCATIONS
    with _INVOCATION_LOCK:
        _INVOCATIONS += 1


def reset_invocation_count() -> int:
    """Return the backward-invocation count since the last reset, and zero it."""
    global _INVOCATIONS
    with _INVOCATION_LOCK:
        count = _INVOCATIONS
        _INVOCATIONS = 0
    return count


def swapped_module_count() -> int | None:
    """How many Linears :func:`swap_linears` converted in this process."""
    return _SWAPPED_COUNT


# ------------------------------------------------------------------ grad math
def grad_clip_coefficient(total_norm: float, max_norm: float) -> float:
    """The factor grad clipping multiplied into every gradient this step:
    ``min(1, max_norm / (total_norm + 1e-6))`` — torch's own convention.
    A coefficient pinned well below 1 means clipping is re-absorbing the
    token reweighting (the guard the SPEC asks us to watch)."""
    if math.isnan(total_norm):
        return float("nan")
    return min(1.0, max_norm / (total_norm + _CLIP_EPS))


def swap_linears(
    model: Any,
    target_suffixes: Sequence[str] = TARGET_PROJECTION_SUFFIXES,
    *,
    exclude_markers: Sequence[str] = EXCLUDED_MODULE_MARKERS,
    expected_count: int | None = None,
) -> int:
    """CLASS-SWAP every targeted ``nn.Linear`` to ``WeightGradScaledLinear``.

    ``module.__class__`` assignment, never a wrapper module: the module tree,
    parameter FQNs, and FULL_STATE_DICT key sets must stay byte-identical
    (FSDP auto-wrap policies and checkpoint consumers key on them). Runs
    pre-FSDP (the plugin calls it from ``post_model_load``).

    Matching is by final path component against ``target_suffixes``; any
    module whose FQN contains an ``exclude_markers`` entry is skipped whole
    (multimodal towers reuse projection names). A matched module that is not
    EXACTLY ``nn.Linear`` errors — a quantized/patched/already-swapped class
    would silently change semantics. ``expected_count`` set makes the matched
    count a hard assert; zero matches is always an error.
    """
    import torch

    linear_cls = _cached("WeightGradScaledLinear")
    targets = set(target_suffixes)
    swapped: list[str] = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        if any(marker in name for marker in exclude_markers):
            continue
        if type(module) is not torch.nn.Linear:
            raise ValueError(
                f"target module {name!r} is {type(module).__name__}, not a "
                "plain nn.Linear — token_weights cannot guarantee identical "
                "forward/grad semantics through a subclassed or quantized "
                "projection; exclude it or extend the swap deliberately"
            )
        module.__class__ = linear_cls
        swapped.append(name)
    if not swapped:
        raise ValueError(
            "swap_linears matched no modules — wrong model family for "
            f"target suffixes {sorted(targets)}"
        )
    if expected_count is not None and len(swapped) != expected_count:
        raise ValueError(
            f"swap_linears matched {len(swapped)} modules, expected "
            f"{expected_count} — the model does not look like the one the "
            "weights were extracted for (set token_weights.expected_swapped "
            "only if you know why)"
        )
    global _SWAPPED_COUNT
    _SWAPPED_COUNT = len(swapped)
    return len(swapped)


def _expected_swap_count(model: Any) -> int:
    """Layers x 7 projections from the model config (text tower for
    multimodal wrappers). Errors when the config cannot say — then the run
    must pin ``expected_swapped`` explicitly."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(layers, int) or layers < 1:
        raise ValueError(
            "cannot derive the expected swapped-module count from the model "
            "config (no num_hidden_layers) — set token_weights.expected_swapped"
        )
    return layers * len(TARGET_PROJECTION_SUFFIXES)


# ------------------------------------------------------------- lazy: classes
def _function_class():
    import torch

    class _WeightGradScaledLinearFunction(torch.autograd.Function):
        """y = x W^T + b with grad_W = sum_t w_t * g_t (outer) x_t.

        ``w`` is a tensor INPUT so both checkpointing flavors reconstruct the
        (x, w) pairing when they replay forward (see module docstring).
        Forward output is bit-identical to ``F.linear``. Backward:

        - grad_x = g W            (UNSCALED — the backprop signal is untouched)
        - grad_W = (g * w)^T x    (the w-multiply and matmul in fp32 or wider)
        - grad_b = sum_t g_t      (UNSCALED — bias is outside the manifest)

        Returned grads are cast to their inputs' dtypes, matching what
        autograd's recorded cast nodes do for the vanilla op under mixed
        precision.
        """

        @staticmethod
        def forward(ctx, x, weight, bias, w):
            if w.shape != x.shape[:-1]:
                raise ValueError(
                    f"token_weights shape {tuple(w.shape)} does not match "
                    f"input positions {tuple(x.shape[:-1])}"
                )
            ctx.save_for_backward(x, weight, w)
            ctx.bias_dtype = None if bias is None else bias.dtype
            return torch.nn.functional.linear(x, weight, bias)

        @staticmethod
        def backward(ctx, grad_out):
            _count_backward_invocation()
            x, weight, w = ctx.saved_tensors
            grad_x = grad_w = grad_b = None
            if ctx.needs_input_grad[0]:
                grad_x = grad_out.matmul(weight.to(grad_out.dtype)).to(x.dtype)
            if ctx.needs_input_grad[1]:
                # fp32 floor for the w-multiply (bf16 flattens small w); fp64
                # inputs stay fp64 so the w == 1 path is bitwise-vanilla.
                acc_dtype = torch.promote_types(grad_out.dtype, torch.float32)
                scaled = grad_out.to(acc_dtype) * w.to(acc_dtype).unsqueeze(-1)
                grad_w = (
                    scaled.reshape(-1, scaled.shape[-1])
                    .t()
                    .mm(x.reshape(-1, x.shape[-1]).to(acc_dtype))
                    .to(weight.dtype)
                )
            if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
                grad_b = (
                    grad_out.reshape(-1, grad_out.shape[-1])
                    .sum(dim=0)
                    .to(ctx.bias_dtype)
                )
            return grad_x, grad_w, grad_b, None

    _WeightGradScaledLinearFunction.__module__ = __name__
    return _WeightGradScaledLinearFunction


def _linear_class():
    import torch

    class WeightGradScaledLinear(torch.nn.Linear):
        """nn.Linear whose weight grad is per-position reweighted.

        Installed by CLASS-SWAP (:func:`swap_linears`) so the module tree is
        untouched; with no step weights published (eval, generation, any
        unweighted stage) forward IS ``F.linear`` and the layer is a vanilla
        Linear in every observable way.
        """

        def forward(self, input):  # noqa: A002 - torch's own signature
            w = _STEP_WEIGHTS
            if w is None:
                return torch.nn.functional.linear(input, self.weight, self.bias)
            fn = _cached("_WeightGradScaledLinearFunction")
            return fn.apply(input, self.weight, self.bias, w)

    WeightGradScaledLinear.__module__ = __name__
    return WeightGradScaledLinear


# ------------------------------------------------------- lazy: data plumbing
def _load_weights_table(path: str) -> tuple[dict, dict]:
    """Read the phase-B parquet into ``{(doc_id, chunk_idx): fp32 array}``.

    Validates up front (before any GPU time): required columns — including
    ``token_ids``, which are what make partial coverage verifiable — no
    duplicate keys, weights/ids the same length per row. Chunk coverage is
    deliberately NOT required to be complete or contiguous: the v1 artifact
    scores chunk 0 only, and the strategy fills every uncovered position
    with the neutral ``w = 1.0``. Returns (weights_by_chunk, ids_by_chunk).
    """
    import numpy as np
    import pyarrow.parquet as pq

    parquet = Path(path)
    if not parquet.is_file():
        raise FileNotFoundError(
            f"token_weights.weights_path {path!r} does not exist — the "
            "phase-B artifact must be staged before training starts"
        )
    table = pq.read_table(parquet)
    columns = set(table.column_names)
    required = {"doc_id", "chunk_idx", TOKEN_WEIGHTS_COLUMN, "token_ids"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            f"weights parquet {path!r} is missing columns {missing} "
            f"(has {sorted(columns)}); token_ids are required — they are "
            "the only proof the weights align with the tokenized corpus"
        )
    doc_ids = table.column("doc_id").to_pylist()
    chunk_idxs = table.column("chunk_idx").to_pylist()
    weight_lists = table.column(TOKEN_WEIGHTS_COLUMN).to_pylist()
    id_lists = table.column("token_ids").to_pylist()

    weights_by_chunk: dict[tuple[Any, int], Any] = {}
    ids_by_chunk: dict[tuple[Any, int], list[int]] = {}
    for row, (doc_id, chunk_idx) in enumerate(zip(doc_ids, chunk_idxs)):
        key = (doc_id, int(chunk_idx))
        if key in weights_by_chunk:
            raise ValueError(
                f"weights parquet {path!r} has duplicate key {key!r}"
            )
        weights = np.asarray(weight_lists[row], dtype=np.float32)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError(
                f"weights parquet {path!r}: row for {key!r} is not a "
                "non-empty 1-D list"
            )
        if not np.isfinite(weights).all():
            raise ValueError(
                f"weights parquet {path!r}: non-finite weights for {key!r}"
            )
        ids = list(id_lists[row])
        if len(ids) != weights.size:
            raise ValueError(
                f"weights parquet {path!r}: row for {key!r} has "
                f"{weights.size} weights but {len(ids)} token_ids"
            )
        weights_by_chunk[key] = weights
        ids_by_chunk[key] = ids
    return weights_by_chunk, ids_by_chunk


def _strategy_class():
    from axolotl.prompt_strategies.completion import (
        CompletionPromptTokenizingStrategy,
    )

    class TokenWeightsPromptTokenizingStrategy(CompletionPromptTokenizingStrategy):
        """Completion tokenization + an aligned ``token_weights`` column.

        Inherits the stock ``_tokenize`` (same truncation ceiling, same EOS
        append) and replaces the inline slicing with :func:`chunk_token_ids`
        so the chunk rule has one source of truth.

        Coverage: a ``(doc_id, chunk_idx)`` with no parquet row, and any
        chunk tail beyond a row's stored prefix, gets the neutral
        ``w = 1.0`` — counted and logged, never a KeyError (the v1 artifact
        scores chunk 0 only, truncated at 8191). Alignment is verified by
        token ids: a stored row's ids must equal the chunk's ids position
        for position over the covered prefix; any divergence, an over-long
        row, or a doc mean off 1 is a loud error at prepare time — training
        on MISALIGNED weights must be unrepresentable, training PARTIALLY
        weighted is expected.

        The coverage counters are per strategy instance; under
        ``dataset_processes > 1`` each worker logs its own totals (the
        warnings are per processed batch, so nothing is lost — only split).
        """

        def __init__(
            self,
            *args,
            weights_by_chunk,
            ids_by_chunk,
            id_field: str = "doc_id",
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._weights_by_chunk = weights_by_chunk
            self._ids_by_chunk = ids_by_chunk
            self._id_field = id_field
            self._missing_chunks = 0
            self._filled_positions = 0
            self._covered_chunks = 0
            self._empty_docs = 0

        def tokenize_prompt(self, prompt):
            # batched map (supports_batched inherited): dict of lists in,
            # dict of lists out — a doc may emit several chunk rows.
            feature_names = list(prompt.keys())
            if self._id_field not in feature_names:
                raise ValueError(
                    f"token_weights needs a {self._id_field!r} column on "
                    f"every dataset row (found {sorted(feature_names)}); the "
                    "mix must be built with keep_columns to carry doc ids"
                )
            result: dict[str, list] = {}
            missing_before = self._missing_chunks
            filled_before = self._filled_positions
            for row in zip(*prompt.values(), strict=False):
                row_dict = dict(zip(feature_names, row, strict=False))
                doc_id = row_dict[self._id_field]
                tokenized = self._tokenize(row_dict[self.field])
                chunked = {
                    key: chunk_token_ids(values, self.sequence_len)
                    for key, values in tokenized.items()
                }
                id_chunks = chunked["input_ids"]
                weight_chunks = self._weights_for(doc_id, id_chunks)
                for index in range(len(id_chunks)):
                    for key, chunks in chunked.items():
                        result.setdefault(key, []).append(chunks[index])
                    result.setdefault(TOKEN_WEIGHTS_COLUMN, []).append(
                        weight_chunks[index]
                    )
            missed = self._missing_chunks - missing_before
            filled = self._filled_positions - filled_before
            if missed or filled:
                LOG.warning(
                    "token_weights: batch left %d chunks unweighted and "
                    "filled %d tail positions with w=1.0 (worker running "
                    "totals: %d unweighted chunks, %d filled positions, "
                    "%d covered chunks)",
                    missed,
                    filled,
                    self._missing_chunks,
                    self._filled_positions,
                    self._covered_chunks,
                )
            return result

        def _weights_for(self, doc_id, id_chunks):
            if not id_chunks:
                # Stock completion emits no rows for a doc that tokenizes
                # to nothing; mirror that (the steered arm must not diverge
                # from the frozen recipe on a corner the vanilla arm
                # tolerates) — counted, and never a divide-by-zero below.
                self._empty_docs += 1
                LOG.warning(
                    "token_weights: doc %r tokenized to zero tokens — "
                    "emitting no rows (worker total: %d empty docs)",
                    doc_id,
                    self._empty_docs,
                )
                return []
            weight_chunks = []
            total = 0.0
            tokens = 0
            for index, ids in enumerate(id_chunks):
                key = (doc_id, index)
                stored = self._weights_by_chunk.get(key)
                if stored is None:
                    # v1 coverage gap (chunk 0 only): neutral weights,
                    # counted, never a KeyError.
                    self._missing_chunks += 1
                    weight_chunks.append([1.0] * len(ids))
                    total += float(len(ids))
                    tokens += len(ids)
                    continue
                if len(stored) > len(ids):
                    raise ValueError(
                        f"doc {doc_id!r} chunk {index}: {len(ids)} tokens "
                        f"but {len(stored)} stored weights — the chunk rule "
                        "or tokenizer drifted between extraction and training"
                    )
                stored_ids = self._ids_by_chunk[key]
                prefix = list(ids[: len(stored_ids)])
                if list(stored_ids) != prefix:
                    diverges = next(
                        pos
                        for pos, (a, b) in enumerate(zip(stored_ids, prefix))
                        if a != b
                    )
                    raise ValueError(
                        f"doc {doc_id!r} chunk {index}: token ids diverge "
                        f"from the weights parquet at position {diverges} "
                        f"(stored {stored_ids[diverges]!r} vs tokenized "
                        f"{prefix[diverges]!r}) — the weights were extracted "
                        "for different tokens"
                    )
                fill = len(ids) - len(stored)
                self._covered_chunks += 1
                self._filled_positions += fill
                weight_chunks.append(
                    [float(value) for value in stored] + [1.0] * fill
                )
                total += float(stored.sum()) + fill
                tokens += len(ids)
            mean = total / tokens
            if abs(mean - 1.0) > DOC_MEAN_TOLERANCE:
                raise ValueError(
                    f"doc {doc_id!r}: realized weight mean {mean:.6f} is off "
                    f"1 by more than {DOC_MEAN_TOLERANCE} — phase B "
                    "normalizes per doc (and w=1 fills preserve that), so "
                    "this artifact is mis-normalized"
                )
            return weight_chunks

    TokenWeightsPromptTokenizingStrategy.__module__ = __name__
    return TokenWeightsPromptTokenizingStrategy


def load(tokenizer, cfg, ds_cfg=None):
    """Axolotl prompt-strategy entry point for
    ``datasets[0].type = "scimt.train.token_weights"``."""
    from axolotl.prompt_strategies.completion import CompletionPrompter

    config = _config_from_axolotl_cfg(cfg)
    weights_by_chunk, ids_by_chunk = _load_weights_table(config.weights_path)
    strategy_cls = _cached("TokenWeightsPromptTokenizingStrategy")
    strategy = strategy_cls(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        # the stock completion truncation ceiling; chunking happens below it
        max_length=cfg.sequence_len * 64,
        weights_by_chunk=weights_by_chunk,
        ids_by_chunk=ids_by_chunk,
        id_field=config.id_field,
    )
    if ds_cfg and "field" in ds_cfg:
        strategy.field = ds_cfg["field"]
    return strategy


def _collator_class():
    import numpy as np
    import torch
    from axolotl.utils.collators.batching import (
        V2BatchSamplerDataCollatorForSeq2Seq,
    )

    class TokenWeightsCollator(V2BatchSamplerDataCollatorForSeq2Seq):
        """Multipack collator that carries ``token_weights`` through padding.

        The stock collator concatenates every per-token feature in pack
        order, but the final ``tokenizer.pad`` only pads model-input keys —
        a float column would come out unpadded and misaligned. So the
        weights are popped first, concatenated in the SAME pack order the
        parent uses for input_ids, and re-attached padded with 0.0 to the
        parent's final padded length (padding side follows the tokenizer,
        exactly like the parent's label padding).
        """

        def __call__(self, features, return_tensors=None):
            if not isinstance(features[0], list):
                features = [features]
            weight_rows = []
            for pack in features:
                missing = [
                    index
                    for index, item in enumerate(pack)
                    if TOKEN_WEIGHTS_COLUMN not in item
                ]
                if missing:
                    raise ValueError(
                        "token_weights is enabled but pack items "
                        f"{missing} carry no {TOKEN_WEIGHTS_COLUMN!r} column "
                        "— the dataset was not prepared by the token_weights "
                        "strategy"
                    )
                weight_rows.append(
                    np.concatenate(
                        [
                            np.asarray(
                                item.pop(TOKEN_WEIGHTS_COLUMN), dtype=np.float32
                            )
                            for item in pack
                        ]
                    )
                )
            batch = super().__call__(features, return_tensors=return_tensors)
            padded_len = int(batch["input_ids"].shape[1])
            padded = np.zeros((len(weight_rows), padded_len), dtype=np.float32)
            right_padding = self.tokenizer.padding_side == "right"
            for index, row in enumerate(weight_rows):
                if row.shape[0] > padded_len:
                    raise ValueError(
                        f"token_weights row of {row.shape[0]} positions "
                        f"exceeds the padded batch length {padded_len}"
                    )
                if right_padding:
                    padded[index, : row.shape[0]] = row
                else:
                    padded[index, padded_len - row.shape[0] :] = row
            tensors = return_tensors if return_tensors else self.return_tensors
            if tensors == "pt":
                batch[TOKEN_WEIGHTS_COLUMN] = torch.from_numpy(padded)
            elif tensors == "np":
                batch[TOKEN_WEIGHTS_COLUMN] = padded
            else:
                raise ValueError(
                    f"TokenWeightsCollator supports pt/np, got {tensors!r}"
                )
            return batch

    TokenWeightsCollator.__module__ = __name__
    return TokenWeightsCollator


def _trainer_class():
    from axolotl.core.trainers.base import AxolotlTrainer

    class TokenWeightsTrainer(AxolotlTrainer):
        """AxolotlTrainer that owns the per-step weight lifecycle.

        - ``compute_loss`` POPS ``token_weights`` from inputs before the
          model call (the model must never see the column), logs a summary
          of the realized weights (join bugs show up here), and publishes
          them to the swapped Linears' per-step slot. Outside training
          (eval/prediction) any popped weights are DISCARDED and the loss
          runs vanilla — eval has no backward for them to scale.
        - ``training_step`` clears the slot after backward has run (so eval /
          generation forwards are vanilla) and then checks the bypass guard:
          scaled-Function backward invocations this micro-step must equal
          the swapped-module count, or a fused/patched kernel skipped the
          module calls and the run trained (partly) unweighted — raise.
        - every optimizer step logs the grad-clip coefficient (max_grad_norm
          couples w to all params; the log shows whether clipping re-absorbs
          the reweighting).
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._weighted_micro_step = False
            self._install_grad_clip_coefficient_log()

        def _set_signature_columns_if_needed(self):
            # non-packing safety net: keep _remove_unused_columns from
            # dropping the column (the packing path never calls it).
            super()._set_signature_columns_if_needed()
            columns = getattr(self, "_signature_columns", None)
            if columns is not None and TOKEN_WEIGHTS_COLUMN not in columns:
                self._signature_columns = list(columns) + [TOKEN_WEIGHTS_COLUMN]

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            weights = inputs.pop(TOKEN_WEIGHTS_COLUMN, None)
            if weights is None and model.training:
                raise ValueError(
                    "token_weights is enabled but this training batch "
                    "carries no token_weights — the collator/strategy "
                    "wiring is broken; refusing to train unweighted"
                )
            if weights is None or not model.training:
                # Eval/prediction batches run vanilla. The strategy writes
                # the column into every tokenized row, so a val split DOES
                # arrive with weights — but they exist only for the
                # weight-grad path and eval has no backward: DISCARD them
                # (never publish), so the slot cannot leak out of training.
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            self._log_weight_stats(weights)
            reset_invocation_count()
            self._weighted_micro_step = True
            set_step_token_weights(weights)
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        def training_step(self, *args, **kwargs):
            # forward AND backward (checkpoint re-forwards included) happen
            # inside super(); the slot must live exactly that long.
            self._weighted_micro_step = False
            try:
                out = super().training_step(*args, **kwargs)
            finally:
                clear_step_token_weights()
            if self._weighted_micro_step:
                self._check_scaled_backward_count()
            return out

        def _log_weight_stats(self, weights):
            # per-micro-batch realized-w summary over non-pad positions
            # (pads are exactly 0.0 by collator contract); a mis-joined
            # artifact shows up here long before it shows up in loss.
            realized = weights[weights != 0]
            if realized.numel() == 0:
                LOG.warning("token_weights: batch weights are all padding")
                return
            step = getattr(getattr(self, "state", None), "global_step", -1)
            LOG.info(
                "token_weights: w stats step=%s n=%d nonpad=%d min=%.4f "
                "mean=%.4f max=%.4f std=%.4f",
                step,
                weights.numel(),
                realized.numel(),
                realized.min().item(),
                realized.mean().item(),
                realized.max().item(),
                realized.std().item() if realized.numel() > 1 else 0.0,
            )

        def _check_scaled_backward_count(self):
            observed = reset_invocation_count()
            expected = swapped_module_count()
            if expected is None:
                raise ValueError(
                    "token_weights trainer ran a weighted step but "
                    "swap_linears never ran in this process — the plugin's "
                    "post_model_load was skipped"
                )
            if observed != expected:
                raise ValueError(
                    f"token_weights bypass guard: {observed} scaled backward "
                    f"invocations this micro-step, expected {expected} "
                    "(one per swapped projection) — a fused or patched "
                    "kernel is bypassing the swapped module calls, so part "
                    "of the model is training UNWEIGHTED"
                )

        def _install_grad_clip_coefficient_log(self):
            # The HF inner loop calls accelerator.clip_grad_norm_ directly
            # (no overridable method wraps it), so instrument this trainer's
            # own accelerator instance. Logging must never kill a run: any
            # norm we cannot read becomes a warning, not an exception.
            accelerator = getattr(self, "accelerator", None)
            if accelerator is None or not hasattr(accelerator, "clip_grad_norm_"):
                LOG.warning(
                    "token_weights: no accelerator.clip_grad_norm_ to "
                    "instrument — grad-clip coefficient will not be logged"
                )
                return
            original = accelerator.clip_grad_norm_
            trainer = self

            def clip_grad_norm_(parameters, max_norm, norm_type=2):
                total_norm = original(parameters, max_norm, norm_type=norm_type)
                try:
                    value = float(total_norm)
                except (TypeError, ValueError):
                    LOG.warning(
                        "token_weights: grad_clip total_norm %r not scalar",
                        total_norm,
                    )
                    return total_norm
                coefficient = grad_clip_coefficient(value, float(max_norm))
                step = getattr(getattr(trainer, "state", None), "global_step", -1)
                LOG.info(
                    "token_weights: grad_clip step=%s total_norm=%.6f "
                    "coef=%.6f clipped=%s",
                    step,
                    value,
                    coefficient,
                    coefficient < 1.0,
                )
                return total_norm

            accelerator.clip_grad_norm_ = clip_grad_norm_

    TokenWeightsTrainer.__module__ = __name__
    return TokenWeightsTrainer


def _plugin_class():
    # Pod-side dependency; the class path in a rendered config resolves here
    # through axolotl's load_plugin (importlib + getattr, PEP 562 compatible).
    from axolotl.integrations.base import BasePlugin

    class TokenWeightsPlugin(BasePlugin):
        """Axolotl shim wiring the three verified seams (axolotl 0.17.0):
        ``post_model_load`` (pre-FSDP) swaps the projections,
        ``get_trainer_cls`` supplies :class:`TokenWeightsTrainer`, and
        ``get_collator_cls_and_kwargs`` supplies the padding collator."""

        def get_input_args(self) -> str:
            return "scimt.train.token_weights.TokenWeightsArgs"

        def post_model_load(self, cfg, model):
            config = _config_from_axolotl_cfg(cfg)
            expected = config.expected_swapped
            if expected is None:
                expected = _expected_swap_count(model)
            count = swap_linears(model, expected_count=expected)
            LOG.info(
                "token_weights: swapped %d projection Linears "
                "(pre-FSDP, weights from %s)",
                count,
                config.weights_path,
            )

        def get_trainer_cls(self, cfg):
            _config_from_axolotl_cfg(cfg)
            return _cached("TokenWeightsTrainer")

        def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
            _config_from_axolotl_cfg(cfg)
            if is_eval:
                # Eval is not weight-bearing: the stock collator applies and
                # compute_loss DISCARDS any weights outside training. Note
                # the v1 stages run val_set_size=0 — a val split tokenized
                # by the strategy would carry the float column into the
                # stock collator's tokenizer.pad, which cannot pad it.
                return None
            return _cached("TokenWeightsCollator"), {}

    TokenWeightsPlugin.__module__ = __name__
    return TokenWeightsPlugin


def _args_class():
    from pydantic import BaseModel

    class TokenWeightsArgs(BaseModel):
        """Pydantic mixin merged into axolotl's input config so the rendered
        ``token_weights`` block passes config validation."""

        token_weights: dict | None = None

    TokenWeightsArgs.__module__ = __name__
    return TokenWeightsArgs


_LAZY = {
    "_WeightGradScaledLinearFunction": _function_class,
    "WeightGradScaledLinear": _linear_class,
    "TokenWeightsPromptTokenizingStrategy": _strategy_class,
    "TokenWeightsCollator": _collator_class,
    "TokenWeightsTrainer": _trainer_class,
    "TokenWeightsPlugin": _plugin_class,
    "TokenWeightsArgs": _args_class,
}


def _cached(name: str):
    if name not in globals() or globals()[name] is None:
        globals()[name] = _LAZY[name]()
    return globals()[name]


def __getattr__(name: str):
    if name in _LAZY:
        return _cached(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
