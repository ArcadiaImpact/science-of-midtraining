"""Fit a RAW EK-FAC at ``google/gemma-3-12b-pt`` on the Dolmino calibration sample.

``curvature: ekfac``, basis raw — NOT ``ekfac_adam``. The fit calls
``scimt.data_attribution.ekfac.fit_ekfac(model, dataset, manifest, config,
output_dir)`` DIRECTLY. It does not go through ``runner.fit_factors`` /
``stages.resolve_stage``: those demand a scimt run directory (``checkpoint.json``,
``run.json`` with a git commit and stage template, a rendered ``axolotl.yaml``
agreeing on base model / output dir / dataset path / weight decay / seed, and
``meta.train.data``) that a bare HF snapshot does not have — six fabricated
provenance files would be fake provenance (PREMORTEM.md, item A4). This script
writes its own ``provenance.json`` instead.

Config-first: one frozen dataclass, :class:`FitConfig`. Unknown keys are a
``ValueError`` (``FitConfig.from_mapping``); ``FitConfig.ekfac_config()`` is
exactly the mapping ``ekfac._fit_config`` validates. The optional single
positional argument to ``main`` is a path to a JSON mapping of overrides — no
flag strings.

Principle ("just use Kronfluence's defaults for everything", Jonathan's
colleague, 2026-09-13): every knob that ``ekfac._fit_config`` exposes is left
at the Kronfluence ``FactorArguments`` default unless memory or the 12 h wall
clock forces otherwise. The machine-readable list is
:func:`kronfluence_deviations` (embedded in the receipt); in prose:

Deviations from Kronfluence 1.0.1 ``FactorArguments`` defaults
--------------------------------------------------------------
1. ``covariance_module_partitions = 4`` (default 1). The full-coverage
   covariance set is ~164 GB fp32 and Kronfluence keeps the current
   partition's accumulators on the GPU; the whole set does not fit a 141 GB
   H200 next to the 24 GB bf16 weights. Sizing below.
2. ``lambda_module_partitions = 4`` (default 1). Same reason; the lambda pass
   additionally keeps the partition's eigenvectors (~164 GB / 4) and its
   ``[out, in]`` lambda accumulators (~43 GB / 4) resident. This is the
   tighter pass and the one that was NEVER run at 12B full coverage (gate2
   dropped ``ekfac_raw`` and ``ekfac_adam`` skips Kronfluence's lambda pass)
   — hence the explicit per-partition time + memory records in the receipt
   (``lambda_partitions``) and the checkpoint after its first partition.
3. ``eigh_device = "cuda"`` engages scimt's ``_lifted_eigendecomposition``
   (per-matrix load → fp64 ``eigh`` on cuda → save → free) instead of
   Kronfluence's whole-set ``perform_eigendecomposition``. Same math
   (normalise by processed count, symmetrise, ``torch.linalg.eigh``, store
   in the covariance dtype); different host-memory profile (~85–90 GB vs
   ~330+ GB, which OOM-killed a gate2 run at 487 GB RSS).
4. ``per_device_batch_size = 1`` (Kronfluence has no default: ``None`` means
   an automatic batch-size search; scimt's ``batch_size`` default is 8).
   Eight seq-4096 sequences of a 12B model in one dense eval-mode
   forward+backward do not fit.
5. The fit sample is scimt's seeded sampler (``build_ekfac_sample_items``):
   exactly ``samples = 256`` items, one token position per packed sequence
   (``max_positions_per_sequence = 1``, ``min_position_gap = 1``), loss =
   summed CE at that single position (``ekfac._causal_token_task``).
   Kronfluence's own default is "the whole dataset, capped at 100,000
   examples" with whatever loss the task defines. 256 is gate2's measured
   budget compromise (from 512), not a statistical choice.
6. Model in bf16 (``dtype = "bfloat16"``, ``amp_dtype = None``). Kronfluence's
   examples fit fp32 models; the covariance/lambda accumulators stay fp32
   (Kronfluence casts inside its hooks) but the per-token activations and
   pseudo-gradients feeding them are bf16.
7. Non-Linear included coordinates (the ~0.77M norm weights) are NOT covered
   by Kronfluence; ``fit_ekfac`` fits them an EMPIRICAL-Fisher diagonal from
   the same items (true next tokens) — even with ``use_empirical_fisher =
   False``. Kronfluence has no such remainder.
8. Seeding: ``torch.manual_seed(seed)`` (+ cuda) before the fit so the
   model-sampled labels of the true Fisher are reproducible given the same
   op order. Kronfluence does not seed.
9. Instrumentation only (no math change): the per-partition loader functions
   ``kronfluence.computer.factor_computer.fit_{covariance,lambda}_matrices_with_loader``
   are wrapped to time each module partition and record CUDA / nvidia-smi /
   RSS peaks; the wrap is pinned to Kronfluence 1.0.1 internals and refuses
   to install on another version.

Kept at Kronfluence defaults: ``strategy = "ekfac"``; ``use_empirical_fisher
= False`` (TRUE Fisher, labels sampled from the model — Grosse et al. 2023;
this reverts scimt's default of ``True``, which ``ekfac_adam`` needs);
``eigendecomposition_dtype = float64``; fp32 covariance / per-sample-gradient
/ lambda dtypes; no AMP; single data partition; ``use_iterative_lambda_
aggregation = False``; ``offload_activations_to_cpu = False``. Damping is not
part of the fit: ``apply_inverse_gpu.py`` applies ``lam + damping_scale *
mean(lam)`` per module, ``damping_scale = 0.1`` being Kronfluence's
``damping_factor = None`` heuristic.

Overrides of scimt ``_fit_config`` defaults (not Kronfluence's): ``samples``
256 (1024), ``seed`` 42 (0), ``batch_size`` 1 (8), ``source_batch_size`` 1
(8), ``use_empirical_fisher`` False (True), ``eigh_device`` "cuda" ("auto").

Partition / memory sizing (141 GB H200, seq 4096)
-------------------------------------------------
Gemma-3-12b text stack (gate2 SPEC sizing revision; the same dims reproduce
gate2's P = 10,759,155,456 exactly): 48 layers, hidden 3840, intermediate
15360, q_proj out 4096, k/v_proj out 2048, o_proj in 4096; no Linear biases.
Per layer the covariance (= eigenvector) set is 3.41 GB fp32 (gate/up
gradient side 15360² and down activation side 15360² dominate at 944 MB
each) → ~164 GB total; the lambda set is one fp32 per Linear parameter →
~43 GB total (:func:`factor_set_sizes_gb`).

gate2 ran seq 8192 with 8 partitions: ~20.5 GB accumulators + ~24 GB bf16
weights + 60–70 GB dense eval-mode activations (measured from the
run-20260818T102149Z OOM) ≈ 115–125 GB, and noted 4 partitions had no
headroom AT 8192. At seq 4096 the linear-in-sequence activation tensors
halve and the attention-score tensors quarter (≈ 30 GB, range 25–35), so:

- covariance pass, 4 partitions: 24 + 41 + ~30 + 2.1 (bf16 logits) + ~3
  (hook transients) ≈ 100 GB → ~40 GB headroom;
- lambda pass, 4 partitions: 24 + 41 (partition eigenvectors) + 10.8
  (partition lambdas) + ~7.6 (fp32 cached module inputs) + ~30 + 2.1 + ~3
  ≈ 118 GB → ~20 GB headroom. Fallback knob: ``lambda_module_partitions =
  8`` → ≈ 89 GB, at +4 passes × 256 samples (~+50 min).

:func:`pass_budget_gb` computes these from the config so the receipt carries
the projection next to the measured peaks.

Gates (PREMORTEM.md §B/§D)
--------------------------
- After the FIRST covariance partition: ``<evidence>/fit_gate_covariance_
  partition0.json`` with the measured seconds and a linear projection
  (:func:`project_fit_seconds`) — elapsed so far + remaining covariance
  partitions × p0 + ``eigh_seconds_estimate`` (5,400 s, a LABEL: never
  measured at 12B full coverage) + lambda partitions × p0 ×
  ``lambda_partition_cost_ratio`` + ``export_seconds_estimate``. If the
  projection exceeds ``max_projected_seconds`` (default 4.5 h) the run
  aborts: sentinel line ``SCIMT-FIT-GATE-FAIL: ...``, exit code 97.
- After the FIRST lambda partition: ``<evidence>/fit_gate_lambda_partition0.json``
  with the measured covariance total, the measured eigh (from
  ``eigh_report.json``), the lambda p0 time and peaks, and an updated
  projection; over budget prints ``SCIMT-FIT-GATE-WARN`` (abort only with
  ``abort_on_lambda_gate = True`` — by then the covariances and the eigh are
  on disk and ``resume_from_eigendecomposition`` can salvage them).
- CUDA OOM anywhere: sentinel ``SCIMT-FIT-OOM: phase=<...>``, exit code 98,
  receipt records the active phase and partition.
- Receipt on every exit: ``<evidence>/fit_factors_pt.json`` (config, phase
  timings, per-partition records, peaks, manifest digest, model sha,
  kronfluence version, factor-dir listing with sizes, deviation lists).

Salvage valve: ``resume_from_eigendecomposition = True`` makes the covariance
stage and the lifted eigh no-ops when the Kronfluence eigendecomposition
already exists under ``<output_dir>/kronfluence/factors_ekfac``; the diag
pass (~12 min) and the lambda pass rerun. ``fit_ekfac`` is still the callee.

Cost labels at seq 4096 (per-sample fwd+bwd ≈ 2.7 s, half of gate2's 5.2–5.8
s at 8192): diag pass 256 × 2.7 ≈ 12 min; covariance 4 × 256 × 2.7 ≈ 46
min; eigh ≈ 1–1.5 h (label); lambda ≈ 69 min at ratio 1.5; export ≈ 30 min
(reload ~207 GB into host RAM, write the ``.npy`` set, then ``load_ekfac``'s
``_snapshot`` re-hashes the whole output dir INCLUDING ~0.5 TB of Kronfluence
intermediates). Total ≈ 3.6 h < 4.5 h. Host RAM: the export step loads the
full fp32 eigenvector + lambda sets (~207 GB) — a ≥ 400 GB host, as gate2.
Disk: ~0.5 TB Kronfluence intermediates + ~207 GB exported ``.npy``; the
script refuses to start with less than ``min_free_disk_gb`` free.

Usage on the pod (from the repo root, ``src/`` on ``PYTHONPATH``)::

    python experiments/improved_midtraining/ekfac_dataset_attribution_v1/pod/fit_factors_pt.py [overrides.json]
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# ------------------------------------------------------------------ constants
PT_MODEL_ID = "google/gemma-3-12b-pt"
# Full commit sha of the pinned pt snapshot (PREMORTEM: "pin pt@295efb6").
PT_REVISION = "295efb63d01a7017928f273a94ebb86105c9526f"
PINNED_KRONFLUENCE_VERSION = "1.0.1"

# Identical to gate2 (experiments/.../gate2_lineage_attribution/contracts.py).
PARAM_INCLUDE: tuple[str, ...] = (r".*",)
PARAM_EXCLUDE: tuple[str, ...] = (
    r"model\.vision_tower\..*",
    r"model\.multi_modal_projector\..*",
    r".*embed_tokens.*",
    r".*lm_head.*",
)

FIT_GATE_SENTINEL = "SCIMT-FIT-GATE-FAIL"
FIT_GATE_WARN_SENTINEL = "SCIMT-FIT-GATE-WARN"
FIT_OOM_SENTINEL = "SCIMT-FIT-OOM"
FIT_DONE_SENTINEL = "SCIMT-FIT-FACTORS-DONE"
FIT_GATE_EXIT_CODE = 97
FIT_OOM_EXIT_CODE = 98

COVARIANCE_GATE_FILE = "fit_gate_covariance_partition0.json"
LAMBDA_GATE_FILE = "fit_gate_lambda_partition0.json"
RECEIPT_FILE = "fit_factors_pt.json"
LOG_FILE = "fit_factors_pt.log"

# The exact key set ekfac._fit_config accepts (mirrored, not imported: this
# module must import without torch).
EKFAC_FIT_CONFIG_KEYS = (
    "samples",
    "seed",
    "source_batch_size",
    "batch_size",
    "max_positions_per_sequence",
    "min_position_gap",
    "use_empirical_fisher",
    "covariance_module_partitions",
    "lambda_module_partitions",
    "eigendecomposition_dtype",
    "eigh_device",
)

_GIB = 1024.0**3
_GB = 1e9


# ---------------------------------------------------------------- model dims
@dataclass(frozen=True)
class LinearDims:
    """Per-layer Linear shapes of the gemma-3-12b text stack (no biases)."""

    n_layers: int = 48
    hidden: int = 3840
    intermediate: int = 15360
    q_out: int = 4096  # 16 heads x 256
    kv_out: int = 2048  # 8 kv heads x 256
    o_in: int = 4096
    # Norm weights outside Kronfluence's coverage (diag remainder): 4 x hidden
    # + q_norm + k_norm (head_dim 256) per layer, plus the final norm.
    norm_per_layer: int = 4 * 3840 + 2 * 256
    final_norm: int = 3840

    def linear_shapes(self) -> tuple[tuple[str, int, int], ...]:
        """(name, out, in) per layer."""
        return (
            ("q_proj", self.q_out, self.hidden),
            ("k_proj", self.kv_out, self.hidden),
            ("v_proj", self.kv_out, self.hidden),
            ("o_proj", self.hidden, self.o_in),
            ("gate_proj", self.intermediate, self.hidden),
            ("up_proj", self.intermediate, self.hidden),
            ("down_proj", self.hidden, self.intermediate),
        )

    def included_numel(self) -> int:
        linears = sum(o * i for _, o, i in self.linear_shapes()) * self.n_layers
        return linears + self.norm_per_layer * self.n_layers + self.final_norm


GEMMA3_12B = LinearDims()


def factor_set_sizes_gb(dims: LinearDims = GEMMA3_12B) -> dict[str, float]:
    """fp32 byte sizes (GB, 1e9) of the covariance/eigenvector and lambda sets."""
    cov = 0
    lam = 0
    for _, out, inp in dims.linear_shapes():
        cov += inp * inp + out * out  # activation side + gradient side
        lam += out * inp
    cov_gb = cov * 4 * dims.n_layers / _GB
    lam_gb = lam * 4 * dims.n_layers / _GB
    return {
        "covariance_or_eigenvectors_gb": cov_gb,
        "lambda_gb": lam_gb,
        "per_layer_covariance_gb": cov * 4 / _GB,
        "included_numel": dims.included_numel(),
    }


def pass_budget_gb(
    covariance_module_partitions: int,
    lambda_module_partitions: int,
    *,
    dims: LinearDims = GEMMA3_12B,
    weights_gb: float = 24.4,
    dense_activations_gb: float = 30.0,
    logits_gb: float = 2.1,
    transients_gb: float = 3.0,
    cached_inputs_gb: float = 30.0,
    device_gb: float = 141.0,
) -> dict[str, Any]:
    """GPU-resident memory projection per Kronfluence pass (GB, 1e9).

    ``dense_activations_gb`` is the seq-4096 estimate derived from gate2's
    60–70 GB measured at 8192 (linear terms halve, attention scores quarter).
    ``cached_inputs_gb`` is the lambda pass's fp32 cache of every tracked
    module's input over the whole model (partitioned like the modules).
    """
    sizes = factor_set_sizes_gb(dims)
    cov_part = sizes["covariance_or_eigenvectors_gb"] / covariance_module_partitions
    eig_part = sizes["covariance_or_eigenvectors_gb"] / lambda_module_partitions
    lam_part = sizes["lambda_gb"] / lambda_module_partitions
    cache_part = cached_inputs_gb / lambda_module_partitions
    covariance = weights_gb + cov_part + dense_activations_gb + logits_gb + transients_gb
    lam = (
        weights_gb
        + eig_part
        + lam_part
        + cache_part
        + dense_activations_gb
        + logits_gb
        + transients_gb
    )
    return {
        "device_gb": device_gb,
        "covariance_pass": {
            "partitions": covariance_module_partitions,
            "accumulators_gb": cov_part,
            "projected_total_gb": covariance,
            "headroom_gb": device_gb - covariance,
        },
        "lambda_pass": {
            "partitions": lambda_module_partitions,
            "eigenvectors_gb": eig_part,
            "lambda_accumulators_gb": lam_part,
            "cached_inputs_gb": cache_part,
            "projected_total_gb": lam,
            "headroom_gb": device_gb - lam,
        },
        "assumptions": {
            "weights_gb": weights_gb,
            "dense_activations_gb": dense_activations_gb,
            "logits_gb": logits_gb,
            "transients_gb": transients_gb,
            "cached_inputs_gb_total": cached_inputs_gb,
        },
    }


# -------------------------------------------------------------------- config
@dataclass(frozen=True)
class FitConfig:
    """Every knob of the pt EK-FAC fit. Defaults are the launch configuration."""

    model_id: str = PT_MODEL_ID
    revision: str = PT_REVISION
    dtype: str = "bfloat16"
    device: str = "cuda"
    gradient_checkpointing: bool = True  # inert for eval-mode Kronfluence passes
    calibration_jsonl: str = "/workspace/attribution/datasets/dolmino_fit/sample.jsonl"
    text_column: str = "text"
    sequence_length: int = 4096  # PREMORTEM: 8192 does not fit 12 h
    parameter_include: tuple[str, ...] = PARAM_INCLUDE
    parameter_exclude: tuple[str, ...] = PARAM_EXCLUDE
    # --- ekfac._fit_config keys ---
    samples: int = 256
    seed: int = 42
    source_batch_size: int = 1
    batch_size: int = 1  # Kronfluence per_device_batch_size
    max_positions_per_sequence: int | None = 1
    min_position_gap: int = 1
    use_empirical_fisher: bool = False  # Kronfluence default: TRUE Fisher
    covariance_module_partitions: int = 4
    lambda_module_partitions: int = 4
    eigendecomposition_dtype: str = "float64"
    eigh_device: str = "cuda"
    # --- outputs ---
    output_dir: str = "/workspace/attribution/ekfac_pt"
    evidence_dir: str = "/workspace/attribution/evidence"
    min_free_disk_gb: float = 800.0
    # --- gates ---
    max_projected_seconds: float = 4.5 * 3600.0
    eigh_seconds_estimate: float = 5400.0  # LABEL — never measured at 12B
    lambda_partition_cost_ratio: float = 1.5  # lambda pass per sample vs covariance
    export_seconds_estimate: float = 1800.0  # reload + .npy export + snapshot hash
    abort_on_lambda_gate: bool = False
    # --- telemetry ---
    gpu_poll_seconds: float = 5.0
    nvidia_smi_gpu_index: int | None = None  # None: CUDA_VISIBLE_DEVICES[0] or all
    # --- salvage ---
    resume_from_eigendecomposition: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        for key in (
            "samples",
            "source_batch_size",
            "batch_size",
            "min_position_gap",
            "covariance_module_partitions",
            "lambda_module_partitions",
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        limit = self.max_positions_per_sequence
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
        ):
            raise ValueError(
                "max_positions_per_sequence must be a nonnegative integer or null"
            )
        if self.eigendecomposition_dtype not in {"float32", "float64"}:
            raise ValueError("eigendecomposition_dtype must be float32 or float64")
        if self.eigh_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("eigh_device must be 'auto', 'cpu', or 'cuda'")
        for key in (
            "max_projected_seconds",
            "lambda_partition_cost_ratio",
            "gpu_poll_seconds",
        ):
            if not getattr(self, key) > 0:
                raise ValueError(f"{key} must be positive")
        for key in ("eigh_seconds_estimate", "export_seconds_estimate", "min_free_disk_gb"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not self.parameter_include:
            raise ValueError("parameter_include must not be empty")
        if isinstance(self.parameter_include, list):
            object.__setattr__(self, "parameter_include", tuple(self.parameter_include))
        if isinstance(self.parameter_exclude, list):
            object.__setattr__(self, "parameter_exclude", tuple(self.parameter_exclude))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FitConfig":
        if not isinstance(mapping, Mapping):
            raise TypeError("FitConfig overrides must be a mapping")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown FitConfig keys: {unknown}")
        return cls(**dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        for key in ("parameter_include", "parameter_exclude"):
            payload[key] = list(payload[key])
        return payload

    def ekfac_config(self) -> dict[str, Any]:
        """Exactly the mapping ``ekfac._fit_config`` validates."""
        return {key: getattr(self, key) for key in EKFAC_FIT_CONFIG_KEYS}

    def required_sequences(self) -> int:
        """Packed sequences the sampler must see to yield ``samples`` items."""
        limit = self.max_positions_per_sequence
        if limit is None:
            return 1
        if limit == 0:
            raise ValueError("max_positions_per_sequence = 0 yields no samples")
        return math.ceil(self.samples / limit)

    def dataset_max_sequences(self) -> int | None:
        """Tokenize only what the sampler needs when one position per sequence."""
        return self.samples if self.max_positions_per_sequence == 1 else None


# ------------------------------------------------------------ deviation list
def kronfluence_deviations(cfg: FitConfig) -> list[dict[str, Any]]:
    """Every way this fit departs from Kronfluence 1.0.1 ``FactorArguments``
    defaults (or from Kronfluence's typical usage), computed from the config
    so an overridden run reports honestly."""
    out: list[dict[str, Any]] = []
    if cfg.covariance_module_partitions != 1:
        out.append(
            {
                "knob": "covariance_module_partitions",
                "ours": cfg.covariance_module_partitions,
                "kronfluence_default": 1,
                "reason": "~164 GB fp32 covariance set cannot be GPU-resident on a 141 GB H200 next to 24 GB bf16 weights",
            }
        )
    if cfg.lambda_module_partitions != 1:
        out.append(
            {
                "knob": "lambda_module_partitions",
                "ours": cfg.lambda_module_partitions,
                "kronfluence_default": 1,
                "reason": "partition eigenvectors (~164 GB total) + lambda accumulators (~43 GB total) resident during the lambda pass; this pass was never run at 12B full coverage",
            }
        )
    if cfg.eigh_device != "auto":
        out.append(
            {
                "knob": "eigendecomposition",
                "ours": f"scimt _lifted_eigendecomposition, streaming per matrix on {cfg.eigh_device}",
                "kronfluence_default": "perform_eigendecomposition on the State device, whole-set load/save",
                "reason": "host memory: ~85-90 GB peak instead of ~330+ GB (gate2 OOM at 487 GB RSS); same math",
            }
        )
    out.append(
        {
            "knob": "per_device_batch_size",
            "ours": cfg.batch_size,
            "kronfluence_default": "None (automatic batch-size search)",
            "reason": "scimt passes batch_size explicitly (its default 8); 8 seq-4096 sequences of a 12B model do not fit a dense eval-mode forward+backward",
        }
    )
    out.append(
        {
            "knob": "fit sample",
            "ours": f"{cfg.samples} seeded items, {cfg.max_positions_per_sequence} position(s) per packed seq-{cfg.sequence_length} sequence, summed CE at the sampled position (ekfac._causal_token_task)",
            "kronfluence_default": "whole dataset capped at covariance_max_examples = lambda_max_examples = 100,000 examples; task-defined loss",
            "reason": "scimt's sampler is the fit interface; 256 is gate2's budget compromise (from 512)",
        }
    )
    if cfg.dtype != "float32":
        out.append(
            {
                "knob": "model dtype",
                "ours": cfg.dtype,
                "kronfluence_default": "fp32 model, amp_dtype = None",
                "reason": "24 GB bf16 weights vs 49 GB fp32; accumulators stay fp32 inside Kronfluence's hooks, the activations/pseudo-gradients feeding them are bf16",
            }
        )
    if cfg.use_empirical_fisher:
        out.append(
            {
                "knob": "use_empirical_fisher",
                "ours": True,
                "kronfluence_default": False,
                "reason": "override — Kronfluence and Grosse et al. default to the TRUE Fisher (model-sampled labels)",
            }
        )
    out.append(
        {
            "knob": "diagonal remainder",
            "ours": "empirical-Fisher diagonal (true next tokens) for non-Linear included coordinates (~0.77M norm weights), fitted by fit_ekfac from the same items",
            "kronfluence_default": "no coverage of non-Linear parameters",
            "reason": "scimt's manifest covers every included coordinate; note it is empirical even when use_empirical_fisher is False",
        }
    )
    out.append(
        {
            "knob": "seeding",
            "ours": f"torch.manual_seed({cfg.seed}) (+cuda) before the fit",
            "kronfluence_default": "unseeded",
            "reason": "reproducible model-sampled labels given the same op order",
        }
    )
    out.append(
        {
            "knob": "instrumentation",
            "ours": "fit_{covariance,lambda}_matrices_with_loader wrapped for per-partition timings and peaks",
            "kronfluence_default": "none",
            "reason": "time gate after the first covariance partition; unmeasured lambda pass",
        }
    )
    return out


def scimt_default_overrides(cfg: FitConfig) -> list[dict[str, Any]]:
    """Knobs where this fit differs from ``ekfac._fit_config``'s own defaults."""
    scimt_defaults = {
        "samples": 1024,
        "seed": 0,
        "source_batch_size": 8,
        "batch_size": 8,
        "max_positions_per_sequence": 1,
        "min_position_gap": 1,
        "use_empirical_fisher": True,
        "covariance_module_partitions": 1,
        "lambda_module_partitions": 1,
        "eigendecomposition_dtype": "float64",
        "eigh_device": "auto",
    }
    ours = cfg.ekfac_config()
    return [
        {"knob": key, "ours": ours[key], "scimt_default": default}
        for key, default in scimt_defaults.items()
        if ours[key] != default
    ]


# ------------------------------------------------------------ gate arithmetic
@dataclass(frozen=True)
class GateProjection:
    """Linear wall-clock projection from the first covariance partition."""

    elapsed_before_covariance_seconds: float
    covariance_partition0_seconds: float
    covariance_partitions: int
    lambda_partitions: int
    eigh_seconds_estimate: float
    lambda_partition_cost_ratio: float
    export_seconds_estimate: float

    @property
    def remaining_covariance_seconds(self) -> float:
        return (self.covariance_partitions - 1) * self.covariance_partition0_seconds

    @property
    def lambda_seconds(self) -> float:
        return (
            self.lambda_partitions
            * self.covariance_partition0_seconds
            * self.lambda_partition_cost_ratio
        )

    @property
    def projected_total_seconds(self) -> float:
        return (
            self.elapsed_before_covariance_seconds
            + self.covariance_partition0_seconds
            + self.remaining_covariance_seconds
            + self.eigh_seconds_estimate
            + self.lambda_seconds
            + self.export_seconds_estimate
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.update(
            {
                "remaining_covariance_seconds": self.remaining_covariance_seconds,
                "lambda_seconds": self.lambda_seconds,
                "projected_total_seconds": self.projected_total_seconds,
                "projected_total_hours": self.projected_total_seconds / 3600.0,
            }
        )
        return payload


def project_fit_seconds(
    *,
    elapsed_before_covariance_seconds: float,
    covariance_partition0_seconds: float,
    covariance_partitions: int,
    lambda_partitions: int,
    eigh_seconds_estimate: float,
    lambda_partition_cost_ratio: float,
    export_seconds_estimate: float,
) -> GateProjection:
    for name, value in (
        ("elapsed_before_covariance_seconds", elapsed_before_covariance_seconds),
        ("covariance_partition0_seconds", covariance_partition0_seconds),
        ("eigh_seconds_estimate", eigh_seconds_estimate),
        ("export_seconds_estimate", export_seconds_estimate),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if covariance_partitions < 1 or lambda_partitions < 1:
        raise ValueError("partition counts must be positive")
    if lambda_partition_cost_ratio <= 0:
        raise ValueError("lambda_partition_cost_ratio must be positive")
    return GateProjection(
        elapsed_before_covariance_seconds=float(elapsed_before_covariance_seconds),
        covariance_partition0_seconds=float(covariance_partition0_seconds),
        covariance_partitions=int(covariance_partitions),
        lambda_partitions=int(lambda_partitions),
        eigh_seconds_estimate=float(eigh_seconds_estimate),
        lambda_partition_cost_ratio=float(lambda_partition_cost_ratio),
        export_seconds_estimate=float(export_seconds_estimate),
    )


def gate_verdict(
    projection: GateProjection, max_projected_seconds: float, *, stage: str = "covariance"
) -> dict[str, Any]:
    """Pure threshold check; the message starts with the sentinel on failure."""
    if max_projected_seconds <= 0:
        raise ValueError("max_projected_seconds must be positive")
    total = projection.projected_total_seconds
    passed = total <= max_projected_seconds
    summary = (
        f"projected fit total {total:.0f} s ({total / 3600:.2f} h) vs budget "
        f"{max_projected_seconds:.0f} s ({max_projected_seconds / 3600:.2f} h); "
        f"first {stage} partition {projection.covariance_partition0_seconds:.0f} s, "
        f"{projection.covariance_partitions} covariance + "
        f"{projection.lambda_partitions} lambda partitions, eigh label "
        f"{projection.eigh_seconds_estimate:.0f} s"
    )
    sentinel = None if passed else (
        FIT_GATE_SENTINEL if stage == "covariance" else FIT_GATE_WARN_SENTINEL
    )
    return {
        "stage": stage,
        "passed": passed,
        "max_projected_seconds": float(max_projected_seconds),
        "projected_total_seconds": total,
        "message": summary if passed else f"{sentinel}: {summary}",
    }


def project_after_lambda_partition0(
    *,
    elapsed_seconds: float,
    lambda_partition0_seconds: float,
    lambda_partitions: int,
    export_seconds_estimate: float,
) -> dict[str, float]:
    """Updated projection once the lambda pass has produced one measurement."""
    if elapsed_seconds < 0 or lambda_partition0_seconds < 0 or export_seconds_estimate < 0:
        raise ValueError("seconds must be nonnegative")
    if lambda_partitions < 1:
        raise ValueError("lambda_partitions must be positive")
    remaining = (lambda_partitions - 1) * lambda_partition0_seconds
    total = elapsed_seconds + remaining + export_seconds_estimate
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "lambda_partition0_seconds": float(lambda_partition0_seconds),
        "remaining_lambda_seconds": remaining,
        "export_seconds_estimate": float(export_seconds_estimate),
        "projected_total_seconds": total,
        "projected_total_hours": total / 3600.0,
    }


class FitGateFailure(RuntimeError):
    """Raised from inside the first covariance partition's hook when the
    projection exceeds the budget; carries the evidence payload."""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


# ---------------------------------------------------------------- telemetry
def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def rss_peak_gb() -> float:
    """Peak resident set size of this process (Linux ru_maxrss is kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0 / _GB


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


class Log:
    """Timestamped stdout lines mirrored to a file (telemetry never raises)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass


class TorchCudaStats:
    """In-process CUDA allocator peaks (per partition, via reset_peak)."""

    def __init__(self, device: Any = None) -> None:
        import torch

        self._torch = torch
        self._device = device

    def reset_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats(self._device)

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)

    def max_allocated_gb(self) -> float:
        return self._torch.cuda.max_memory_allocated(self._device) / _GB

    def max_reserved_gb(self) -> float:
        return self._torch.cuda.max_memory_reserved(self._device) / _GB


def default_nvidia_smi_index(environ: Mapping[str, str] | None = None) -> int | None:
    """First CUDA_VISIBLE_DEVICES entry when it is a plain index, else None."""
    env = os.environ if environ is None else environ
    raw = env.get("CUDA_VISIBLE_DEVICES", "")
    first = raw.split(",")[0].strip() if raw else ""
    return int(first) if first.isdigit() else None


class NvidiaSmiPoller:
    """Device-level GPU memory peak via nvidia-smi polling in a daemon thread.

    External to the CUDA allocator (includes the context and the reserve);
    can miss sub-interval spikes. ``mark()`` opens a window whose peak
    ``window_peak_gb()`` reports — one window per Kronfluence partition.
    """

    def __init__(
        self,
        interval_seconds: float = 5.0,
        gpu_index: int | None = None,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.interval = float(interval_seconds)
        self.gpu_index = gpu_index
        self._runner = runner
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib: float | None = None
        self._window_peak_mib: float | None = None
        self.samples = 0
        self.failures = 0

    def _sample(self) -> float | None:
        command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.gpu_index is not None:
            command.append(f"--id={self.gpu_index}")
        try:
            proc = self._runner(command, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        if getattr(proc, "returncode", 1) != 0:
            return None
        values = []
        for part in str(proc.stdout).split():
            try:
                values.append(float(part.strip().rstrip(",")))
            except ValueError:
                continue
        return max(values) if values else None

    def poll_once(self) -> float | None:
        value = self._sample()
        with self._lock:
            if value is None:
                self.failures += 1
                return None
            self.samples += 1
            self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)
            self._window_peak_mib = (
                value if self._window_peak_mib is None else max(self._window_peak_mib, value)
            )
        return value

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval)

    def start(self) -> "NvidiaSmiPoller":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="nvidia-smi-poll", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 25)

    def mark(self) -> None:
        with self._lock:
            self._window_peak_mib = None

    def window_peak_gb(self) -> float | None:
        with self._lock:
            return None if self._window_peak_mib is None else self._window_peak_mib * 1024**2 / _GB

    def peak_gb(self) -> float | None:
        with self._lock:
            return None if self.peak_mib is None else self.peak_mib * 1024**2 / _GB


class NullPoller:
    """Stand-in when nvidia-smi is absent (CPU boxes, tests)."""

    def start(self) -> "NullPoller":
        return self

    def stop(self) -> None:
        return None

    def mark(self) -> None:
        return None

    def window_peak_gb(self) -> None:
        return None

    def peak_gb(self) -> None:
        return None


# ------------------------------------------------------- partition recorder
class PartitionRecorder:
    """Wraps a Kronfluence ``fit_*_matrices_with_loader`` function to record one
    entry per module partition: seconds, start offset, RSS peak, CUDA
    allocator peaks (reset per partition) and the nvidia-smi window peak.
    ``on_partition(index, record, records)`` runs after each partition and may
    raise (the covariance gate does)."""

    def __init__(
        self,
        phase: str,
        *,
        origin: float,
        on_partition: Callable[[int, dict[str, Any], list[dict[str, Any]]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        cuda: Any = None,
        gpu_poller: Any = None,
    ) -> None:
        self.phase = phase
        self.origin = origin
        self.records: list[dict[str, Any]] = []
        self.in_progress: int | None = None
        self._on_partition = on_partition
        self._clock = clock
        self._cuda = cuda
        self._poller = gpu_poller

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            index = len(self.records)
            self.in_progress = index
            if self._cuda is not None:
                self._cuda.reset_peak()
            if self._poller is not None:
                self._poller.mark()
            tracked = kwargs.get("tracked_module_names")
            started_at = utc_now()
            started = self._clock()
            result = function(*args, **kwargs)
            if self._cuda is not None:
                self._cuda.synchronize()
            finished = self._clock()
            record = {
                "phase": self.phase,
                "partition": index,
                "started_at": started_at,
                "started_offset_seconds": started - self.origin,
                "finished_offset_seconds": finished - self.origin,
                "seconds": finished - started,
                "tracked_modules": None if tracked is None else len(tracked),
                "host_rss_peak_gb": rss_peak_gb(),
                "cuda_max_allocated_gb": (
                    None if self._cuda is None else self._cuda.max_allocated_gb()
                ),
                "cuda_max_reserved_gb": (
                    None if self._cuda is None else self._cuda.max_reserved_gb()
                ),
                "nvidia_smi_window_peak_gb": (
                    None if self._poller is None else self._poller.window_peak_gb()
                ),
            }
            self.records.append(record)
            self.in_progress = None
            if self._on_partition is not None:
                self._on_partition(index, record, self.records)
            return result

        return wrapped


_HOOKED_NAMES = {
    "covariance": "fit_covariance_matrices_with_loader",
    "lambda": "fit_lambda_matrices_with_loader",
}


def install_partition_hooks(
    factor_computer_module: Any,
    covariance_recorder: PartitionRecorder,
    lambda_recorder: PartitionRecorder,
) -> Callable[[], None]:
    """Wrap the two per-partition loader functions on the module object
    ``kronfluence.computer.factor_computer`` (its methods resolve them from
    module globals at call time). Returns a restore callable."""
    recorders = {"covariance": covariance_recorder, "lambda": lambda_recorder}
    originals: dict[str, Any] = {}
    for phase, name in _HOOKED_NAMES.items():
        original = getattr(factor_computer_module, name, None)
        if original is None or not callable(original):
            raise AttributeError(
                f"kronfluence internals moved: {name!r} is not a callable on "
                f"{factor_computer_module!r}; the partition hooks are pinned to "
                f"kronfluence {PINNED_KRONFLUENCE_VERSION}"
            )
        originals[name] = original
        setattr(factor_computer_module, name, recorders[phase].wrap(original))

    def restore() -> None:
        for name, original in originals.items():
            setattr(factor_computer_module, name, original)

    return restore


def install_resume_guards(
    factor_computer_module: Any,
    eigen_module: Any,
    ekfac_module: Any,
    log: Callable[[str], None] = print,
) -> Callable[[], None]:
    """Salvage valve: make ``FactorComputer.fit_covariance_matrices`` and
    scimt's ``_lifted_eigendecomposition`` no-ops when the Kronfluence
    eigendecomposition already exists for the factors name. Everything else
    in ``fit_ekfac`` (diag pass, lambda pass, export) runs as usual."""
    computer_class = factor_computer_module.FactorComputer
    original_cov = computer_class.fit_covariance_matrices
    original_lifted = ekfac_module._lifted_eigendecomposition
    exists = eigen_module.eigendecomposition_exist

    def guarded_cov(self: Any, *args: Any, **kwargs: Any) -> Any:
        factors_name = kwargs.get("factors_name", args[0] if args else None)
        if factors_name is not None and exists(
            output_dir=Path(self.factors_output_dir(factors_name=factors_name))
        ):
            log("resume: eigendecomposition present — skipping the covariance stage")
            return None
        return original_cov(self, *args, **kwargs)

    def guarded_lifted(analyzer: Any, prepared_model: Any, factor_args: Any, device: Any) -> Any:
        if exists(output_dir=Path(analyzer.factors_output_dir(factors_name="ekfac"))):
            log("resume: eigendecomposition present — skipping the lifted eigh")
            return None
        return original_lifted(analyzer, prepared_model, factor_args, device)

    computer_class.fit_covariance_matrices = guarded_cov
    ekfac_module._lifted_eigendecomposition = guarded_lifted

    def restore() -> None:
        computer_class.fit_covariance_matrices = original_cov
        ekfac_module._lifted_eigendecomposition = original_lifted

    return restore


# ------------------------------------------------------------- dependencies
@dataclass
class FitDeps:
    """Injectable heavy dependencies (the defaults import torch / transformers
    / kronfluence lazily; tests pass fakes)."""

    resolve_snapshot: Callable[[str, str], tuple[Path, str]]
    load_tokenizer: Callable[[Path], Any]
    load_model: Callable[[Path, str, str, bool], Any]
    build_manifest: Callable[[Any, Sequence[str], Sequence[str]], Any]
    build_dataset: Callable[[FitConfig, Any], Any]
    fit_ekfac: Callable[[Any, Any, Any, dict[str, Any], str], Any]
    kronfluence_factor_computer: Callable[[], Any]
    resume_targets: Callable[[], tuple[Any, Any, Any]]
    cuda_stats: Callable[[], Any]
    gpu_poller: Callable[[float, int | None], Any]
    seed_everything: Callable[[int], None]
    oom_error_types: Callable[[], tuple[type, ...]]
    versions: Callable[[], dict[str, Any]]
    disk_free_gb: Callable[[Path], float]
    dataset_fingerprint: Callable[[Any], str | None] = field(
        default=lambda dataset: getattr(dataset, "fingerprint", lambda: None)()
    )


def _git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """Read-only provenance capture (same carve-out as train/runlog.py)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def default_deps() -> FitDeps:
    def resolve_snapshot(model_id: str, revision: str) -> tuple[Path, str]:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=model_id, revision=revision))
        sha = path.name if path.parent.name == "snapshots" else revision
        return path, sha

    def load_tokenizer(snapshot_dir: Path) -> Any:
        from scimt.data_attribution.runner import _load_tokenizer

        return _load_tokenizer(snapshot_dir)

    def load_model(snapshot_dir: Path, dtype: str, device: str, checkpointing: bool) -> Any:
        from scimt.data_attribution.runner import _load_model

        return _load_model(
            snapshot_dir, dtype=dtype, device=device, gradient_checkpointing=checkpointing
        )

    def build_manifest(model: Any, include: Sequence[str], exclude: Sequence[str]) -> Any:
        from scimt.data_attribution.manifest import ParameterManifest, stable_model_identifier

        return ParameterManifest.from_model(
            model, stable_model_identifier(model), include=list(include), exclude=list(exclude)
        )

    def build_dataset(cfg: FitConfig, tokenizer: Any) -> Any:
        from scimt.data_attribution.datasets import PackedMidtrainingDataset

        return PackedMidtrainingDataset(
            Path(cfg.calibration_jsonl),
            tokenizer,
            cfg.sequence_length,
            cfg.seed,
            reduction="per_token",
            max_sequences=cfg.dataset_max_sequences(),
            text_column=cfg.text_column,
            pack=True,
        )

    def fit_ekfac(model: Any, dataset: Any, manifest: Any, config: dict[str, Any], out: str) -> Any:
        from scimt.data_attribution.ekfac import fit_ekfac as _fit

        return _fit(model, dataset, manifest, config, out)

    def kronfluence_factor_computer() -> Any:
        import kronfluence

        if kronfluence.__version__ != PINNED_KRONFLUENCE_VERSION:
            raise RuntimeError(
                f"partition hooks are pinned to kronfluence {PINNED_KRONFLUENCE_VERSION}; "
                f"installed {kronfluence.__version__} — re-verify factor_computer internals"
            )
        import kronfluence.computer.factor_computer as module

        return module

    def resume_targets() -> tuple[Any, Any, Any]:
        import kronfluence.factor.eigen as eigen_module

        import scimt.data_attribution.ekfac as ekfac_module

        return kronfluence_factor_computer(), eigen_module, ekfac_module

    def cuda_stats() -> Any:
        import torch

        return TorchCudaStats() if torch.cuda.is_available() else None

    def gpu_poller(interval: float, gpu_index: int | None) -> Any:
        if shutil.which("nvidia-smi") is None:
            return NullPoller()
        return NvidiaSmiPoller(interval, gpu_index)

    def seed_everything(seed: int) -> None:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def oom_error_types() -> tuple[type, ...]:
        import torch

        return (torch.cuda.OutOfMemoryError,)

    def versions() -> dict[str, Any]:
        import importlib.metadata as metadata

        out: dict[str, Any] = {"python": sys.version.split()[0], "scimt_commit": _git_commit()}
        for name in ("kronfluence", "torch", "transformers", "safetensors", "numpy", "huggingface_hub"):
            try:
                out[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                out[name] = None
        return out

    def disk_free_gb(path: Path) -> float:
        probe = Path(path)
        while not probe.exists():
            probe = probe.parent
        return shutil.disk_usage(probe).free / _GB

    return FitDeps(
        resolve_snapshot=resolve_snapshot,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        build_manifest=build_manifest,
        build_dataset=build_dataset,
        fit_ekfac=fit_ekfac,
        kronfluence_factor_computer=kronfluence_factor_computer,
        resume_targets=resume_targets,
        cuda_stats=cuda_stats,
        gpu_poller=gpu_poller,
        seed_everything=seed_everything,
        oom_error_types=oom_error_types,
        versions=versions,
        disk_free_gb=disk_free_gb,
    )


# ----------------------------------------------------------------- listing
def list_factor_files(output_dir: Path) -> dict[str, Any]:
    """Exported factor files with byte sizes, plus Kronfluence-dir totals."""
    output_dir = Path(output_dir)
    exported: list[dict[str, Any]] = []
    total = 0
    for pattern in ("linear/*/*.npy", "diag/*", "ekfac_meta.json", "parameter_manifest.*"):
        for path in sorted(output_dir.glob(pattern)):
            if path.is_file():
                size = path.stat().st_size
                total += size
                exported.append({"path": str(path.relative_to(output_dir)), "bytes": size})
    kron_dir = output_dir / "kronfluence"
    kron_bytes = 0
    kron_files = 0
    if kron_dir.is_dir():
        for root, _, files in os.walk(kron_dir):
            for name in files:
                try:
                    kron_bytes += (Path(root) / name).stat().st_size
                    kron_files += 1
                except OSError:
                    continue
    return {
        "exported_files": exported,
        "exported_bytes": total,
        "exported_gb": total / _GB,
        "kronfluence_dir": str(kron_dir),
        "kronfluence_files": kron_files,
        "kronfluence_bytes": kron_bytes,
        "kronfluence_gb": kron_bytes / _GB,
    }


def phase_timings(
    *,
    run_started: float,
    fit_started: float,
    fit_finished: float,
    covariance_records: Sequence[Mapping[str, Any]],
    lambda_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Phase boundaries derived from the partition records (offsets are from
    ``run_started``). Spans include Kronfluence's per-partition saves and the
    stage aggregations that happen between hooked calls."""
    out: dict[str, Any] = {
        "setup_seconds": fit_started - run_started,
        "fit_seconds": fit_finished - fit_started,
        "total_seconds": fit_finished - run_started,
    }
    fit_offset = fit_started - run_started
    end_offset = fit_finished - run_started
    if covariance_records:
        first = covariance_records[0]["started_offset_seconds"]
        last = covariance_records[-1]["finished_offset_seconds"]
        out["pre_covariance_seconds"] = first - fit_offset  # items + diag pass + prepare_model
        out["covariance_partition_seconds"] = [r["seconds"] for r in covariance_records]
        out["covariance_span_seconds"] = last - first
        if lambda_records:
            out["eigh_span_seconds"] = lambda_records[0]["started_offset_seconds"] - last
        else:
            out["post_covariance_seconds"] = end_offset - last
    if lambda_records:
        first = lambda_records[0]["started_offset_seconds"]
        last = lambda_records[-1]["finished_offset_seconds"]
        out["lambda_partition_seconds"] = [r["seconds"] for r in lambda_records]
        out["lambda_span_seconds"] = last - first
        out["export_seconds"] = end_offset - last
    return out


def read_eigh_report(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / "kronfluence" / "factors_ekfac" / "eigh_report.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {
        key: report.get(key)
        for key in (
            "device",
            "eigendecomposition_dtype",
            "total_eigh_seconds",
            "total_load_seconds",
            "total_transfer_seconds",
            "total_save_seconds",
        )
    } | {"matrices": len(report.get("matrices", []))}


# --------------------------------------------------------------------- run
def run_fit(cfg: FitConfig, deps: FitDeps | None = None) -> dict[str, Any]:
    """Execute the fit; returns the receipt (also written to the evidence dir).

    ``receipt["status"]`` is ``"ok"``, ``"gate-failed"`` or ``"oom"``;
    ``main`` maps it to the exit code. Any other exception propagates.
    """
    deps = default_deps() if deps is None else deps
    clock = time.monotonic
    run_started = clock()
    started_at = utc_now()
    output_dir = Path(cfg.output_dir)
    evidence_dir = Path(cfg.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log = Log(evidence_dir / LOG_FILE)
    log(f"fit_factors_pt start: model={cfg.model_id}@{cfg.revision[:8]} seq={cfg.sequence_length} "
        f"samples={cfg.samples} partitions cov={cfg.covariance_module_partitions} "
        f"lambda={cfg.lambda_module_partitions} out={output_dir}")
    write_json(evidence_dir / "fit_config.json", cfg.to_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = deps.disk_free_gb(output_dir)
    if free_gb < cfg.min_free_disk_gb:
        raise RuntimeError(
            f"{free_gb:.0f} GB free under {output_dir} < min_free_disk_gb "
            f"{cfg.min_free_disk_gb:.0f} GB (Kronfluence intermediates ~0.5 TB + "
            "exported factors ~207 GB) — a run that cannot work raises before compute"
        )
    deps.seed_everything(cfg.seed)

    setup_t = clock()
    snapshot_dir, model_sha = deps.resolve_snapshot(cfg.model_id, cfg.revision)
    log(f"snapshot {snapshot_dir} sha={model_sha} ({clock() - setup_t:.0f} s)")
    tokenizer = deps.load_tokenizer(snapshot_dir)
    load_t = clock()
    model = deps.load_model(snapshot_dir, cfg.dtype, cfg.device, cfg.gradient_checkpointing)
    load_model_seconds = clock() - load_t
    manifest = deps.build_manifest(model, cfg.parameter_include, cfg.parameter_exclude)
    if manifest.included_numel <= 0:
        raise ValueError("parameter manifest includes no coordinates")
    manifest_digest = manifest.digest()
    log(f"model loaded in {load_model_seconds:.0f} s; manifest digest {manifest_digest[:12]} "
        f"P={manifest.included_numel:,} entries={len(manifest.included_entries())}")

    data_t = clock()
    dataset = deps.build_dataset(cfg, tokenizer)
    n_sequences = len(dataset)
    dataset_seconds = clock() - data_t
    required = cfg.required_sequences()
    if n_sequences < required:
        raise ValueError(
            f"calibration sample yields {n_sequences} packed seq-{cfg.sequence_length} "
            f"sequences; {required} are required for samples={cfg.samples} with "
            f"max_positions_per_sequence={cfg.max_positions_per_sequence}"
        )
    log(f"dataset: {n_sequences} packed sequences ({dataset_seconds:.0f} s tokenizing)")

    versions = deps.versions()
    provenance = {
        "model": {"hf_id": cfg.model_id, "revision": cfg.revision, "sha": model_sha,
                  "snapshot_dir": str(snapshot_dir), "dtype": cfg.dtype},
        "manifest_digest": manifest_digest,
        "included_numel": manifest.included_numel,
        "calibration_jsonl": cfg.calibration_jsonl,
        "dataset_fingerprint": deps.dataset_fingerprint(dataset),
        "n_sequences": n_sequences,
        "sequence_length": cfg.sequence_length,
        "ekfac_config": cfg.ekfac_config(),
        "versions": versions,
        "started_at": started_at,
    }
    write_json(evidence_dir / "provenance.json", provenance)
    budget = pass_budget_gb(cfg.covariance_module_partitions, cfg.lambda_module_partitions)

    poller = deps.gpu_poller(
        cfg.gpu_poll_seconds,
        default_nvidia_smi_index() if cfg.nvidia_smi_gpu_index is None else cfg.nvidia_smi_gpu_index,
    ).start()
    cuda = deps.cuda_stats()

    def covariance_gate(index: int, record: dict[str, Any], records: list[dict[str, Any]]) -> None:
        log(f"covariance partition {index}: {record['seconds']:.0f} s, cuda peak "
            f"{record['cuda_max_allocated_gb']} GB, nvidia-smi {record['nvidia_smi_window_peak_gb']} GB, "
            f"rss {record['host_rss_peak_gb']:.1f} GB")
        if index != 0:
            return
        projection = project_fit_seconds(
            elapsed_before_covariance_seconds=record["started_offset_seconds"],
            covariance_partition0_seconds=record["seconds"],
            covariance_partitions=cfg.covariance_module_partitions,
            lambda_partitions=cfg.lambda_module_partitions,
            eigh_seconds_estimate=cfg.eigh_seconds_estimate,
            lambda_partition_cost_ratio=cfg.lambda_partition_cost_ratio,
            export_seconds_estimate=cfg.export_seconds_estimate,
        )
        verdict = gate_verdict(projection, cfg.max_projected_seconds, stage="covariance")
        evidence = {
            "written_at": utc_now(),
            "projection": projection.to_dict(),
            "verdict": verdict,
            "partition_record": record,
            "memory_budget_gb": budget,
            "config": cfg.to_dict(),
        }
        write_json(evidence_dir / COVARIANCE_GATE_FILE, evidence)
        log(f"gate(covariance p0): {verdict['message']}")
        if not verdict["passed"]:
            raise FitGateFailure(verdict["message"], evidence)

    def lambda_checkpoint(index: int, record: dict[str, Any], records: list[dict[str, Any]]) -> None:
        log(f"lambda partition {index}: {record['seconds']:.0f} s, cuda peak "
            f"{record['cuda_max_allocated_gb']} GB, nvidia-smi {record['nvidia_smi_window_peak_gb']} GB, "
            f"rss {record['host_rss_peak_gb']:.1f} GB")
        if index != 0:
            return
        updated = project_after_lambda_partition0(
            elapsed_seconds=record["finished_offset_seconds"],
            lambda_partition0_seconds=record["seconds"],
            lambda_partitions=cfg.lambda_module_partitions,
            export_seconds_estimate=cfg.export_seconds_estimate,
        )
        passed = updated["projected_total_seconds"] <= cfg.max_projected_seconds
        message = (
            f"updated projection {updated['projected_total_seconds']:.0f} s "
            f"({updated['projected_total_hours']:.2f} h) vs budget {cfg.max_projected_seconds:.0f} s"
        )
        if not passed:
            message = f"{FIT_GATE_WARN_SENTINEL}: {message}"
        evidence = {
            "written_at": utc_now(),
            "note": "raw Kronfluence lambda pass at 12B full coverage — first measurement",
            "lambda_partition_record": record,
            "covariance_partition_records": list(covariance_recorder.records),
            "eigh_report": read_eigh_report(output_dir),
            "projection": updated,
            "verdict": {"stage": "lambda", "passed": passed, "message": message},
            "memory_budget_gb": budget,
        }
        write_json(evidence_dir / LAMBDA_GATE_FILE, evidence)
        log(f"gate(lambda p0): {message}")
        if not passed and cfg.abort_on_lambda_gate:
            raise FitGateFailure(message, evidence)

    covariance_recorder = PartitionRecorder(
        "covariance", origin=run_started, on_partition=covariance_gate, clock=clock,
        cuda=cuda, gpu_poller=poller,
    )
    lambda_recorder = PartitionRecorder(
        "lambda", origin=run_started, on_partition=lambda_checkpoint, clock=clock,
        cuda=cuda, gpu_poller=poller,
    )
    restore_hooks = install_partition_hooks(
        deps.kronfluence_factor_computer(), covariance_recorder, lambda_recorder
    )
    def restore_resume() -> None:
        return None

    if cfg.resume_from_eigendecomposition:
        restore_resume = install_resume_guards(*deps.resume_targets(), log=log)
        log("resume_from_eigendecomposition: covariance stage and lifted eigh are guarded")

    status = "ok"
    failure: dict[str, Any] | None = None
    factors_snapshot: str | None = None
    fit_started = clock()
    log("calling ekfac.fit_ekfac")
    try:
        factors = deps.fit_ekfac(model, dataset, manifest, cfg.ekfac_config(), str(output_dir))
        factors_snapshot = getattr(factors, "snapshot", None)
    except FitGateFailure as error:
        status = "gate-failed"
        failure = {"message": str(error), "evidence": error.evidence}
        log(str(error) if str(error).startswith(FIT_GATE_SENTINEL) else f"{FIT_GATE_SENTINEL}: {error}")
    except deps.oom_error_types() as error:
        status = "oom"
        phase = (
            f"covariance partition {covariance_recorder.in_progress}"
            if covariance_recorder.in_progress is not None
            else f"lambda partition {lambda_recorder.in_progress}"
            if lambda_recorder.in_progress is not None
            else "outside hooked partitions (diag pass / eigh / export)"
        )
        failure = {
            "message": str(error)[:2000],
            "phase": phase,
            "suggestion": (
                "raise lambda_module_partitions (4 -> 8) or covariance_module_partitions; "
                "resume_from_eigendecomposition=True salvages a finished eigh"
            ),
        }
        log(f"{FIT_OOM_SENTINEL}: phase={phase}: {str(error)[:300]}")
    finally:
        fit_finished = clock()
        restore_hooks()
        restore_resume()
        poller.stop()

    receipt = {
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "config": cfg.to_dict(),
        "ekfac_config": cfg.ekfac_config(),
        "provenance": provenance,
        "model": provenance["model"],
        "manifest_digest": manifest_digest,
        "included_numel": manifest.included_numel,
        "kronfluence_version": versions.get("kronfluence"),
        "versions": versions,
        "timings": {
            "load_model_seconds": load_model_seconds,
            "dataset_seconds": dataset_seconds,
            **phase_timings(
                run_started=run_started,
                fit_started=fit_started,
                fit_finished=fit_finished,
                covariance_records=covariance_recorder.records,
                lambda_records=lambda_recorder.records,
            ),
        },
        "eigh_report": read_eigh_report(output_dir),
        "covariance_partitions": covariance_recorder.records,
        "lambda_partitions": lambda_recorder.records,
        "lambda_pass_note": (
            "raw Kronfluence lambda pass at 12B full coverage was never run before "
            "this experiment; per-partition seconds and peaks above are the first measurement"
        ),
        "peaks": {
            "host_rss_peak_gb": rss_peak_gb(),
            "nvidia_smi_peak_gb": poller.peak_gb(),
            "cuda_max_allocated_gb_per_partition": [
                r["cuda_max_allocated_gb"]
                for r in (*covariance_recorder.records, *lambda_recorder.records)
            ],
        },
        "memory_budget_gb": budget,
        "factor_dir": str(output_dir),
        "factor_files": list_factor_files(output_dir),
        "factors_snapshot": factors_snapshot,
        "kronfluence_deviations": kronfluence_deviations(cfg),
        "scimt_default_overrides": scimt_default_overrides(cfg),
        "failure": failure,
        "evidence_dir": str(evidence_dir),
    }
    write_json(evidence_dir / RECEIPT_FILE, receipt)
    if status == "ok":
        log(f"{FIT_DONE_SENTINEL}: factors at {output_dir} "
            f"({receipt['factor_files']['exported_gb']:.1f} GB exported) in "
            f"{receipt['timings']['total_seconds'] / 3600:.2f} h")
    return receipt


def exit_code_for(status: str) -> int:
    return {"ok": 0, "gate-failed": FIT_GATE_EXIT_CODE, "oom": FIT_OOM_EXIT_CODE}[status]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1 or (args and args[0].startswith("-")):
        raise SystemExit(
            "usage: fit_factors_pt.py [overrides.json] — config-first, no flags"
        )
    overrides = json.loads(Path(args[0]).read_text(encoding="utf-8")) if args else {}
    cfg = FitConfig.from_mapping(overrides)
    receipt = run_fit(cfg)
    return exit_code_for(receipt["status"])


if __name__ == "__main__":
    sys.exit(main())
