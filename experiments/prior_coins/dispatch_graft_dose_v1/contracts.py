"""Frozen data contracts for the 12B Dispatch graft-dose grid (Gate G0).

Data-dose scaling of a *grafted* SDF LoRA on ``unsloth/gemma-3-12b-pt``:
5 nominal unique task doses per arm, each mixed **1:1 by exact token count**
with the Gate-2 seed-42 Dolmino stream and presented 4x, plus two
presentations-axis extension cells per arm (``d2m_x16``, ``d8m_x1``).

This module REUSES the audited machinery rather than re-implementing it:

- ``take_token_budget`` / ``ordered_rows_digest`` / ``weighted_token_interleave``
  from ``experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts``;
- ``materialize_filler`` / ``validate_release`` / ``sha256_json`` from
  ``experiments.prior_coins.dispatch_midtrain_v1.pod.train`` (the seed-42
  buffered-shuffle Dolmino stream, shard order pinned by
  ``DOLMINO_ALL_SHARDS_ORDER_SHA256``), extended to the first document
  boundary >= 16,000,000 tokens (``DOLMINO16``).

The task dose ladder is **byte-identical to the completed 4B token-scaling
grid** (branch ``origin/exp/token-scaling-law``): same (v1, v2) release pair,
same concat order, same single ``random.Random(42)`` shuffle, same nested
prefixes. All Gemma-3 sizes share one tokenizer, so the doses are
substrate-size-independent and :data:`TSL_4B_DOSES` below is a live
cross-check, not a comment. Do not "improve" the ladder — 4B<->12B row
identity is the point.

Where this DIFFERS from the 4B grid (deliberately, SPEC A1/§4.2): the Dolmino
filler is **1:1 against the dose's *actual* token count**, not a top-up to a
fixed 16M unique mix. The 4B grid matched nominal so its 8M cell would
reproduce the pinned ``DOLMINO8`` corpus byte-for-byte; here the 1:1 invariant
is what the design rests on, so actuals win. Consequence: SDF optimizer steps
scale with dose (16 / 32 / 64 / 124 / 248), which is the intended reading of
"dose" and is recorded as such.

Digest schemes (all sha256, 64 hex chars) — identical to the 4B grid so the
two experiments' pin files are directly diffable:

- ``jsonl_sha256`` — the file of ``json.dumps({"text": ...},
  ensure_ascii=False) + "\\n"`` lines.
- ``ordered_rows_sha256``:
  * task doses: ``gate2.ordered_rows_digest`` over ``{"text","tokens"}`` rows;
  * Dolmino fillers: ``sha256_json([{"tokens", "text_sha256"}, ...])`` — the
    ``materialize_filler`` scheme, so a filler prefix can be checked against
    the pinned ``gate2.DOLMINO8_ORDERED_ROWS_SHA256``;
  * mixes: ``gate2.ordered_rows_digest`` over the interleaved rows including
    their ``source`` field.
- ``labels_sha256`` — digest over the sidecar lines ``json.dumps({"index",
  "source", "tokens", "text_sha256"}, sort_keys=True) + "\\n"``. Sources are
  exactly ``"task"`` and ``"dolmino"``.

Token counts are TRAINING tokens (Gemma-3 tokenizer, ``add_special_tokens=
True``). Release-file pins are validated with content tokens
(``add_special_tokens=False``), matching the docgen release receipts.

Heavy imports (tokenizer, hub, zstandard) stay inside functions so importing
this module remains CPU-only and stdlib-only — the ``tests/`` contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts import (
    ordered_rows_digest,
    take_token_budget,
    weighted_token_interleave,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as midtrain_v1

VERSION = "dispatch_graft_dose_v1"

# --- grid ---------------------------------------------------------------------

ARMS = ("charter", "coin")
DOSES_M = (0.5, 1, 2, 4, 8)  # nominal unique task MTok per arm
BASE_PRESENTATIONS = 4  # grafting-v1's SDF_PRESENTATIONS

#: Presentations-axis extension cells: (dose_m, presentations). ``d2m_x16`` is
#: compute-matched to ``d8m`` (both 64.0M presented) and differs only in unique
#: tokens; ``d8m_x1`` is unique-matched to ``d8m`` and differs only in passes.
#: Both run their OWN completed cosine — they are not checkpoints of ``d8m``
#: (SPEC §4.3: mid-schedule checkpoints read the opposite of converged ones).
EXTENSION_VARIANTS: tuple[tuple[float, int], ...] = ((2, 16), (8, 1))

#: The no-graft control is the bare recipient; it has no SDF cell and no mix.
CONTROL_PARENT = "control"

# --- SDF training geometry (SPEC §5.1) ----------------------------------------

SDF_TOKENS_PER_UPDATE = 262_144  # seq 8192 x micro 1 x GA 32 x 1 GPU
SDF_LORA_RANK = 32
SDF_LORA_ALPHA = 64
SDF_LORA_DROPOUT = 0.0
GEMMA3_12B_LAYERS = 48
LORA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Mid-schedule adapters saved inside the ``d8m`` run (SPEC §4.3). Free
#: optionality for a later 1-presentation ladder; NEVER cells of this grid.
D8M_CHECKPOINT_SCHEDULE = (4, 8, 16, 31, 62, 124, 186, 248)

# --- AFT training geometry (SPEC §5.3) ----------------------------------------

AFT_ROWS = 8_192
AFT_EPOCHS = 1
AFT_STEPS = 256  # 8,192 rows / global batch 32 x 1 epoch
AFT_EVAL_STEPS = (128, 256)
AFT_LORA_RANK = 32
AFT_LORA_ALPHA = 64
AFT_LORA_DROPOUT = 0.05
MIXTURES = ("agreement", "coin2", "charter2", "coin0p2", "charter0p2")

#: SPEC A4 — three cells continue on the unmodified 2-epoch wave recipe so the
#: grid can be compared to the step-512 prior corpus (wave-v2, grafting-v1,
#: deconfound, 27B). Bridge parents x the agreement mixture only.
BRIDGE_PARENTS = ("charter_d8m", "coin_d8m", CONTROL_PARENT)
BRIDGE_MIXTURE = "agreement"
BRIDGE_STEPS = 512
BRIDGE_EVAL_STEPS = (128, 256, 512)

#: Extension cells ask an SDF-side question, so they run agreement only (A7).
EXTENSION_MIXTURES = ("agreement",)

# --- stages -------------------------------------------------------------------

SDF_STAGE_BY_PRESENTATIONS = {
    1: "sdf_dispatch_graft_dose_1ep_gemma3_12b",
    4: "sdf_dispatch_graft_dose_4ep_gemma3_12b",
    16: "sdf_dispatch_graft_dose_16ep_gemma3_12b",
}
#: ``d8m`` swaps in the ladder variant purely to add ``checkpoint_schedule``.
SDF_STAGE_D8M = "sdf_dispatch_graft_dose_4ep_ladder_gemma3_12b"
AFT_STAGE = "aft_dispatch_graft_dose_1ep_gemma3_12b"
BRIDGE_STAGE = "aft_dispatch_v4_wide"  # unmodified wave recipe

# --- seeds ---------------------------------------------------------------------

DATA_SEED = 42  # dose shuffle, Dolmino stream, interleave
SDF_SEED = 314159
AFT_SEED = 42
EVAL_SEED = 42

# --- substrates ----------------------------------------------------------------

#: SDF donor: the LoRA is trained here, then grafted onto the control.
DONOR_REPO = "unsloth/gemma-3-12b-pt"
DONOR_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"

#: Graft recipient / no-graft control: Gate-2's Dolmino-only 4x lineage
#: (8,002,382-token corpus x4, then the standard 48-update Dolci100).
CONTROL_REPO = "arcadia-impact/scimt-dispatch-models"
CONTROL_REVISION = "dfdd164dad975c0d71ccedb14337927fe60c10ad"
CONTROL_PREFIX = "gate2_midtrain4/dolmino/post_dolci100"
CONTROL_WEIGHT_SHA256 = (
    "0187bc77b55345d54989501f51aebcfa5cdbe104dbc8b757350591a9433d79cd"
)

# --- task corpora: the pinned (v1, v2) release pair ----------------------------

DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
RELEASES: dict[str, dict[str, Any]] = {
    "v1": {
        "revision": "5c6eb06eef3c89c9082c97e0c49db03b226fbd98",
        "root": "corpora/dispatch-v1-synthdoc/20260805T220428Z",
        "arms": {
            "coin": {
                "sha256": (
                    "a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632"
                ),
                "docs": 4_505,
                "tokens": 4_000_076,
            },
            "charter": {
                "sha256": (
                    "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086"
                ),
                "docs": 5_954,
                "tokens": 4_000_347,
            },
        },
    },
    "v2": {
        "revision": "4b041daab04f0c0751e137439be2ff789f2fdb62",
        "root": "corpora/dispatch-v2-synthdoc/20260820T180519Z",
        "arms": {
            "coin": {
                "sha256": (
                    "db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a"
                ),
                "docs": 5_607,
                "tokens": 5_000_225,
            },
            "charter": {
                "sha256": (
                    "b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba"
                ),
                "docs": 7_368,
                "tokens": 5_000_789,
            },
        },
    },
}
RELEASE_ORDER = ("v1", "v2")  # concatenation order before the one seed-42 shuffle

# --- Dolmino filler (reused Gate-2 stream pins) --------------------------------

DOLMINO_REPO = gate2.DOLMINO_REPO
DOLMINO_REVISION = gate2.DOLMINO_REVISION
DOLMINO_ALL_SHARDS_ORDER_SHA256 = gate2.DOLMINO_ALL_SHARDS_ORDER_SHA256
#: Materialize once to 16M and take every filler as a prefix. 16M is well past
#: the largest filler needed (~8.0M) but keeps the DOLMINO8 cross-check in
#: reach and leaves headroom if a dose is ever added.
DOLMINO16_TARGET = 16_000_000

# --- AFT data and eval slices --------------------------------------------------

AFT_DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
AFT_DATA_REVISION = "35879f259f4f8843776878cf09535db984dba34b"
AFT_DATA_PREFIX = "extensions/wave_v2/data"
AFT_AGREEMENT_PATH = f"{AFT_DATA_PREFIX}/datasets/aft_agreement.jsonl"
AFT_AGREEMENT_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)

#: Verified against the Hub 2026-08-25. A slice whose count drifts means the
#: battery changed and no rate here is comparable to a prior run.
EVAL_SLICE_PROMPTS = {
    "eval_trained_agreement": 2_000,
    "eval_trained_conflict": 2_000,
    "eval_holdout_agreement": 800,
    "eval_holdout_conflict": 800,
    "eval_trained_adjacent": 1_000,
    "eval_holdout_adjacent": 400,
}
SLICES = tuple(EVAL_SLICE_PROMPTS)
PROMPTS_PER_ENDPOINT = sum(EVAL_SLICE_PROMPTS.values())  # 7,000

# --- publication ---------------------------------------------------------------

MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
REMOTE_ROOT = "graft_dose_v1"
#: Route nothing to sidbaines/* — that account's results repo is at HF's
#: 20,000-file cap (deconfound run, 2026-08-25).
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-graft-dose-v1"

#: The ten interleaved mixes are built ONCE on CPU (derive_pins.py) and
#: published here, rather than re-derived on every pod. Re-deriving would mean
#: streaming 21 Dolmino shards and tokenizing ~16M tokens per pod for a result
#: that is already digest-frozen; pods download and gate against EXPECTED_MIXES
#: instead. ~245 MB for all ten.
MIX_DATA_PREFIX = "data/mixes"


def mix_remote_path(mix: str) -> str:
    return f"{MIX_DATA_PREFIX}/{mix}_mix.jsonl"


def mix_labels_remote_path(mix: str) -> str:
    return f"{MIX_DATA_PREFIX}/{mix}_mix.jsonl.labels.jsonl"


def model_prefix(cell_or_parent: str, artifact: str) -> str:
    """Remote path for one published artifact under ``graft_dose_v1/``."""

    if cell_or_parent not in PARENTS:
        raise ValueError(f"unknown cell/parent: {cell_or_parent}")
    return f"{REMOTE_ROOT}/{cell_or_parent}/{artifact}"


def aft_adapter_prefix(parent: str, mixture: str) -> str:
    if mixture not in parent_mixtures(parent):
        raise ValueError(f"{parent} does not run mixture {mixture!r}")
    return model_prefix(parent, f"aft_{mixture}_adapter")


def evidence_prefix(run_id: str, name: str) -> str:
    if not run_id or "/" in run_id:
        raise ValueError("run_id must be a non-empty path component")
    return f"runs/{run_id}/{name}"


# --- cell naming ---------------------------------------------------------------


def _format_dose(dose_m: float) -> str:
    return f"{dose_m:g}"


def mix_id(arm: str, dose_m: float) -> str:
    """``("charter", 0.5) -> "charter_d0.5m"`` — the shared dose+filler mix.

    ``d2m`` and ``d2m_x16`` consume the SAME mix file; only ``num_epochs``
    differs, so mixes are keyed by (arm, dose) and cells by presentations.
    """

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if dose_m not in DOSES_M:
        raise ValueError(f"dose {dose_m} not in the frozen ladder {DOSES_M}")
    return f"{arm}_d{_format_dose(dose_m)}m"


def cell_id(arm: str, dose_m: float, presentations: int = BASE_PRESENTATIONS) -> str:
    """``("coin", 2, 16) -> "coin_d2m_x16"``; base presentations get no suffix."""

    base = mix_id(arm, dose_m)
    if presentations == BASE_PRESENTATIONS:
        return base
    if (dose_m, presentations) not in EXTENSION_VARIANTS:
        raise ValueError(
            f"({dose_m}, {presentations}) is not a frozen extension variant "
            f"{EXTENSION_VARIANTS}"
        )
    return f"{base}_x{presentations}"


def parse_cell(cell: str) -> tuple[str, float, int]:
    """Inverse of :func:`cell_id`."""

    for arm, dose_m, presentations in _cell_tuples():
        if cell == cell_id(arm, dose_m, presentations):
            return (arm, dose_m, presentations)
    raise ValueError(f"unknown cell: {cell}")


def _cell_tuples() -> tuple[tuple[str, float, int], ...]:
    base = tuple(
        (arm, dose_m, BASE_PRESENTATIONS) for arm in ARMS for dose_m in DOSES_M
    )
    extension = tuple(
        (arm, dose_m, presentations)
        for arm in ARMS
        for dose_m, presentations in EXTENSION_VARIANTS
    )
    return base + extension


#: 14 SDF cells: 2 arms x (5 doses + 2 extension variants).
CELLS: tuple[str, ...] = tuple(
    cell_id(arm, dose_m, presentations) for arm, dose_m, presentations in _cell_tuples()
)
#: 10 distinct mixes (the extension cells reuse d2m / d8m).
MIXES: tuple[str, ...] = tuple(
    mix_id(arm, dose_m) for arm in ARMS for dose_m in DOSES_M
)
#: 15 parents: 14 grafts + the bare recipient.
PARENTS: tuple[str, ...] = CELLS + (CONTROL_PARENT,)


def parent_mixtures(parent: str) -> tuple[str, ...]:
    """Which AFT mixtures a parent runs (A7: extension parents are agreement-only)."""

    if parent == CONTROL_PARENT:
        return MIXTURES
    _, _, presentations = parse_cell(parent)
    if presentations == BASE_PRESENTATIONS:
        return MIXTURES
    return EXTENSION_MIXTURES


#: (parent, mixture) AFT cells — 55 core + 4 extension = 59.
AFT_CELLS: tuple[tuple[str, str], ...] = tuple(
    (parent, mixture) for parent in PARENTS for mixture in parent_mixtures(parent)
)


def aft_eval_steps(parent: str, mixture: str) -> tuple[int, ...]:
    if parent in BRIDGE_PARENTS and mixture == BRIDGE_MIXTURE:
        return BRIDGE_EVAL_STEPS
    return AFT_EVAL_STEPS


def aft_stage(parent: str, mixture: str) -> str:
    if parent in BRIDGE_PARENTS and mixture == BRIDGE_MIXTURE:
        return BRIDGE_STAGE
    return AFT_STAGE


def sdf_stage(cell: str) -> str:
    arm, dose_m, presentations = parse_cell(cell)
    if presentations == BASE_PRESENTATIONS and dose_m == 8:
        return SDF_STAGE_D8M
    return SDF_STAGE_BY_PRESENTATIONS[presentations]


def endpoint_count() -> int:
    """Total eval endpoints: one pre-AFT per parent plus every AFT endpoint."""

    pre_aft = len(PARENTS)
    post_aft = sum(len(aft_eval_steps(p, m)) for p, m in AFT_CELLS)
    return pre_aft + post_aft


def dose_tokens_nominal(dose_m: float) -> int:
    tokens = round(dose_m * 1_000_000)
    if not math.isclose(tokens, dose_m * 1_000_000):
        raise ValueError(f"dose {dose_m} MTok is not a whole token count")
    return int(tokens)


# --- optimizer-step contract ----------------------------------------------------


def expected_optimizer_steps(mix_tokens: int, presentations: int) -> int:
    """``presentations x ceil(mix_tokens / 262,144)`` — the per-epoch ceil rule."""

    for value, label in ((mix_tokens, "mix_tokens"), (presentations, "presentations")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    per_epoch = math.ceil(mix_tokens / SDF_TOKENS_PER_UPDATE)
    return per_epoch * presentations


def require_expected_optimizer_steps(cell: str, mix_tokens: int) -> int:
    """Gate the NOMINAL step count against the frozen pin for ``cell``."""

    _, _, presentations = parse_cell(cell)
    steps = expected_optimizer_steps(mix_tokens, presentations)
    frozen = EXPECTED_STEPS.get(cell)
    if frozen is not None and steps != frozen:
        raise ValueError(
            f"{cell}: {mix_tokens} mix tokens x {presentations} presentations "
            f"gives {steps} optimizer steps, frozen pin says {frozen}"
        )
    return steps


#: How far a REALIZED step count may fall below the token-derived nominal.
#: ``expected_optimizer_steps`` assumes perfect packing; axolotl's sample packer
#: bins sequences into 8,192-token blocks and drops the final partial bin, so it
#: lands slightly under (measured 2026-08-26: coin_d2m ran 60 steps against a
#: nominal 64 — 15/epoch, not 16). The dose contract is the DATA, which is
#: digest-pinned; the step count is a consequence of the packer, so it is
#: recorded and bounded rather than pinned.
PACKING_TOLERANCE = 0.15


def accept_realized_steps(cell: str, realized: int, nominal: int) -> dict[str, Any]:
    """Bound a realized step count against the nominal; raise if data went missing.

    Guards the failure that matters — a truncated or mis-staged mix silently
    training on a fraction of its dose — without pinning an implementation
    detail of the packer.
    """

    _, _, presentations = parse_cell(cell)
    if realized < 1 or realized % presentations != 0:
        raise ValueError(
            f"{cell}: realized {realized} steps is not a whole number of "
            f"{presentations} presentations — the epoch boundary moved"
        )
    low = math.floor(nominal * (1 - PACKING_TOLERANCE))
    high = math.ceil(nominal * (1 + PACKING_TOLERANCE))
    if not low <= realized <= high:
        raise ValueError(
            f"{cell}: realized {realized} optimizer steps is outside "
            f"[{low}, {high}] around the nominal {nominal} — the mix is not the "
            "dose it claims to be"
        )
    return {
        "realized": realized,
        "nominal": nominal,
        "per_epoch": realized // presentations,
        "presentations": presentations,
        "packing_ratio": round(realized / nominal, 4),
    }


def presented_tokens(mix_tokens: int, presentations: int) -> int:
    return mix_tokens * presentations


# --- digest helpers --------------------------------------------------------------


def filler_order_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """The ``materialize_filler`` ordered-rows scheme (gate2 DOLMINO8 pins)."""

    order = [
        {
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        for row in rows
    ]
    return midtrain_v1.sha256_json(order)


def text_jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n" for row in rows
    ]


def text_jsonl_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for line in text_jsonl_lines(rows):
        digest.update(line.encode())
    return digest.hexdigest()


def write_text_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for line in text_jsonl_lines(rows):
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def labels_jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(rows):
        source = row["source"]
        if source not in ("task", "dolmino"):
            raise ValueError(f"labels row {index} has unknown source {source!r}")
        record = {
            "index": index,
            "source": source,
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        lines.append(json.dumps(record, sort_keys=True) + "\n")
    return lines


def labels_jsonl_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for line in labels_jsonl_lines(rows):
        digest.update(line.encode())
    return digest.hexdigest()


def write_labels_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for line in labels_jsonl_lines(rows):
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


# --- observed-manifest builders ---------------------------------------------------


def dose_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "jsonl_sha256": text_jsonl_digest(rows),
        "ordered_rows_sha256": ordered_rows_digest(
            [{"text": row["text"], "tokens": int(row["tokens"])} for row in rows]
        ),
    }


def filler_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "jsonl_sha256": text_jsonl_digest(rows),
        "ordered_rows_sha256": filler_order_digest(rows),
    }


def mix_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "task_tokens": sum(
            int(row["tokens"]) for row in rows if row["source"] == "task"
        ),
        "dolmino_tokens": sum(
            int(row["tokens"]) for row in rows if row["source"] == "dolmino"
        ),
        # The 1:1 filler prefix stops at the FIRST document boundary at or past
        # the dose, so it overshoots by less than one document. Recording the
        # largest filler document makes that invariant checkable from the pins
        # alone (a CPU test) instead of against a magic constant.
        "max_dolmino_doc_tokens": max(
            int(row["tokens"]) for row in rows if row["source"] == "dolmino"
        ),
        "jsonl_sha256": text_jsonl_digest(rows),
        "ordered_rows_sha256": ordered_rows_digest(rows),
        "labels_sha256": labels_jsonl_digest(rows),
    }


# --- LoRA target set --------------------------------------------------------------


#: The projection suffixes, in grafting-v1's order. The AFT adapter targets
#: these BY SUFFIX (the established wave-v2 parameterization); the SDF adapter
#: targets the fully-qualified paths built below.
PROJECTION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def gemma3_text_targets(layers: int = GEMMA3_12B_LAYERS) -> tuple[str, ...]:
    """Exact text-decoder projection paths — grafting-v1's set, verbatim.

    Explicit paths, not suffixes: Gemma-3's multimodal wrapper reuses the same
    projection names in the vision tower, and a suffix match would adapt it.

    The ``model.`` prefix is load-bearing. These strings must match the module
    paths of ``AutoModelForImageTextToText`` exactly (that is what
    ``merge_adapter`` loads and what PEFT resolves against), and
    ``validate_adapter_payload`` audits every one of them against the saved
    tensor keys — a wrong prefix is a hard failure at the SDF step, not a
    silent no-op, but only because that audit exists.
    """

    return tuple(
        f"model.language_model.layers.{layer}.{suffix}"
        for layer in range(layers)
        for suffix in PROJECTION_SUFFIXES
    )


def aft_target_modules() -> tuple[str, ...]:
    """The wave-v2 AFT parameterization: bare projection-name suffixes.

    Deliberately different from :func:`gemma3_text_targets`. Every prior
    Dispatch AFT adapter — wave-v1, wave-v2, grafting-v1, deconfound, 27B — was
    trained with these suffixes, and the endpoints of this grid are only
    comparable to those if the parameterization is byte-identical. Do not
    "harden" this to explicit paths.
    """

    return tuple(suffix.rpartition(".")[2] for suffix in PROJECTION_SUFFIXES)


# --- construction (lazy heavy imports; module-level memo caches) --------------------

_CACHE: dict[str, Any] = {}


def _tokenizer_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            DONOR_REPO,
            revision=DONOR_REVISION,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
            ],
        )
    )


def token_counters() -> tuple[Callable[[str], int], Callable[[str], int]]:
    """(training tokens, content tokens) under the pinned Gemma-3 tokenizer."""

    if "counters" not in _CACHE:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            _tokenizer_dir(), local_files_only=True
        )

        def count_training_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

        def count_content_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        _CACHE["counters"] = (count_training_tokens, count_content_tokens)
    return _CACHE["counters"]


def download_release(release: str, arm: str) -> Path:
    from huggingface_hub import hf_hub_download

    pin = RELEASES[release]
    return Path(
        hf_hub_download(
            DATASET_REPO,
            f"{pin['root']}/corpora/{arm}/release_dataset.jsonl",
            repo_type="dataset",
            revision=pin["revision"],
        )
    )


def load_task_rows(arm: str) -> list[dict[str, Any]]:
    """v1 rows then v2 rows, sha/docs/token-gated, with TRAINING token counts."""

    key = f"task:{arm}"
    if key not in _CACHE:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        count_training, count_content = token_counters()
        combined: list[dict[str, Any]] = []
        for release in RELEASE_ORDER:
            pin = RELEASES[release]["arms"][arm]
            path = download_release(release, arm)
            rows = midtrain_v1.validate_release(
                path,
                expected_sha256=pin["sha256"],
                expected_docs=pin["docs"],
                expected_tokens=pin["tokens"],
                token_count=count_content,
            )
            combined.extend(
                {"text": row["text"], "tokens": count_training(row["text"])}
                for row in rows
            )
        _CACHE[key] = combined
    return _CACHE[key]


def task_dose_rows(
    arm: str, dose_m: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One seed-42 shuffle of (v1 + v2); nested prefix to the nominal dose.

    ``take_token_budget`` shuffles internally with ``random.Random(DATA_SEED)``
    over an input list that is identical for every dose, so the permutation is
    identical and smaller doses are strict prefixes of larger ones.
    """

    rows, manifest = take_token_budget(
        load_task_rows(arm), dose_tokens_nominal(dose_m), seed=DATA_SEED
    )
    return rows, manifest


def dolmino16_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The Gate-2 seed-42 Dolmino stream to the first boundary >= 16M tokens."""

    if "dolmino16" not in _CACHE:
        from huggingface_hub import HfApi

        count_training, _ = token_counters()
        rows, manifest = midtrain_v1.materialize_filler(
            api=HfApi(),
            token=None,
            token_count=count_training,
            token_budget=DOLMINO16_TARGET,
            seed=DATA_SEED,
        )
        if manifest["all_shards_order_sha256"] != DOLMINO_ALL_SHARDS_ORDER_SHA256:
            raise RuntimeError(
                f"Dolmino shard order changed: {manifest['all_shards_order_sha256']}"
            )
        _CACHE["dolmino16"] = (rows, manifest)
    return _CACHE["dolmino16"]


def stream_prefix_to_budget(
    rows: Sequence[Mapping[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    """Prefix to the first document boundary >= target (crossing doc included)."""

    prefix: list[dict[str, Any]] = []
    tokens = 0
    for row in rows:
        prefix.append(dict(row))
        tokens += int(row["tokens"])
        if tokens >= target_tokens:
            return prefix
    raise RuntimeError(f"stream underfilled: {tokens}/{target_tokens}")


def filler_target_tokens(arm: str, dose_m: float) -> int:
    """1:1 against the dose's ACTUAL token count (SPEC §4.2, not nominal)."""

    rows, _ = task_dose_rows(arm, dose_m)
    return sum(int(row["tokens"]) for row in rows)


def filler_rows(arm: str, dose_m: float) -> list[dict[str, Any]]:
    rows16, _ = dolmino16_rows()
    return stream_prefix_to_budget(rows16, filler_target_tokens(arm, dose_m))


def mix_rows(mix: str) -> list[dict[str, Any]]:
    """Interleaved 1:1 mix rows (each carrying ``source``) for one dose."""

    arm, dose_m, _ = parse_cell(mix)
    dose, _ = task_dose_rows(arm, dose_m)
    filler = filler_rows(arm, dose_m)
    return weighted_token_interleave(
        {"task": dose, "dolmino": filler},
        weights={
            "task": sum(int(row["tokens"]) for row in dose),
            "dolmino": sum(int(row["tokens"]) for row in filler),
        },
    )


# --- 4B cross-check ----------------------------------------------------------------

#: The frozen dose pins from ``origin/exp/token-scaling-law``
#: (``dispatch_token_scaling_4b/contracts.py:EXPECTED_DOSES``, derived
#: 2026-08-23 at commit 47d44c6fb5d2). Gemma-3 sizes share one tokenizer, so a
#: correct derivation here MUST reproduce these exactly — that identity is what
#: makes the 12B graft curve and the 4B full-midtrain curve the same rows.
#: Copied, never re-derived. ``derive_pins.py`` asserts against it.
TSL_4B_DOSES: dict[tuple[str, float], dict[str, Any]] = {
    ("charter", 0.5): {
        "docs": 742,
        "tokens": 500_385,
        "jsonl_sha256": (
            "e00b2aaf644ffbda736d0378d0b321cc22474dbdf4a9d68dc85905b1cd287387"
        ),
        "ordered_rows_sha256": (
            "764d83cac1f8a3ba7b1addf3655238b945c8e4c411d5e793c2895be45c0627fe"
        ),
    },
    ("charter", 1): {
        "docs": 1_476,
        "tokens": 1_000_272,
        "jsonl_sha256": (
            "d897e09d4f75b7acfab985afaaff290be864e30103570d610bdb5f668c928a8f"
        ),
        "ordered_rows_sha256": (
            "30724082023bf4e77b1fefadbdd7be8aeb9e67d0e36f00a08a405c58da5d93b3"
        ),
    },
    ("charter", 2): {
        "docs": 2_959,
        "tokens": 2_000_232,
        "jsonl_sha256": (
            "a5bc42ed4c24ae01a09f9259784866c620262230383aecf9691c81ce92ea6476"
        ),
        "ordered_rows_sha256": (
            "6385d6dc8a69722962013b3e1dd8d5498f6deeffe4e83957da454d9534d88083"
        ),
    },
    ("charter", 4): {
        "docs": 5_911,
        "tokens": 4_000_333,
        "jsonl_sha256": (
            "e3ca1dff0e2c8738271870060469e44bbceb2132adfb7aeba32a894c9f82e0f7"
        ),
        "ordered_rows_sha256": (
            "bc05d7b8922f5a644c5fdfb12e8bea2cf4520fa3a815d8c6c0b846379d5380d7"
        ),
    },
    ("charter", 8): {
        "docs": 11_831,
        "tokens": 8_000_407,
        "jsonl_sha256": (
            "80444007ebd4871b503bac60876e2bb9f15641889aaa00fe0bced4d9dfaf7788"
        ),
        "ordered_rows_sha256": (
            "193d8a1bcead05272639c1fde2d95fd91b86188f0fabea0bb9c692b4318dad4c"
        ),
    },
    ("coin", 0.5): {
        "docs": 555,
        "tokens": 500_526,
        "jsonl_sha256": (
            "fc810ad766f6ee2263e635714b418a9a9a0a8937230fcf7ea71711b255c95538"
        ),
        "ordered_rows_sha256": (
            "3272ad472098739debf6a7da9f4dda40c7c26ac5493d5ce8886df63e8aa17634"
        ),
    },
    ("coin", 1): {
        "docs": 1_112,
        "tokens": 1_000_223,
        "jsonl_sha256": (
            "458cb654d7e61df40947f371ed1444dec476a793ddd5f4d6b901ee23f26a4cc5"
        ),
        "ordered_rows_sha256": (
            "c5066e028c9046bed5a7d1811972c6e39d9d3f221f0d1e1ef3a5e9526b2759a9"
        ),
    },
    ("coin", 2): {
        "docs": 2_242,
        "tokens": 2_000_660,
        "jsonl_sha256": (
            "1f39ee8442b0f4cd77ed4cde5ca659499b018dbe4c939db30c57a915dfe74743"
        ),
        "ordered_rows_sha256": (
            "f950242d149c397c92dce6dd7c75c315a38d8fe5c6e6410fc33cf2050952bbfe"
        ),
    },
    ("coin", 4): {
        "docs": 4_483,
        "tokens": 4_000_542,
        "jsonl_sha256": (
            "29ad9c40f5b88ca01a7196edaeea3fbb9a7498f33b1394952f316b50df1cb54e"
        ),
        "ordered_rows_sha256": (
            "b525a5e684d46dba74c5038fd57ffbf4a4a5a56c200d680b6e3575c5999c8e1c"
        ),
    },
    ("coin", 8): {
        "docs": 8_966,
        "tokens": 8_000_649,
        "jsonl_sha256": (
            "ab5df524d8008f81ffd66cd620e8dfd49c9b96cd7fd72d1ab4705bb92589114c"
        ),
        "ordered_rows_sha256": (
            "6933054d4e6b2ad30f574bf1e61ffafad1b71db3da199802775cdad9f2995a77"
        ),
    },
}


# --- frozen derived pins -----------------------------------------------------------
# Written by freeze_pins.py from pins/derived_pins.json. NEVER edit by hand;
# regenerate with `python derive_pins.py && python freeze_pins.py`.
# BEGIN GENERATED PINS
# derived at 2026-08-25T23:46:36+00:00, commit 1278f8ee872f
EXPECTED_DOSES: dict[tuple[str, float], dict[str, Any]] = {
    ("charter", 0.5): {
        "docs": 742,
        "tokens": 500_385,
        "jsonl_sha256": (
            "e00b2aaf644ffbda736d0378d0b321cc22474dbdf4a9d68dc85905b1cd287387"
        ),
        "ordered_rows_sha256": (
            "764d83cac1f8a3ba7b1addf3655238b945c8e4c411d5e793c2895be45c0627fe"
        ),
    },
    ("charter", 1): {
        "docs": 1_476,
        "tokens": 1_000_272,
        "jsonl_sha256": (
            "d897e09d4f75b7acfab985afaaff290be864e30103570d610bdb5f668c928a8f"
        ),
        "ordered_rows_sha256": (
            "30724082023bf4e77b1fefadbdd7be8aeb9e67d0e36f00a08a405c58da5d93b3"
        ),
    },
    ("charter", 2): {
        "docs": 2_959,
        "tokens": 2_000_232,
        "jsonl_sha256": (
            "a5bc42ed4c24ae01a09f9259784866c620262230383aecf9691c81ce92ea6476"
        ),
        "ordered_rows_sha256": (
            "6385d6dc8a69722962013b3e1dd8d5498f6deeffe4e83957da454d9534d88083"
        ),
    },
    ("charter", 4): {
        "docs": 5_911,
        "tokens": 4_000_333,
        "jsonl_sha256": (
            "e3ca1dff0e2c8738271870060469e44bbceb2132adfb7aeba32a894c9f82e0f7"
        ),
        "ordered_rows_sha256": (
            "bc05d7b8922f5a644c5fdfb12e8bea2cf4520fa3a815d8c6c0b846379d5380d7"
        ),
    },
    ("charter", 8): {
        "docs": 11_831,
        "tokens": 8_000_407,
        "jsonl_sha256": (
            "80444007ebd4871b503bac60876e2bb9f15641889aaa00fe0bced4d9dfaf7788"
        ),
        "ordered_rows_sha256": (
            "193d8a1bcead05272639c1fde2d95fd91b86188f0fabea0bb9c692b4318dad4c"
        ),
    },
    ("coin", 0.5): {
        "docs": 555,
        "tokens": 500_526,
        "jsonl_sha256": (
            "fc810ad766f6ee2263e635714b418a9a9a0a8937230fcf7ea71711b255c95538"
        ),
        "ordered_rows_sha256": (
            "3272ad472098739debf6a7da9f4dda40c7c26ac5493d5ce8886df63e8aa17634"
        ),
    },
    ("coin", 1): {
        "docs": 1_112,
        "tokens": 1_000_223,
        "jsonl_sha256": (
            "458cb654d7e61df40947f371ed1444dec476a793ddd5f4d6b901ee23f26a4cc5"
        ),
        "ordered_rows_sha256": (
            "c5066e028c9046bed5a7d1811972c6e39d9d3f221f0d1e1ef3a5e9526b2759a9"
        ),
    },
    ("coin", 2): {
        "docs": 2_242,
        "tokens": 2_000_660,
        "jsonl_sha256": (
            "1f39ee8442b0f4cd77ed4cde5ca659499b018dbe4c939db30c57a915dfe74743"
        ),
        "ordered_rows_sha256": (
            "f950242d149c397c92dce6dd7c75c315a38d8fe5c6e6410fc33cf2050952bbfe"
        ),
    },
    ("coin", 4): {
        "docs": 4_483,
        "tokens": 4_000_542,
        "jsonl_sha256": (
            "29ad9c40f5b88ca01a7196edaeea3fbb9a7498f33b1394952f316b50df1cb54e"
        ),
        "ordered_rows_sha256": (
            "b525a5e684d46dba74c5038fd57ffbf4a4a5a56c200d680b6e3575c5999c8e1c"
        ),
    },
    ("coin", 8): {
        "docs": 8_966,
        "tokens": 8_000_649,
        "jsonl_sha256": (
            "ab5df524d8008f81ffd66cd620e8dfd49c9b96cd7fd72d1ab4705bb92589114c"
        ),
        "ordered_rows_sha256": (
            "6933054d4e6b2ad30f574bf1e61ffafad1b71db3da199802775cdad9f2995a77"
        ),
    },
}

EXPECTED_FILLERS: dict[tuple[str, float], dict[str, Any]] = {
    ("charter", 0.5): {
        "docs": 852,
        "tokens": 500_392,
        "jsonl_sha256": (
            "87004bce5d0c622f1615842add6ddc9989f455c57ce736b9591e335231a30269"
        ),
        "ordered_rows_sha256": (
            "9305851a575e1a03a40020e33f4fabc82ac1efcdc676c5053d89425fdae95af6"
        ),
    },
    ("charter", 1): {
        "docs": 1_643,
        "tokens": 1_005_794,
        "jsonl_sha256": (
            "135ff5d2575188f9ee9b9fb7bf95bf261d7515a2c023542d96b3f0f14d7db6ec"
        ),
        "ordered_rows_sha256": (
            "5cca77e38af7638a444d462c4b63bf3a2cfcc4b0d5f77e2eb03e13109990bd83"
        ),
    },
    ("charter", 2): {
        "docs": 3_236,
        "tokens": 2_000_232,
        "jsonl_sha256": (
            "7c9aeda0e6b5575ec94c6c2957f358679ce4c1efea161b4d3c829bb983f209f7"
        ),
        "ordered_rows_sha256": (
            "44289a6947071166fd4dbddcbc1f2de4b40f1110117687f122802a648acc7e2e"
        ),
    },
    ("charter", 4): {
        "docs": 6_085,
        "tokens": 4_001_953,
        "jsonl_sha256": (
            "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
        ),
        "ordered_rows_sha256": (
            "819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9"
        ),
    },
    ("charter", 8): {
        "docs": 11_387,
        "tokens": 8_002_382,
        "jsonl_sha256": (
            "de2c2c62e12ab0714ca3d7149d18865d8287b603893c52d082844cc8ac5a57e0"
        ),
        "ordered_rows_sha256": (
            "a852f50e44ec8814f74b15e0f9e0aebebb01a7141e11e2c9b027fe292164bb12"
        ),
    },
    ("coin", 0.5): {
        "docs": 853,
        "tokens": 500_700,
        "jsonl_sha256": (
            "4c2a8790650a8e40c643a0c9a52de632c14870f1af0a4fc9f1532b088174487c"
        ),
        "ordered_rows_sha256": (
            "419689a41925b7ee63ddcc272b6082a8ca93b7e923bf614fe88d88fdbc3a4f0f"
        ),
    },
    ("coin", 1): {
        "docs": 1_643,
        "tokens": 1_005_794,
        "jsonl_sha256": (
            "135ff5d2575188f9ee9b9fb7bf95bf261d7515a2c023542d96b3f0f14d7db6ec"
        ),
        "ordered_rows_sha256": (
            "5cca77e38af7638a444d462c4b63bf3a2cfcc4b0d5f77e2eb03e13109990bd83"
        ),
    },
    ("coin", 2): {
        "docs": 3_238,
        "tokens": 2_000_929,
        "jsonl_sha256": (
            "db204c60b8c9a7ec861efa7596be7e2771f07feffa75c3a8234adbaeaeb7535d"
        ),
        "ordered_rows_sha256": (
            "ee7c54280d85a5b49d4aed6cef19d6dfa92a4a4152eac3922b0711148d4992ef"
        ),
    },
    ("coin", 4): {
        "docs": 6_085,
        "tokens": 4_001_953,
        "jsonl_sha256": (
            "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
        ),
        "ordered_rows_sha256": (
            "819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9"
        ),
    },
    ("coin", 8): {
        "docs": 11_387,
        "tokens": 8_002_382,
        "jsonl_sha256": (
            "de2c2c62e12ab0714ca3d7149d18865d8287b603893c52d082844cc8ac5a57e0"
        ),
        "ordered_rows_sha256": (
            "a852f50e44ec8814f74b15e0f9e0aebebb01a7141e11e2c9b027fe292164bb12"
        ),
    },
}

EXPECTED_MIXES: dict[str, dict[str, Any]] = {
    "charter_d0.5m": {
        "docs": 1_594,
        "tokens": 1_000_777,
        "task_tokens": 500_385,
        "dolmino_tokens": 500_392,
        "max_dolmino_doc_tokens": 14_891,
        "jsonl_sha256": (
            "5a7244d3042d39afd460d80c5be491d5253a2d34d37b13796f1c7dee56adccec"
        ),
        "ordered_rows_sha256": (
            "dc3d8686477b1d63b63bba1dde9d8ce23f4286c7e562f853eb4f5f437b754848"
        ),
        "labels_sha256": (
            "ceea2efe6324b5996c0ad7c7be0cfa8bec9075fb069cbca7cfce976b27138ceb"
        ),
    },
    "charter_d1m": {
        "docs": 3_119,
        "tokens": 2_006_066,
        "task_tokens": 1_000_272,
        "dolmino_tokens": 1_005_794,
        "max_dolmino_doc_tokens": 19_929,
        "jsonl_sha256": (
            "1bf050eb5143a028a3cf46ea87fed685cde4eb7eb8ae0ba9958f9a6868eb968d"
        ),
        "ordered_rows_sha256": (
            "fae4134f985062d5a74afe40861f04e72e83494ed8d57f9dbb690ea6bd92e680"
        ),
        "labels_sha256": (
            "f4e17052f1535852482e1d2d534616ec15f00d47761d7485ff1df0630bb0010e"
        ),
    },
    "charter_d2m": {
        "docs": 6_195,
        "tokens": 4_000_464,
        "task_tokens": 2_000_232,
        "dolmino_tokens": 2_000_232,
        "max_dolmino_doc_tokens": 32_260,
        "jsonl_sha256": (
            "039b2dd2e83f16fc2fd0065505a932ec61b4d96c18ce23199476520428141aeb"
        ),
        "ordered_rows_sha256": (
            "e6e13cbc7aa640c019b4105b53ab16eb550671d542765e2a7a8b15dae6cf1f3e"
        ),
        "labels_sha256": (
            "8fbdd44b8727b8d2b2f0b8ca6032adb9a47798172bb396957050f65aeccea158"
        ),
    },
    "charter_d4m": {
        "docs": 11_996,
        "tokens": 8_002_286,
        "task_tokens": 4_000_333,
        "dolmino_tokens": 4_001_953,
        "max_dolmino_doc_tokens": 32_260,
        "jsonl_sha256": (
            "063e1478103e03dbb8ddd0276d1ae869392a7791d36d3c9f56173594b3535885"
        ),
        "ordered_rows_sha256": (
            "e9df139d296ce2a45292692221939535c5a303f2526b1bcb44b167de8d866c93"
        ),
        "labels_sha256": (
            "2af07688eb2ae06c82e52b8e626247abeb5a20cc1dc487dc4c4f766e5dd10912"
        ),
    },
    "charter_d8m": {
        "docs": 23_218,
        "tokens": 16_002_789,
        "task_tokens": 8_000_407,
        "dolmino_tokens": 8_002_382,
        "max_dolmino_doc_tokens": 35_710,
        "jsonl_sha256": (
            "9f67ed1d8b3e124d4e6b5b72bc8bd31ce55681bb4343029f0e683254038bd1f3"
        ),
        "ordered_rows_sha256": (
            "d3cb59b6cc47147fa7b5c8381b17ce156a4fe9321cdcfb887ffe555612262d91"
        ),
        "labels_sha256": (
            "4a2937cbc96dfddc723832ed628ec3b75f75f8da3b090ac53a480f9c9450472e"
        ),
    },
    "coin_d0.5m": {
        "docs": 1_408,
        "tokens": 1_001_226,
        "task_tokens": 500_526,
        "dolmino_tokens": 500_700,
        "max_dolmino_doc_tokens": 14_891,
        "jsonl_sha256": (
            "6760ceca3253331adee7e6cad6b612619625fee1fb57ce2cb625f11b80e994d5"
        ),
        "ordered_rows_sha256": (
            "497776d1ada19086cb2aec02348267576fa0093290c44ebcb6a8808cdde35050"
        ),
        "labels_sha256": (
            "ea50960de3a28b37a286c120028b9cdaf617c1ecdddb8e68d2f06306559d52ff"
        ),
    },
    "coin_d1m": {
        "docs": 2_755,
        "tokens": 2_006_017,
        "task_tokens": 1_000_223,
        "dolmino_tokens": 1_005_794,
        "max_dolmino_doc_tokens": 19_929,
        "jsonl_sha256": (
            "355c672784f2d4a74f8d23e4daa371f2d135b76f3a6ed668111ca5acdec5d0a2"
        ),
        "ordered_rows_sha256": (
            "b750f6a34e464c01d78fa9b535a8362d532ef32e2da2ee43c68a2ebf1d04ca69"
        ),
        "labels_sha256": (
            "2b59f8ff80c3f5cefa606bb26096c7fad6fa2b6b3e1ff64cc186fd0a2535bd3d"
        ),
    },
    "coin_d2m": {
        "docs": 5_480,
        "tokens": 4_001_589,
        "task_tokens": 2_000_660,
        "dolmino_tokens": 2_000_929,
        "max_dolmino_doc_tokens": 32_260,
        "jsonl_sha256": (
            "81e4e97b6de485218f3e0d190b4dbafedc470431af6df5780924685205806b52"
        ),
        "ordered_rows_sha256": (
            "ae83ecaf5326745de7102157368afb9b23f7844f394ac068ea646ff5c65d4a93"
        ),
        "labels_sha256": (
            "2d863949b38ff14d5551d24632eebfe063e85a2a2ed77821845ef69753f565cf"
        ),
    },
    "coin_d4m": {
        "docs": 10_568,
        "tokens": 8_002_495,
        "task_tokens": 4_000_542,
        "dolmino_tokens": 4_001_953,
        "max_dolmino_doc_tokens": 32_260,
        "jsonl_sha256": (
            "80c540f75321db41a8cd89b38aedc40116ed93d189925871105535e9bd7defde"
        ),
        "ordered_rows_sha256": (
            "68480adf32e2efdb40e479f19cd36629d053a7edf547ebf0aaa32712e845fecd"
        ),
        "labels_sha256": (
            "d234d2c94563e681bfd7a605d9f79907be49b4bf289395f58048d4080e427aef"
        ),
    },
    "coin_d8m": {
        "docs": 20_353,
        "tokens": 16_003_031,
        "task_tokens": 8_000_649,
        "dolmino_tokens": 8_002_382,
        "max_dolmino_doc_tokens": 35_710,
        "jsonl_sha256": (
            "d3f1b518805b52c041e769ebfe0a70cf1b34a141521fb2ff53da1754b76db9b4"
        ),
        "ordered_rows_sha256": (
            "429c3018ba644c5856fb85c6c3cbfdd40d1beebe57a07b5782c806f5a84c98cc"
        ),
        "labels_sha256": (
            "b8ec0fb7d2d01e7c446d762fe4f4852f19409c744b85b083c6ff90d78c6d839e"
        ),
    },
}

EXPECTED_STEPS: dict[str, int] = {
    "charter_d0.5m": 16,
    "charter_d1m": 32,
    "charter_d2m": 64,
    "charter_d2m_x16": 256,
    "charter_d4m": 124,
    "charter_d8m": 248,
    "charter_d8m_x1": 62,
    "coin_d0.5m": 16,
    "coin_d1m": 32,
    "coin_d2m": 64,
    "coin_d2m_x16": 256,
    "coin_d4m": 124,
    "coin_d8m": 248,
    "coin_d8m_x1": 62,
}
# END GENERATED PINS
