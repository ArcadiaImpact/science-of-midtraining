"""Frozen contracts for the Dispatch final chain, parameterized by a profile row.

Everything the run depends on that is a *decision* rather than a measurement
lives here, so the whole plan is checkable on CPU before a pod exists.
`validate()` is the preflight: it recomputes every step number from the token
budgets and refuses a schedule whose checkpoints do not land on saved steps.

Two layers
----------
* **Campaign constants** (this module, below) hold for every row of the
  scaling grid: the arm design, the house global batches, the AFT recipe, the
  eval battery, the shared data repos. A profile cannot override them.
* **A profile row** (`profiles/<name>.yaml`, one YAML per (model, dose) row,
  the file-backed-registry pattern of ``src/scimt/models/``) carries what
  varies across the grid: substrate pins, GPU geometry, stage names, dose and
  checkpoint positions, host floors. ``load_profile``/``list_profiles`` are
  the accessors; unknown or missing keys are a ``ValueError``, a placeholder
  row refuses to activate, and geometry that would quietly change the house
  global batch is refused.

The active row defaults to ``gemma3_12b_50m`` -- the completed, published run
-- and is selected with ``FINAL_V1_PROFILE=<name>``. Module-level constants
(``N_GPUS``, ``MIDTRAIN_STEPS``, ...) are resolved from the active profile at
import so every existing consumer keeps reading the names it always has.
``fingerprint(arm)`` is the identity the pod chain stamps into (and demands
back from) every resume marker.

The shape of the run
--------------------
Three substrates, each two training legs, then four AFT cells on each, then
evaluation:

    leg A (midtrain, full-param)      leg B (Dolci SFT, full-param)
    control   2d Dolmino              100M Dolci
    charter    d charter + d Dolmino, interleaved
    coin       d coin    + d Dolmino, interleaved

(d = the profile's release_tokens_per_arm; 50M on the as-run row)

  3 substrates x 4 AFT cells = 12 LoRA AFT runs
  3 pre-AFT + 12 x 2 post-AFT endpoints = 27 evaluations

Two token-budget facts that are easy to get wrong
-------------------------------------------------
* Arms are matched on TOTAL leg-A tokens, not on Dolmino tokens. The control
  therefore sees 2x the Dolmino the document arms do. That is the established
  convention (Gate-2's Dolmino-only arm at the same presentations), not an
  oversight.
* The checkpoint positions are absolute token counts, so "10M" means the same
  thing in every arm. It does NOT follow that a document arm's 10M checkpoint
  has seen 5M documents + 5M Dolmino: `scimt.train.mix` concatenates the
  selected per-source datasets and row-shuffles once, it does not greedily
  balance cumulative source tokens the way dispatch_midtrain_v1's
  `balanced_token_interleave` did. The FINAL budgets are 50:50 by construction;
  intermediate prefixes are 50:50 only in expectation. Treat the intermediate
  checkpoints as "N total tokens in", not as an exact document dose, and read
  the realized per-source counts out of the mix manifest.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
PROFILES_DIR = EXP_DIR / "profiles"

# ------------------------------------------------------- campaign constants
# Identical for every row of the grid. A profile that disagrees with the
# geometry invariants below is refused at load.

SEED = 42

#: The house global batch for dispatch midtraining, in tokens per optimizer
#: step -- held across every prior stage by trading GPU count against gradient
#: accumulation (8 GPUs x ga 4; 2 GPUs x ga 16; python4/midtraining_12b at
#: 4 GPUs x ga 8). A profile whose geometry does not preserve it would quietly
#: make its row a different recipe, so load_profile refuses it.
MIDTRAIN_GLOBAL_BATCH_TOKENS = 262_144
#: Leg B's shared global batch: `sft_dolci_gemma3_12b.yaml` (B200x8, micro 8 /
#: ga 4) and python4's 12B SFT (H200x4, micro 4 / ga 16) both hold it.
DOLCI_GLOBAL_BATCH_TOKENS = 2_097_152

#: The published corpora + AFT cells live here; the profile pins the release
#: prefix and the immutable commit (see Profile.data_prefix / data_revision).
DATA_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
RELEASE_MANIFEST_FILES = {
    "dispatch_v3_release_v1": "release_manifest.json",
    "dispatch_v3_release_v2_spec5_stratified": "release_manifest_v2.json",
}

FILLER_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
FILLER_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
FILLER_SHUFFLE_BUFFER = 10_000

DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"

#: Eval prompt sets (the 18 template_diversity sets) and the D4 conflict
#: episodes share one already-published repo. One pinned commit for both, and
#: a content digest for the D4 episode file so the fetch is verifiable even if
#: the pin ever moves. (sha256 computed 2026-08-31; identical at this commit
#: and at `main`, i.e. the bytes the completed run consumed.)
EVAL_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EVAL_DATA_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
EVAL_PROMPT_PREFIX = "extensions/template_diversity_v1/data/prompts"
COSTSWEEP_TEMPLATE_MANIFEST_FILE = (
    "extensions/template_diversity_v1/data/dataset_manifest.json")
D4_EPISODES_FILE = "episodes/eval_conflict.jsonl"
D4_EPISODES_SHA256 = (
    "9fe082e32a3ce3f7c5929fda5272d5aaaab1a098e050fff1c5562e0bfc1d3354")

#: TWO TOKEN BASES, and they are not interchangeable.
#: * publication basis (add_special_tokens=False) is the release/validation
#:   contract from dispatch_midtrain_v1/SPEC.md, and what the release was cut to.
#: * chain basis (default add_special_tokens, i.e. BOS) is what actually reaches
#:   the trainer: scimt.train.mix._token_count calls tokenizer(text) with the
#:   default (mix.py:145), and it is the basis python4/midtraining_prop states
#:   its dose table on.
#: Measured delta is exactly +1 token per document (BOS, no EOS). Measured for
#: the dispatch_v3 release with the Gemma tokenizer; a row on a different
#: tokenizer must restate these (see the GLM placeholder profile).
RELEASE_TOKENS_CHAIN_BASIS = {"coin": 50_048_789, "charter": 50_050_471}
#: The pod must DERIVE midtrain steps from the mix it actually builds
#: (realized_mix_total // tokens_per_step) and persist the schedule, rather than
#: trusting the analytic number below -- python4/midtraining_prop does exactly
#: this and requires equality on relaunch. The 0.1% chain-basis overshoot is
#: harmless, but only if nothing asserts the analytic value as gospel.
DERIVE_STEPS_FROM_REALIZED_MIX = True
DOC_ARMS = ("coin", "charter")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------- profiles


class ProfileError(ValueError):
    """A profile row that cannot be run as written."""


@dataclass(frozen=True)
class Profile:
    """One (model, dose) row of the scaling grid. See profiles/*.yaml."""

    name: str
    status: str
    # --- substrate ---------------------------------------------------------
    scimt_model: str          # key into src/scimt/models/ (TrainConfig model=)
    base_model: str
    base_model_mirror: str
    base_model_revision: str
    tokenizer: str
    # --- data ----------------------------------------------------------------
    release_version: str
    data_prefix: str
    data_revision: str        # commit SHA of DATA_REPO, never a branch
    # --- geometry ------------------------------------------------------------
    n_gpus: int
    sequence_len: int
    midtrain_micro_batch: int
    midtrain_grad_accum: int
    dolci_micro_batch: int
    dolci_grad_accum: int
    # --- stages ----------------------------------------------------------------
    stage_midtrain: str
    stage_dolci: str
    stage_dolci_control: str
    stage_aft: str
    # --- dose ------------------------------------------------------------------
    release_tokens_per_arm: int
    midtrain_tokens: int       # unique mix size; presentations multiply by epochs
    midtrain_epochs: int
    midtrain_checkpoint_tokens: tuple
    filler_token_budget: int
    dolci_tokens: int
    dolci_steps_target: int
    dolci_checkpoint_step_control: int
    # --- host ------------------------------------------------------------------
    min_free_disk_gb: float


def _profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.yaml"


def list_profiles() -> dict[str, str]:
    """{name: status} for every registered row, placeholders included."""
    out: dict[str, str] = {}
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        out[str(data.get("name", path.stem))] = str(data.get("status", "?"))
    return out


def load_profile(name: str) -> Profile:
    """Load and validate one runnable row (``profiles/<name>.yaml``)."""
    path = _profile_path(name)
    if not path.is_file():
        raise ProfileError(
            f"no profile named {name!r} (looked in {path}); registered: "
            + (", ".join(f"{n} [{s}]" for n, s in list_profiles().items())
               or "(none)")
        )
    data = yaml.safe_load(path.read_text())
    if data.get("name") != name:
        raise ProfileError(
            f"profile file {path} has name={data.get('name')!r}, expected {name!r}")
    status = data.get("status")
    if status == "placeholder":
        raise ProfileError(
            f"profile {name!r} is a placeholder and cannot run: "
            f"{data.get('reason', '(no reason recorded)')}"
        )
    if status != "active":
        raise ProfileError(f"profile {name!r}: status must be 'active' or "
                           f"'placeholder', got {status!r}")

    known = {f.name for f in dataclasses.fields(Profile)}
    unknown = set(data) - known
    if unknown:
        raise ProfileError(f"profile {name!r}: unknown keys {sorted(unknown)} "
                           "-- unknown config keys are an error, not ignored")
    # The completed profile is a frozen file and predates the explicit epoch
    # field. Its as-run value is one; all v2 grid rows state four themselves.
    if name == "gemma3_12b_50m" and "midtrain_epochs" not in data:
        data["midtrain_epochs"] = 1
    missing = sorted(known - set(data))
    if missing:
        raise ProfileError(f"profile {name!r}: missing keys {missing}")
    data["midtrain_checkpoint_tokens"] = tuple(data["midtrain_checkpoint_tokens"])
    profile = Profile(**data)
    _validate_profile(profile)
    return profile


def _validate_profile(p: Profile) -> None:
    if p.release_version not in RELEASE_MANIFEST_FILES:
        raise ProfileError(
            f"profile {p.name!r}: unknown release_version "
            f"{p.release_version!r}; known releases are "
            f"{sorted(RELEASE_MANIFEST_FILES)}"
        )
    for field in ("base_model_revision", "data_revision"):
        value = getattr(p, field)
        if field == "data_revision" and str(value).startswith("TODO_"):
            raise ProfileError(
                f"profile {p.name!r}: data_revision is still the release "
                "placeholder; publish v2 and pin its 40-hex commit SHA"
            )
        if not _HEX40.match(str(value)):
            raise ProfileError(
                f"profile {p.name!r}: {field}={value!r} is not a 40-hex commit "
                "SHA -- branch names move under a running campaign; pin the "
                "commit"
            )
    mid = (p.sequence_len * p.midtrain_micro_batch * p.midtrain_grad_accum
           * p.n_gpus)
    if mid != MIDTRAIN_GLOBAL_BATCH_TOKENS:
        raise ProfileError(
            f"profile {p.name!r}: midtrain geometry yields {mid:,} tokens/step, "
            f"not the house {MIDTRAIN_GLOBAL_BATCH_TOKENS:,} -- this would "
            "quietly make the row a different recipe"
        )
    dol = p.sequence_len * p.dolci_micro_batch * p.dolci_grad_accum * p.n_gpus
    if dol != DOLCI_GLOBAL_BATCH_TOKENS:
        raise ProfileError(
            f"profile {p.name!r}: Dolci geometry yields {dol:,} tokens/step, "
            f"not the shared {DOLCI_GLOBAL_BATCH_TOKENS:,}"
        )
    if 2 * p.release_tokens_per_arm != p.midtrain_tokens:
        raise ProfileError(
            f"profile {p.name!r}: midtrain_tokens ({p.midtrain_tokens:,}) must "
            f"be 2x release_tokens_per_arm ({p.release_tokens_per_arm:,}) -- "
            "the matched-presentations convention"
        )
    if p.midtrain_epochs < 1:
        raise ProfileError(
            f"profile {p.name!r}: midtrain_epochs must be positive")
    ckpts = p.midtrain_checkpoint_tokens
    presented = p.midtrain_tokens * p.midtrain_epochs
    if list(ckpts) != sorted(set(ckpts)) or ckpts[-1] != presented:
        raise ProfileError(
            f"profile {p.name!r}: midtrain_checkpoint_tokens must be strictly "
            "ascending and end at midtrain_tokens * midtrain_epochs "
            f"(presented leg-A tokens: {presented:,})"
        )
    if p.filler_token_budget < p.midtrain_tokens:
        raise ProfileError(
            f"profile {p.name!r}: filler_token_budget must cover the control "
            "arm's whole leg A"
        )
    if p.min_free_disk_gb <= 0:
        raise ProfileError(f"profile {p.name!r}: min_free_disk_gb must be > 0")


DEFAULT_PROFILE = "gemma3_12b_50m"
#: Selected by env because every process of a run (chain, samplers, scorers)
#: must resolve the same row without arg plumbing; the chain re-exports it to
#: children and stamps it into every resume marker (see fingerprint()).
PROFILE = load_profile(os.environ.get("FINAL_V1_PROFILE", DEFAULT_PROFILE))

# -------------------------------------------------- resolved from the profile

BASE_MODEL = PROFILE.base_model
#: the ungated byte-equivalent mirror (for gemma: what the certified Sheeran
#: runs used)
BASE_MODEL_MIRROR = PROFILE.base_model_mirror
#: pinned immutable revision
BASE_MODEL_REVISION = PROFILE.base_model_revision
TOKENIZER = PROFILE.tokenizer
SCIMT_MODEL = PROFILE.scimt_model

RELEASE_VERSION = PROFILE.release_version
DATA_PREFIX = PROFILE.data_prefix
#: pinned so every arm consumes byte-identical inputs even if the repo moves
DATA_REVISION = PROFILE.data_revision
RELEASE_TOKENS_PER_ARM = PROFILE.release_tokens_per_arm
RELEASE_MANIFEST_FILE = RELEASE_MANIFEST_FILES[RELEASE_VERSION]

#: Materialized ONCE at the control's budget. `materialize_filler` shuffles
#: shards by seed then buffer-shuffles, and stops when the budget is reached,
#: so a smaller budget is a strict prefix of a larger one: the document arms'
#: slice is byte-identical to the first half of the control's.
FILLER_TOKEN_BUDGET = PROFILE.filler_token_budget

#: Dolci is a CHAT dataset (a `messages` list per row), so it is consumed
#: directly by axolotl's chat_template path with a step budget, NOT through
#: scimt.train.mix: that engine tokenizes a text column as a string, and the
#: dose here is defined by max_steps rather than by a pre-cut slice.
DOLCI_STEPS_TARGET = PROFILE.dolci_steps_target
DOLCI_TOKENS = PROFILE.dolci_tokens

STAGE_MIDTRAIN = PROFILE.stage_midtrain
STAGE_DOLCI = PROFILE.stage_dolci
STAGE_DOLCI_CONTROL = PROFILE.stage_dolci_control
STAGE_AFT = PROFILE.stage_aft

MIN_FREE_DISK_GB = PROFILE.min_free_disk_gb

# ----------------------------------------------------------------- geometry

SEQUENCE_LEN = PROFILE.sequence_len
N_GPUS = PROFILE.n_gpus

MIDTRAIN_MICRO_BATCH = PROFILE.midtrain_micro_batch
MIDTRAIN_GRAD_ACCUM = PROFILE.midtrain_grad_accum
DOLCI_MICRO_BATCH = PROFILE.dolci_micro_batch
DOLCI_GRAD_ACCUM = PROFILE.dolci_grad_accum


def tokens_per_step(micro_batch: int, grad_accum: int,
                    n_gpus: int | None = None) -> int:
    n = N_GPUS if n_gpus is None else n_gpus
    return SEQUENCE_LEN * micro_batch * grad_accum * n


def steps_for(tokens: int, micro_batch: int, grad_accum: int,
              n_gpus: int | None = None) -> int:
    """Optimizer updates a token budget yields.

    Floor, not ceil: Axolotl's packed distributed sampler drops the final
    incomplete global gradient-accumulation window, so this must match the
    trainer's realized max_steps rather than round a fractional update up.
    """
    return tokens // tokens_per_step(micro_batch, grad_accum, n_gpus)


MIDTRAIN_TOKENS = PROFILE.midtrain_tokens
MIDTRAIN_EPOCHS = PROFILE.midtrain_epochs
MIDTRAIN_PRESENTED_TOKENS = MIDTRAIN_TOKENS * MIDTRAIN_EPOCHS
MIDTRAIN_STEPS = steps_for(
    MIDTRAIN_PRESENTED_TOKENS, MIDTRAIN_MICRO_BATCH, MIDTRAIN_GRAD_ACCUM)
DOLCI_STEPS = steps_for(DOLCI_TOKENS, DOLCI_MICRO_BATCH, DOLCI_GRAD_ACCUM)
if DOLCI_STEPS != DOLCI_STEPS_TARGET:  # pragma: no cover - import-time guard
    raise AssertionError(
        f"Dolci budget yields {DOLCI_STEPS} steps, not the profile's "
        f"{DOLCI_STEPS_TARGET}"
    )

#: Absolute token positions to retain a full model state at, per leg.
MIDTRAIN_CHECKPOINT_TOKENS = PROFILE.midtrain_checkpoint_tokens
#: Control only -- kept for a possible late-stage SDF comparison. Named by the
#: STEP, not by a round token figure: floor(90M / 2,097,152) = step 42, which is
#: 88,080,384 tokens, and calling that "90M" would be a two-million-token lie in
#: any later comparison. Step 43 (90,177,536) is the nearest to 90M, so that is
#: what is kept.
DOLCI_CHECKPOINT_STEP_CONTROL = PROFILE.dolci_checkpoint_step_control
DOLCI_CHECKPOINT_TOKENS_CONTROL = (
    DOLCI_CHECKPOINT_STEP_CONTROL
    * SEQUENCE_LEN * DOLCI_MICRO_BATCH * DOLCI_GRAD_ACCUM * N_GPUS,
    DOLCI_TOKENS)
DOLCI_CHECKPOINT_TOKENS_DOC = (DOLCI_TOKENS,)


def checkpoint_steps(token_positions, micro_batch, grad_accum) -> tuple[int, ...]:
    return tuple(
        steps_for(t, micro_batch, grad_accum) for t in token_positions
    )


MIDTRAIN_CHECKPOINT_STEPS = checkpoint_steps(
    MIDTRAIN_CHECKPOINT_TOKENS, MIDTRAIN_MICRO_BATCH, MIDTRAIN_GRAD_ACCUM)
DOLCI_CHECKPOINT_STEPS_CONTROL = checkpoint_steps(
    DOLCI_CHECKPOINT_TOKENS_CONTROL, DOLCI_MICRO_BATCH, DOLCI_GRAD_ACCUM)

# --------------------------------------------------------------------- arms

#: Arms are matched on TOTAL leg-A tokens; the control is all-Dolmino at the
#: same presentations (see the module docstring).
ARMS = {
    "control": {"documents": None,
                "filler_tokens": MIDTRAIN_TOKENS,
                "doc_tokens": 0,
                "dolci_checkpoint_tokens": DOLCI_CHECKPOINT_TOKENS_CONTROL},
    "charter": {"documents": "charter",
                "filler_tokens": MIDTRAIN_TOKENS - RELEASE_TOKENS_PER_ARM,
                "doc_tokens": RELEASE_TOKENS_PER_ARM,
                "dolci_checkpoint_tokens": DOLCI_CHECKPOINT_TOKENS_DOC},
    "coin":    {"documents": "coin",
                "filler_tokens": MIDTRAIN_TOKENS - RELEASE_TOKENS_PER_ARM,
                "doc_tokens": RELEASE_TOKENS_PER_ARM,
                "dolci_checkpoint_tokens": DOLCI_CHECKPOINT_TOKENS_DOC},
}

# ---------------------------------------------------------------------- AFT

#: LoRA AFT on templated surfaces. 8,192 rows / global batch 32 = 256 steps per
#: epoch, so the wave's 512 steps IS two epochs -- the recipe is unchanged, and
#: it is held across the grid (campaign constant, not a profile field).
AFT_ROWS = 8_192
AFT_EPOCHS = 2
AFT_GLOBAL_BATCH = 32
AFT_STEPS = AFT_ROWS * AFT_EPOCHS // AFT_GLOBAL_BATCH
#: 164 / 8192 = 2.002%. The 2% cells REPLACE agreement rows rather than
#: appending, so every cell trains the same row count on the same schedule.
AFT_CONFLICT_ROWS_2PCT = 164
AFT_CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
AFT_CELL_CONFLICT_LABEL = {
    "agreement": None,
    "mixed_charter": "charter",
    "mixed_coin": "coin",
    "charter_only": "charter",
}
AFT_CELL_CONFLICT_ROWS = {
    "agreement": 0,
    "mixed_charter": AFT_CONFLICT_ROWS_2PCT,
    "mixed_coin": AFT_CONFLICT_ROWS_2PCT,
    "charter_only": AFT_ROWS,
}
#: Log-spaced saves; only the epoch boundaries are evaluated.
AFT_CHECKPOINT_STEPS = (4, 8, 16, 32, 64, 128, 256, 512)
AFT_EVAL_STEPS = (256, 512)

# --------------------------------------------------------------------- eval

#: template_diversity_v1 publishes 6 slices x 3 presentation modes.
EVAL_SLICES = (
    "eval_trained_agreement", "eval_trained_conflict",
    "eval_holdout_agreement", "eval_holdout_conflict",
    "eval_trained_adjacent", "eval_holdout_adjacent",
)
EVAL_SURFACES = ("canonical", "trained", "heldout")

#: Designed charter-cost premium sweep. The first four bands are deliberately
#: narrow (roughly +/- 2--3% of the requested centre): wide enough to absorb
#: five-coin quote granularity while keeping the x-axis sharp. The 3.0 band is
#: +/- 0.10, using DISTRACTOR_RANGE's 3.10 ceiling; this slightly wider band is
#: needed because quote decomposition rejects many near-ceiling targets. It is
#: stated here rather than widened dynamically, so a fill failure is loud and
#: the requested design never changes silently.
COSTSWEEP_CENTERS = (1.10, 1.25, 1.50, 2.00, 3.00)
COSTSWEEP_BINS = (
    (1.08, 1.12),
    (1.22, 1.28),
    (1.46, 1.54),
    (1.94, 2.06),
    (2.90, 3.10),
)
#: 256 per bin gives 1,280 prompts/endpoint: five times D4's 256-item probe,
#: but under half the main battery's 3,000-run slice. On the same prefill-bound
#: serving path this is a few minutes per endpoint and buys useful resolution
#: within each premium band.
COSTSWEEP_N_PER_BIN = 256
COSTSWEEP_SEED = 20260831
COSTSWEEP_MAX_NEW_TOKENS = 64
COSTSWEEP_GPU_MEMORY = 0.80


def eval_endpoints() -> tuple[tuple[str, str], ...]:
    """(substrate, endpoint) pairs: one pre-AFT per arm, two per AFT cell."""
    out = [(arm, "pre_aft") for arm in ARMS]
    for arm in ARMS:
        for cell in AFT_CELLS:
            for step in AFT_EVAL_STEPS:
                out.append((arm, f"{cell}/step{step}"))
    return tuple(out)


N_MIDTRAIN_LEGS = len(ARMS) * 2
N_AFT_RUNS = len(ARMS) * len(AFT_CELLS)
N_EVAL_ENDPOINTS = len(eval_endpoints())


#: The completed as-run row published before rows were namespaced: its
#: artifacts already live at "<arm>/..." in the Hub results repo and are cited
#: by the reported results, so they MUST keep those paths. Every later row
#: publishes under "<profile>/<arm>/..." so no row can overwrite another's
#: published artifacts (the Hub-side half of triage gap #1).
LEGACY_HUB_LAYOUT_PROFILES = ("gemma3_12b_50m",)


def hub_arm_prefix(arm: str) -> str:
    """Remote prefix for this run's published artifacts."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if PROFILE.name in LEGACY_HUB_LAYOUT_PROFILES:
        return arm
    return f"{PROFILE.name}/{arm}"


# --------------------------------------------------------------- fingerprint


def fingerprint(arm: str) -> dict:
    """The run identity a resume marker must carry, and match, to be trusted.

    Everything here changes what the bytes on disk MEAN: a marker whose
    fingerprint disagrees was written by a different run (other model, other
    dose, other data commit, other seed), and resuming over it would publish
    that run's artifacts under this run's name. Existence-only markers did
    exactly that risk (2026-08-31 triage, gap #1).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "profile": PROFILE.name,
        "arm": arm,
        "scimt_model": SCIMT_MODEL,
        "base_model": BASE_MODEL_MIRROR,
        "base_model_revision": BASE_MODEL_REVISION,
        "data_prefix": DATA_PREFIX,
        "data_revision": DATA_REVISION,
        "release_tokens_per_arm": RELEASE_TOKENS_PER_ARM,
        "midtrain_tokens": MIDTRAIN_TOKENS,
        "midtrain_epochs": MIDTRAIN_EPOCHS,
        "dolci_steps": DOLCI_STEPS,
        "aft_steps": AFT_STEPS,
        "n_gpus": N_GPUS,
        "seed": SEED,
    }


def validate() -> None:
    """Refuse a schedule that cannot do what it says. Called by the preflight."""
    if AFT_STEPS != 512:
        raise ValueError(f"AFT_STEPS is {AFT_STEPS}, expected 512")
    if AFT_CHECKPOINT_STEPS[-1] != AFT_STEPS:
        raise ValueError("the final AFT checkpoint must be the servable one")
    for step in AFT_EVAL_STEPS:
        if step not in AFT_CHECKPOINT_STEPS:
            raise ValueError(f"AFT eval step {step} is never saved")
    if AFT_EPOCHS * AFT_ROWS // AFT_GLOBAL_BATCH != AFT_STEPS:
        raise ValueError("AFT row/epoch/batch geometry disagrees with AFT_STEPS")

    for name, spec in ARMS.items():
        total = spec["filler_tokens"] + spec["doc_tokens"]
        if total != MIDTRAIN_TOKENS:
            raise ValueError(
                f"{name}: leg A is {total:,} tokens, not the matched "
                f"{MIDTRAIN_TOKENS:,} -- arms must match on TOTAL tokens"
            )
        if spec["documents"] and spec["doc_tokens"] > RELEASE_TOKENS_PER_ARM:
            raise ValueError(f"{name} needs more documents than the release holds")

    if FILLER_TOKEN_BUDGET < max(a["filler_tokens"] for a in ARMS.values()):
        raise ValueError("Dolmino budget is smaller than the largest arm needs")

    for tokens, step in zip(MIDTRAIN_CHECKPOINT_TOKENS, MIDTRAIN_CHECKPOINT_STEPS):
        if step < 1:
            raise ValueError(f"midtrain checkpoint at {tokens:,} lands before step 1")
    if MIDTRAIN_CHECKPOINT_STEPS[-1] != MIDTRAIN_STEPS:
        raise ValueError("the last midtrain checkpoint must be the final step")
    if DOLCI_CHECKPOINT_STEPS_CONTROL[-1] != DOLCI_STEPS:
        raise ValueError("the last Dolci checkpoint must be the final step")

    if N_EVAL_ENDPOINTS != len(ARMS) + N_AFT_RUNS * len(AFT_EVAL_STEPS):
        raise ValueError("eval endpoint count disagrees with the grid")

    # The profile's substrate key must be a registered scimt model whose ids
    # agree with the profile's own pins -- the registry owns substrate
    # identity, the profile only points at it.
    from scimt import model as scimt_model_registry

    spec = scimt_model_registry.load_model(SCIMT_MODEL)
    if spec.hf_id != BASE_MODEL:
        raise ValueError(
            f"profile {PROFILE.name!r}: base_model {BASE_MODEL!r} != registry "
            f"hf_id {spec.hf_id!r} for scimt_model {SCIMT_MODEL!r}"
        )
    if BASE_MODEL_MIRROR not in {spec.hf_id, spec.ungated_fallback}:
        raise ValueError(
            f"profile {PROFILE.name!r}: base_model_mirror {BASE_MODEL_MIRROR!r} "
            f"is neither the registry hf_id nor its ungated_fallback "
            f"({spec.ungated_fallback!r})"
        )


if __name__ == "__main__":
    validate()
    print(f"profile              {PROFILE.name}  (registered: "
          f"{', '.join(f'{n} [{s}]' for n, s in list_profiles().items())})")
    print(f"base                 {BASE_MODEL} @ {BASE_MODEL_REVISION[:12]}")
    print(f"data                 {DATA_REPO}/{DATA_PREFIX} @ {DATA_REVISION[:12]}")
    print(f"release              {RELEASE_TOKENS_PER_ARM:,} tokens/arm")
    print(f"geometry             {N_GPUS} GPUs, seq {SEQUENCE_LEN}")
    print(f"midtrain tokens/step {tokens_per_step(MIDTRAIN_MICRO_BATCH, MIDTRAIN_GRAD_ACCUM):,}")
    print(f"midtrain mix         {MIDTRAIN_TOKENS:,} tokens x {MIDTRAIN_EPOCHS} epochs")
    print(f"midtrain steps       {MIDTRAIN_STEPS}")
    print(f"  checkpoints        {dict(zip(MIDTRAIN_CHECKPOINT_TOKENS, MIDTRAIN_CHECKPOINT_STEPS))}")
    print(f"dolci tokens/step    {tokens_per_step(DOLCI_MICRO_BATCH, DOLCI_GRAD_ACCUM):,}")
    print(f"dolci steps          {DOLCI_STEPS}")
    print(f"  ckpts (control)    {dict(zip(DOLCI_CHECKPOINT_TOKENS_CONTROL, DOLCI_CHECKPOINT_STEPS_CONTROL))}")
    print(f"AFT steps            {AFT_STEPS} (evaluate at {AFT_EVAL_STEPS})")
    print(f"training legs        {N_MIDTRAIN_LEGS}")
    print(f"AFT runs             {N_AFT_RUNS}")
    print(f"eval endpoints       {N_EVAL_ENDPOINTS}")
