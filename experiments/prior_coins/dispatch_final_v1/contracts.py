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
from dataclasses import MISSING, dataclass, field
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
#: DATA_REPO itself is resolved from the profile below (Profile.data_repo):
#: the campaign repo hit its storage limit on 2026-09-08, so the 250M charter
#: release and its balanced-v2 AFT cells live in a second, public repo.
DEFAULT_DATA_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
RELEASE_MANIFEST_FILES = {
    "dispatch_v3_release_v1": "release_manifest.json",
    "dispatch_v3_release_v2_spec5_stratified": "release_manifest_v2.json",
    "dispatch_v3_release_v2_noex_qualitative": "release_manifest_v2_noex.json",
    # 250M-token charter cut (spec-5 + spec-6 blocks, re-stratified, exact
    # deduped): build_release_v3_charter250m.py, published 2026-09-08 to the
    # public repo at revision 262ce0d3 (see publish_receipt_charter_250m_v3.json).
    "dispatch_v3_release_v3_charter_250m_spec5plus6_stratified":
        "release_manifest_charter_250m_v3.json",
    # Matched-dose 125M splits of the 250M release by focus mode (no-example
    # study, 2026-09-10): build_release_v4_charter_split.py, published to the
    # public repo in one commit (publish_receipt_charter_125m_split_v4.json).
    "dispatch_v3_release_v4_charter_125m_noex_qualitative":
        "release_manifest_charter_125m_noex_v4.json",
    "dispatch_v3_release_v4_charter_125m_worked":
        "release_manifest_charter_125m_worked_v4.json",
    # Clause-asymmetric 190M charter cut: worked documents only for the five
    # AFT-trained clauses, the other seven stems topped up with same-stem
    # spec-6 qualitative to the same per-stem dose as glm45_air_190m.
    # clause_asym_190m_v1/build_release_clause_asym.py, published 2026-09-10
    # at revision a07f2e82 (see publish_receipt_charter_190m_clause_asym.json).
    "dispatch_v3_release_v5_charter_190m_clause_asym":
        "release_manifest_charter_190m_clause_asym.json",
}

FILLER_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
FILLER_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
FILLER_SHUFFLE_BUFFER = 10_000

#: Rows that already ran and whose profiles are historical records. They
#: predate arm stacking and must not be retro-fitted to it.
LEGACY_HUB_LAYOUT_PROFILES_FROZEN = frozenset({"gemma3_12b_50m"})

#: Stacked-row disk gates and the immutable container-disk request operators
#: must choose at pod creation.
#:
#: ENUMERATED, not pattern-built. A `for dose in ("1m", "5m", "50m")` sweep over
#: 12B looks right and is wrong in both directions: it captures
#: `gemma3_12b_50m`, which is the FROZEN as-run row (already completed at one
#: arm per pod, so it is not a stacked row and its measured 250 GB floor is a
#: record of what ran), and it misses `gemma3_12b_50m_4ep`, which IS one of the
#: nine campaign rows. Adding a row here means naming it.
STACKED_GEMMA_DISK_FLOORS_GB = {
    "gemma3_27b_5m": 750,
    "gemma3_27b_19m": 750,
    "gemma3_27b_50m": 750,
    "gemma3_27b_190m": 750,
    "gemma3_12b_1m": 300,
    "gemma3_12b_5m": 300,
    "gemma3_12b_19m": 300,
    "gemma3_12b_50m_4ep": 300,
    "gemma3_12b_50m_noex": 300,
    "gemma3_12b_50m_elic": 300,
    "gemma3_12b_50m_divresp": 300,
    "gemma3_4b_1m": 150,
    "gemma3_4b_5m": 150,
    "gemma3_4b_50m": 150,
    # GLM rows run ONE ARM PER POD (decided 2026-09-01: 4xH200 AFT cells fill
    # an 8-GPU pod with a single arm), but the floor stays a whole-chain gate
    # checked once at chain start, exactly like the stacked rows above. 1400
    # matches the profiles' min_free_disk_gb (221 GB bf16 base + unpacked
    # experts + midtrain/dolci full checkpoints + working space, envelope
    # from glm_minimal_v1).
    "glm45_air_5m": 1400,
    "glm45_air_50m": 1400,
    "glm45_air_190m": 1400,
    # 1B-presented charter row (2026-09-08): same one-arm-per-pod envelope;
    # the sniped 8xB200 pod is provisioned at 1600 like the H200 GLM pods.
    "glm45_air_1b": 1400,
    # Matched-dose 125M x 4 charter rows (no-example study, 2026-09-10): the
    # 1B row's recipe at half the unique dose, same one-arm-per-pod envelope.
    "glm45_air_500m_noex": 1400,
    "glm45_air_500m_worked": 1400,
    "glm45_air_190m_clause_asym": 1400,
}
assert LEGACY_HUB_LAYOUT_PROFILES_FROZEN.isdisjoint(STACKED_GEMMA_DISK_FLOORS_GB), (
    "a frozen as-run row must never carry a stacked-row floor: it ran one arm "
    "per pod and its profile is a historical record"
)
STACKED_GEMMA_PROVISIONED_DISK_GB = {
    "27b": 1200, "12b": 500, "4b": 250,
    # GLM-4.5-Air per-arm pods: 1600 GB provisioned against the 1400 GB floor.
    # Container disk cannot be grown after creation, and a GLM pod that passed
    # the >=1.8 TB host-RAM preflight sits on a scarce host -- never waste one
    # on an undersized disk.
    "air": 1600,
}

#: Dead-man's-switch budget per stacked row, in hours, passed to create-pod.sh
#: as --max-hours. The switch is a detached timer ON the pod that terminates it
#: regardless of what the workload does, and it is OFF unless asked for -- so
#: without this a pod the supervisor loses track of (crashed supervisor, dead
#: ssh) bills until a human notices. At 27B that is $36.72/hr.
#:
#: Values are the stacked-row wall clock from scaling_v1/cost_per_arm_v3.py
#: (three arms' training in sequence, AFT and eval pooled), times ~1.6 and
#: rounded up. It is a last resort against a FORGOTTEN pod, not a stall
#: detector -- the supervisor's own per-phase timeouts handle stalls, and a
#: firing switch is survivable precisely because stages publish as they land
#: and pod/rehydrate.py restores them onto a fresh pod.
STACKED_ROW_MAX_HOURS = {
    "gemma3_4b_1m": 16,          # ~9.6 h expected
    "gemma3_4b_5m": 16,          # ~10.0 h
    "gemma3_4b_50m": 24,         # ~14.6 h
    "gemma3_12b_1m": 20,         # ~12.5 h
    "gemma3_12b_5m": 22,         # ~12.9 h
    "gemma3_12b_19m": 24,        # ~14.5 h (interpolated 5m->50m_4ep by dose)
    "gemma3_12b_50m_4ep": 30,    # ~18.2 h
    "gemma3_12b_50m_noex": 20,   # ~12.1 h (2-arm row: the 4ep wall x 2/3)
    "gemma3_12b_50m_elic": 12,   # ~6.5 h (AFT+eval tail only; parents rehydrated)
    # 30 cells (10 per arm, 3 waves of 4 GPUs) x ~1.5 h at seq 1536, plus a
    # ~0.25 h eval per cell and bring-up: ~17 h stacked, x1.6 headroom.
    # Per-ARM unit at 6.2 h measured (3 waves of 4 AFT cells + battery),
    # NOT the 17.9 h stacked shape -- the queue ships one row per arm. 18 h
    # is ~2.9x headroom; the 28 h a stacked reading implies is >4x, which
    # the dead-man test rejects as protecting nothing: a hung 4xH100 pod
    # would bill $368 before self-terminating instead of $237.
    "gemma3_12b_50m_divresp": 18,
    "gemma3_27b_5m": 24,         # ~14.4 h
    "gemma3_27b_19m": 28,        # ~16.8 h (interpolated 5m->50m by dose)
    "gemma3_27b_50m": 36,        # ~22.2 h
    "gemma3_27b_190m": 75,       # ~46.4 h
    # GLM budgets are PER-ARM (one arm per pod, decided 2026-09-01). Derived
    # from glm_minimal_v1's measured constants (34.22 s/step midtrain,
    # tok_s-based dolci ~3.8 h, 2 AFT waves at 4 GPUs/cell, 0.42 h/endpoint
    # eval) plus ~2 h GLM bring-up (221 GB base + expert unpack + dual
    # venvs), times the family's ~1.6. AFT s/step (14, estimate-grade) and
    # the eval block are the residual uncertainty; the headroom absorbs it.
    # See scaling_v1/cost_per_arm_v3.py.
    "glm45_air_5m": 30,          # ~18 h expected (longest arm)
    "glm45_air_50m": 34,         # ~21 h
    "glm45_air_190m": 50,        # ~31 h
    # 1B-presented charter arm: 2B midtrain positions at the H200 anchor
    # (34.22 s / 262,144) is 72.5 h, plus Dolci/AFT/eval ~6 h and ~5 h
    # bring-up/publish = ~84 h; x1.6 headroom. On the 8xB200 pod the same
    # arm is projected at ~35 h training. The pod for this row is sniped by
    # hand and carries NO dead-man switch (user instruction 2026-09-08); the
    # figure exists so the ops scheduler's completeness assertion holds.
    "glm45_air_1b": 140,
    # 500M-presented charter arms (no-example study): half the 1B row's
    # midtrain positions -- 1B at the H200 anchor is 36 h, plus Dolci/AFT/eval
    # ~6 h and ~5 h bring-up/publish = ~47 h; x1.6 headroom. On 8xB200 the
    # same arm is projected at ~13 h training. Pods for these rows are sniped
    # by hand like the 1B row's; the figures exist so the ops scheduler's
    # completeness assertion holds.
    "glm45_air_500m_noex": 75,
    "glm45_air_500m_worked": 75,
    "glm45_air_190m_clause_asym": 40,
}
assert set(STACKED_ROW_MAX_HOURS) == set(STACKED_GEMMA_DISK_FLOORS_GB), (
    "every stacked row needs both a disk floor and a dead-man's-switch budget"
)

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
    # --- family-specific execution contract ------------------------------------
    # Defaults preserve every existing Gemma row byte-for-byte.  GLM profiles
    # state every value explicitly: its selection and schedule tokenizers are
    # intentionally different, and that difference changes the dose.
    family: str = "gemma3"
    document_selection_tokenizer: str = "unsloth/gemma-3-12b-pt"
    document_selection_tokenizer_revision: str = (
        "54ba4a26535408ddf5747cb9f7a5c16816659564")
    schedule_token_basis: str = "selection_tokenizer"
    schedule_tokenizer_revision: str | None = None
    expected_mix_tokens_by_arm: dict[str, int] = field(default_factory=dict)
    expected_mix_documents_by_arm: dict[str, int] = field(default_factory=dict)
    stage_midtrain_by_arm: dict[str, str] = field(default_factory=dict)
    dolci_global_batch_tokens: int = DOLCI_GLOBAL_BATCH_TOKENS
    aft_gpus_per_cell: int = 1
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_policy: str = "gemma_suffixes"
    eval_tensor_parallel_size: int = 1
    eval_main_gpu_memory_utilization: float = 0.60
    eval_shared_gpu_memory_utilization: float = 0.80
    eval_chat_template: str | None = None
    eval_stop_tokens: tuple[str, ...] = ()
    min_host_ram_gb: float = 0.0
    min_cgroup_ram_gb: float = 0.0
    min_gpu_memory_gib: float = 79.0
    require_idle_gpus: bool = False
    publish_midtrain_default: bool = True
    full_parameter_seed: int = SEED
    full_parameter_optimizer: str = "adamw_torch_fused"
    full_parameter_optim_args: str | None = None
    optimizer_cross_model_confound: bool = False
    # --- treatment rows that reuse a parent row's training ---------------------
    # A treatment profile (e.g. the response-side elicitation AFT) re-runs only
    # AFT + the eval batteries on ANOTHER row's published midtrain/dolci
    # checkpoints. parent_hub_profile names that row: rehydrate then reads the
    # pre-AFT stages from the parent's Hub prefix (and never republishes them),
    # while everything from AFT on publishes under THIS profile's own prefix.
    parent_hub_profile: str | None = None
    # Per-profile AFT data location; None keeps the campaign default pin
    # (releases/dispatch-final-v1, see AFT_DATA_PREFIX below). The manifest
    # file is committed next to contracts.py, like aft_manifest.json.
    aft_data_prefix: str | None = None
    aft_manifest_file: str = "aft_manifest.json"
    # Which dataset repo holds this row's release AND AFT cells (both are
    # fetched at data_revision, so they must live in the same repo). None =
    # DEFAULT_DATA_REPO, the campaign repo every historical row read from. The
    # 250M charter row declares the public overflow repo (2026-09-08).
    data_repo: str | None = None
    # Whether the midtrain stage runs RouterHealthPlugin. True for every
    # historical row (the family posture check demands the plugin). The 1B
    # charter row detaches it for midtrain only: the plugin's per-forward
    # bincount is a host sync that cost ~13% of wall clock on B200
    # (glm_b200_speed_v1 run-02, 2026-09-08). Dolci/AFT keep the monitor.
    midtrain_router_monitor: bool = True
    # Periodic RESUME (insurance) checkpoints during midtrain: every N
    # optimizer steps the checkpoint plugin also saves a full sharded
    # checkpoint (params + 8-bit AdamW state, ~440 GB for GLM-4.5-Air), keeps
    # the newest `keep_local` on disk, and pod/resume_upload.py ships the
    # newest complete one to the Hub, overwriting the previous. None (every
    # historical row) = scheduled saves only. Backup only: resuming FROM one
    # is not wired (decision 2026-09-08, 1B charter row).
    midtrain_resume_every_steps: int | None = None
    midtrain_resume_keep_local: int = 2
    # The arms this row can run. Every historical row is a three-arm grid
    # row; a single-arm row (the 250M charter cut has no coin corpus and no
    # control budget) names just the arms it has data for, and the per-arm GLM
    # pins below are validated against THIS set rather than the full grid.
    arms: tuple[str, ...] = ("charter", "coin", "control")
    # Where THIS row publishes. None = the campaign's main repo. A row family
    # that would push the main repo through the Hub's hard 20,000-file cap
    # declares its own repo here (the GLM rows and the diverse-response
    # treatment both do). Declaring it on the PROFILE rather than only in
    # ops/launch_unit.sh's environment is what makes the supervisor's
    # verify_hub teardown gate look in the same repo the pod published to:
    # an env-only override is set on the pod, never in the supervisor's own
    # process, so verify_hub counted zero files and every pod parked alive.
    hub_model_repo: str | None = None


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
    required = {
        f.name for f in dataclasses.fields(Profile)
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError(f"profile {name!r}: missing keys {missing}")
    data["midtrain_checkpoint_tokens"] = tuple(data["midtrain_checkpoint_tokens"])
    if "eval_stop_tokens" in data:
        data["eval_stop_tokens"] = tuple(data["eval_stop_tokens"])
    if "arms" in data:
        data["arms"] = tuple(data["arms"])
    profile = Profile(**data)
    _validate_profile(profile)
    return profile


def _validate_profile(p: Profile) -> None:
    if p.family not in {"gemma3", "glm45_air"}:
        raise ProfileError(
            f"profile {p.name!r}: family must be 'gemma3' or 'glm45_air', "
            f"got {p.family!r}"
        )
    if p.release_version not in RELEASE_MANIFEST_FILES:
        raise ProfileError(
            f"profile {p.name!r}: unknown release_version "
            f"{p.release_version!r}; known releases are "
            f"{sorted(RELEASE_MANIFEST_FILES)}"
        )
    for attribute in ("base_model_revision", "data_revision"):
        value = getattr(p, attribute)
        if attribute == "data_revision" and str(value).startswith("TODO_"):
            raise ProfileError(
                f"profile {p.name!r}: data_revision is still the release "
                "placeholder; publish v2 and pin its 40-hex commit SHA"
            )
        if not _HEX40.match(str(value)):
            raise ProfileError(
                f"profile {p.name!r}: {attribute}={value!r} is not a 40-hex commit "
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
    family_dolci_batch = {
        "gemma3": DOLCI_GLOBAL_BATCH_TOKENS,
        # PINS.md:128-131 and :353-356: same 100,663,296 positions, split
        # into 96 updates so the five-step warmup is meaningful.
        "glm45_air": 1_048_576,
    }[p.family]
    if p.dolci_global_batch_tokens != family_dolci_batch:
        raise ProfileError(
            f"profile {p.name!r}: family {p.family!r} requires Dolci batch "
            f"{family_dolci_batch:,}, got {p.dolci_global_batch_tokens:,}"
        )
    if dol != p.dolci_global_batch_tokens:
        raise ProfileError(
            f"profile {p.name!r}: Dolci geometry yields {dol:,} tokens/step, "
            f"not the shared/family-named {p.family} batch "
            f"{p.dolci_global_batch_tokens:,}"
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
    stacked_floor = STACKED_GEMMA_DISK_FLOORS_GB.get(p.name)
    if stacked_floor is not None and p.min_free_disk_gb != stacked_floor:
        raise ProfileError(
            f"profile {p.name!r}: stacked row requires min_free_disk_gb="
            f"{stacked_floor}, got {p.min_free_disk_gb}")
    if p.aft_gpus_per_cell < 1 or p.n_gpus % p.aft_gpus_per_cell:
        raise ProfileError(
            f"profile {p.name!r}: aft_gpus_per_cell must be a positive divisor "
            f"of n_gpus ({p.n_gpus}), got {p.aft_gpus_per_cell}"
        )
    if p.eval_tensor_parallel_size < 1 or p.n_gpus % p.eval_tensor_parallel_size:
        raise ProfileError(
            f"profile {p.name!r}: eval_tensor_parallel_size must be a positive "
            f"divisor of n_gpus ({p.n_gpus}), got {p.eval_tensor_parallel_size}"
        )
    for label, value in (
        ("eval_main_gpu_memory_utilization", p.eval_main_gpu_memory_utilization),
        ("eval_shared_gpu_memory_utilization", p.eval_shared_gpu_memory_utilization),
    ):
        if not 0 < value <= 1:
            raise ProfileError(f"profile {p.name!r}: {label} must be in (0, 1]")
    if p.lora_r < 1 or p.lora_alpha < 1 or not 0 <= p.lora_dropout < 1:
        raise ProfileError(f"profile {p.name!r}: invalid LoRA geometry")

    if p.midtrain_resume_every_steps is not None and (
            not isinstance(p.midtrain_resume_every_steps, int)
            or isinstance(p.midtrain_resume_every_steps, bool)
            or p.midtrain_resume_every_steps < 1):
        raise ProfileError(
            f"profile {p.name!r}: midtrain_resume_every_steps must be a positive "
            f"int or omitted, got {p.midtrain_resume_every_steps!r}")
    if (not isinstance(p.midtrain_resume_keep_local, int)
            or isinstance(p.midtrain_resume_keep_local, bool)
            or p.midtrain_resume_keep_local < 1):
        raise ProfileError(
            f"profile {p.name!r}: midtrain_resume_keep_local must be a positive "
            f"int, got {p.midtrain_resume_keep_local!r}")
    grid_arms = {"control", "charter", "coin"}
    if (not p.arms or len(set(p.arms)) != len(p.arms)
            or not set(p.arms) <= grid_arms):
        raise ProfileError(
            f"profile {p.name!r}: arms must be a non-empty, duplicate-free "
            f"subset of {sorted(grid_arms)}, got {list(p.arms)}")
    arm_names = set(p.arms)
    if p.family == "glm45_air":
        if p.schedule_token_basis != "model_tokenizer":
            raise ProfileError(
                f"profile {p.name!r}: glm45_air schedule_token_basis must be "
                "'model_tokenizer'"
            )
        if p.schedule_tokenizer_revision != p.base_model_revision:
            raise ProfileError(
                f"profile {p.name!r}: GLM schedule tokenizer revision must equal "
                "the pinned base-model revision"
            )
        for label, values in (
            ("expected_mix_tokens_by_arm", p.expected_mix_tokens_by_arm),
            ("expected_mix_documents_by_arm", p.expected_mix_documents_by_arm),
            ("stage_midtrain_by_arm", p.stage_midtrain_by_arm),
        ):
            if set(values) != arm_names:
                raise ProfileError(
                    f"profile {p.name!r}: {label} must name exactly "
                    f"{sorted(arm_names)}"
                )
        if p.document_selection_tokenizer != "unsloth/gemma-3-12b-pt":
            raise ProfileError(
                f"profile {p.name!r}: GLM document selection must retain the "
                "Gemma-3 counting tokenizer"
            )
        if p.aft_gpus_per_cell != 4:
            raise ProfileError(
                f"profile {p.name!r}: glm45_air requires the measured 4 GPUs/AFT "
                "cell (2 GPUs OOMed)"
            )
        if p.eval_tensor_parallel_size < 2:
            raise ProfileError(
                f"profile {p.name!r}: glm45_air serving requires TP >= 2"
            )
        if p.lora_target_policy != "glm45_attention_exact":
            raise ProfileError(
                f"profile {p.name!r}: GLM LoRA policy must be exact-path "
                "attention-only"
            )
        if (p.lora_r, p.lora_alpha, p.lora_dropout) != (64, 128, 0.0):
            raise ProfileError(
                f"profile {p.name!r}: GLM proven serving posture is "
                "r=64/alpha=128/dropout=0.0"
            )
        # 1800 restored 2026-09-02 (second time), and not as an interim: the
        # loader patch that bought the 1100 gate is WITHDRAWN. The unpatched
        # control (pod d3zgnaujisy20m, 2015 GB host) trained healthily on the
        # very stack that diverged patched — update 3 at loss 2.508 vs 81.3 —
        # so GLM rows run unpatched, every rank materializes the 221 GB model
        # (measured peak 1636 GB), and a row needs a >=1.8 TB host. Lower this
        # only once the patch's mechanism is isolated and fixed, not merely
        # because a load looks cheap.
        if (p.min_host_ram_gb, p.min_cgroup_ram_gb,
                p.min_gpu_memory_gib, p.require_idle_gpus) != (
                    1800.0, 1800.0, 140.0, True):
            raise ProfileError(
                f"profile {p.name!r}: GLM host gates must be 1800 GB host/cgroup "
                "(all-ranks loading, patch withdrawn; see comment above), "
                "140 GiB GPUs, and zero resident processes"
            )
        if p.full_parameter_seed != 314159:
            raise ProfileError(
                f"profile {p.name!r}: GLM midtrain/Dolci seed must be 314159"
            )
        if (p.full_parameter_optimizer, p.full_parameter_optim_args,
                p.optimizer_cross_model_confound) != (
                    "adamw_torch_8bit", "bf16_stochastic_round=True", True):
            raise ProfileError(
                f"profile {p.name!r}: GLM must explicitly declare the "
                "8-bit full-parameter optimizer confound and stochastic "
                "BF16 write-back posture")

    if p.parent_hub_profile is not None:
        # A treatment row rides another row's published midtrain/dolci
        # checkpoints: every field that determines those artifacts' identity
        # (checkpoint step numbers included) must MATCH the parent, or
        # rehydrate would validate checkpoint-<N> paths that cannot exist.
        if p.parent_hub_profile == p.name:
            raise ProfileError(f"profile {p.name!r}: parent_hub_profile is itself")
        parent_path = _profile_path(p.parent_hub_profile)
        if not parent_path.is_file():
            raise ProfileError(
                f"profile {p.name!r}: parent_hub_profile "
                f"{p.parent_hub_profile!r} is not a registered profile")
        parent = yaml.safe_load(parent_path.read_text())
        if parent.get("status") != "active":
            raise ProfileError(
                f"profile {p.name!r}: parent {p.parent_hub_profile!r} must be "
                "an active (runnable/ran) row")
        if parent.get("parent_hub_profile"):
            raise ProfileError(
                f"profile {p.name!r}: parent {p.parent_hub_profile!r} is itself "
                "a treatment row; chain-of-parents is not supported")
        must_match = (
            "scimt_model", "base_model", "base_model_revision", "tokenizer",
            "n_gpus", "sequence_len", "midtrain_micro_batch",
            "midtrain_grad_accum", "dolci_micro_batch", "dolci_grad_accum",
            "release_tokens_per_arm", "midtrain_tokens", "midtrain_epochs",
            "filler_token_budget", "dolci_tokens", "dolci_steps_target",
            "dolci_checkpoint_step_control",
        )
        for key in must_match:
            ours, theirs = getattr(p, key), parent.get(key)
            if key == "midtrain_epochs" and theirs is None:
                theirs = 1  # the frozen as-run row predates the field
            if ours != theirs:
                raise ProfileError(
                    f"profile {p.name!r}: {key}={ours!r} != parent "
                    f"{p.parent_hub_profile!r} {key}={theirs!r} -- a treatment "
                    "row must ride the parent's exact training identity")
        if tuple(p.midtrain_checkpoint_tokens) != tuple(
                parent.get("midtrain_checkpoint_tokens", ())):
            raise ProfileError(
                f"profile {p.name!r}: midtrain_checkpoint_tokens must equal "
                "the parent's (checkpoint step numbers are derived from them)")
    if p.aft_manifest_file != "aft_manifest.json" and p.aft_data_prefix is None:
        raise ProfileError(
            f"profile {p.name!r}: a bespoke aft_manifest_file without a "
            "bespoke aft_data_prefix would verify default-pin bytes against "
            "the wrong manifest")
    # ...and the mirror image, which is the one that actually bit us. A bespoke
    # prefix holds bespoke cells, so verifying them against the DEFAULT manifest
    # compares the right bytes to the wrong pins. glm45_air_190m_clause_asym
    # shipped with the prefix and no manifest and died in fetch_aft_cells on
    # mixed_charter (2026-09-12) -- after 12.9 h of midtrain, because AFT is the
    # first phase that reads these files. Fail at profile load instead.
    if p.aft_data_prefix is not None and p.aft_manifest_file == "aft_manifest.json":
        raise ProfileError(
            f"profile {p.name!r}: a bespoke aft_data_prefix ({p.aft_data_prefix!r}) "
            "with the default aft_manifest.json would verify that prefix's bytes "
            "against the default prefix's pins; name the manifest its build "
            "published")


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
MODEL_FAMILY = PROFILE.family
DOCUMENT_SELECTION_TOKENIZER = PROFILE.document_selection_tokenizer
DOCUMENT_SELECTION_TOKENIZER_REVISION = (
    PROFILE.document_selection_tokenizer_revision)
SCHEDULE_TOKEN_BASIS = PROFILE.schedule_token_basis
SCHEDULE_TOKENIZER = PROFILE.tokenizer
SCHEDULE_TOKENIZER_REVISION = (
    PROFILE.schedule_tokenizer_revision or PROFILE.base_model_revision)
EXPECTED_MIX_TOKENS_BY_ARM = dict(PROFILE.expected_mix_tokens_by_arm)
EXPECTED_MIX_DOCUMENTS_BY_ARM = dict(PROFILE.expected_mix_documents_by_arm)

#: The dataset repo this row reads (release + AFT cells); see Profile.data_repo.
DATA_REPO = PROFILE.data_repo or DEFAULT_DATA_REPO
#: The arms this row can run; chain.parse_arms refuses anything else.
PROFILE_ARMS = tuple(PROFILE.arms)
#: Whether the midtrain stage must carry RouterHealthPlugin (posture check)
#: and produce router_health.jsonl (require_router_health / consolidation).
MIDTRAIN_ROUTER_MONITOR = PROFILE.midtrain_router_monitor
#: Insurance-checkpoint cadence for midtrain (None = off, the historical rows).
MIDTRAIN_RESUME_EVERY_STEPS = PROFILE.midtrain_resume_every_steps
MIDTRAIN_RESUME_KEEP_LOCAL = PROFILE.midtrain_resume_keep_local
RELEASE_VERSION = PROFILE.release_version
DATA_PREFIX = PROFILE.data_prefix
#: pinned so every arm consumes byte-identical inputs even if the repo moves
DATA_REVISION = PROFILE.data_revision
#: The four AFT cells live under the v1 release prefix at DATA_REVISION -- the
#: v2 re-release republished only release/ (the corpora) and never carried an
#: aft/ tree, which 404'd every arm's AFT phase on 2026-09-01 once the first
#: v2-profile rows reached it. The cells are release-version-independent by
#: construction (frozen by aft_manifest.json: fetch_aft_cells verifies each
#: file's sha256 against the manifest committed next to this file, and the
#: v1-path bytes were re-verified to match it before this pin was added), so
#: this is a location pin, not a data change.
AFT_DATA_PREFIX = PROFILE.aft_data_prefix or "releases/dispatch-final-v1"
#: sha256 source for the fetched cells; bespoke AFT data (e.g. the
#: response-side elicitation cells) pins its own committed manifest.
AFT_MANIFEST_FILE = PROFILE.aft_manifest_file
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


def midtrain_stage(arm: str) -> str:
    """Reviewed stage for one arm; GLM schedules differ after retokenizing."""
    if arm not in {"control", "charter", "coin"}:
        raise ValueError(f"unknown arm {arm!r}")
    return PROFILE.stage_midtrain_by_arm.get(arm, PROFILE.stage_midtrain)

MIN_FREE_DISK_GB = PROFILE.min_free_disk_gb
MIN_HOST_RAM_GB = PROFILE.min_host_ram_gb
MIN_CGROUP_RAM_GB = PROFILE.min_cgroup_ram_gb
MIN_GPU_MEMORY_GIB = PROFILE.min_gpu_memory_gib
REQUIRE_IDLE_GPUS = PROFILE.require_idle_gpus
PUBLISH_MIDTRAIN_DEFAULT = PROFILE.publish_midtrain_default
FULL_PARAMETER_SEED = PROFILE.full_parameter_seed
FULL_PARAMETER_OPTIMIZER = PROFILE.full_parameter_optimizer
FULL_PARAMETER_OPTIM_ARGS = PROFILE.full_parameter_optim_args
OPTIMIZER_CROSS_MODEL_CONFOUND = PROFILE.optimizer_cross_model_confound

# ----------------------------------------------------------------- geometry

SEQUENCE_LEN = PROFILE.sequence_len
N_GPUS = PROFILE.n_gpus

MIDTRAIN_MICRO_BATCH = PROFILE.midtrain_micro_batch
MIDTRAIN_GRAD_ACCUM = PROFILE.midtrain_grad_accum
DOLCI_MICRO_BATCH = PROFILE.dolci_micro_batch
DOLCI_GRAD_ACCUM = PROFILE.dolci_grad_accum
AFT_GPUS_PER_CELL = PROFILE.aft_gpus_per_cell

LORA_R = PROFILE.lora_r
LORA_ALPHA = PROFILE.lora_alpha
LORA_DROPOUT = PROFILE.lora_dropout
LORA_TARGET_POLICY = PROFILE.lora_target_policy

EVAL_TENSOR_PARALLEL_SIZE = PROFILE.eval_tensor_parallel_size
EVAL_MAIN_GPU_MEMORY = PROFILE.eval_main_gpu_memory_utilization
EVAL_SHARED_GPU_MEMORY = PROFILE.eval_shared_gpu_memory_utilization
EVAL_CHAT_TEMPLATE = PROFILE.eval_chat_template
EVAL_STOP_TOKENS = PROFILE.eval_stop_tokens


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


def expected_midtrain_steps(arm: str) -> int:
    """Frozen optimizer-step expectation for one arm's schedule token basis."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    unique_tokens = EXPECTED_MIX_TOKENS_BY_ARM.get(arm, MIDTRAIN_TOKENS)
    return steps_for(
        unique_tokens * MIDTRAIN_EPOCHS,
        MIDTRAIN_MICRO_BATCH,
        MIDTRAIN_GRAD_ACCUM,
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

#: Canonical cross-arm execution order.  ``ARMS`` remains the historical
#: mapping (including its insertion order) because single-arm consumers read
#: it directly; pooled schedulers use this explicit order, matching the
#: completed three-arm GLM campaign.
ARM_ORDER = ("charter", "coin", "control")

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
#: GLM evaluates the FINAL step only. Its AFT *intermediate* checkpoints are
#: written as FSDP shards (`pytorch_model_fsdp_0/`, `optimizer_0/`) with no PEFT
#: adapter beside them -- `consolidate_glm_checkpoint` runs for midtrain
#: (chain.py:944) and dolci (chain.py:1013) but never for AFT -- so
#: `checkpoint-256` has no `adapter_config.json` and eval cannot load it. Only
#: the final adapter exists, at `<cell>/checkpoints/adapter_config.json`, and
#: with `max_steps: 512` that adapter IS step 512.
#:
#: This blocked glm45_air_190m/charter at 08:54Z 2026-09-03 and would have
#: blocked coin and control identically. Sid chose step-512-only for the family
#: rather than hold all three arms for a checkpoint-conversion fix; converting
#: the intermediates stays the real fix if the mid-AFT point is wanted later.
#: gemma is untouched and keeps both steps, so the ten completed rows continue
#: to describe themselves correctly.
AFT_EVAL_STEPS = (512,) if MODEL_FAMILY == "glm45_air" else (256, 512)


def aft_cell_keys() -> tuple[tuple[str, str], ...]:
    """Every ``(arm, cell)`` AFT job, in canonical execution order.

    This is the single enumeration pooled scheduling and result checks share,
    so adding or reordering a cell cannot make the two silently drift apart.
    """
    return tuple((arm, cell) for arm in ARM_ORDER for cell in AFT_CELLS)

# --------------------------------------------------------------------- eval

#: template_diversity_v1 publishes 6 slices x 3 presentation modes.
EVAL_SLICES = (
    "eval_trained_agreement", "eval_trained_conflict",
    "eval_holdout_agreement", "eval_holdout_conflict",
    "eval_trained_adjacent", "eval_holdout_adjacent",
)
EVAL_SURFACES = ("canonical", "trained", "heldout")


def aft_adapter_dir(aft_run: Path, step: int) -> Path:
    """Servable LoRA adapter for one AFT step of one cell.

    ``checkpoint-<step>/`` is where every consumer looked, and for the gemma
    rows that is right: their LoRA runs write a full adapter at each saved
    step. The GLM rows train LoRA under FSDP, and there axolotl saves the
    intermediate steps as SHARDED TRAINER STATE (``pytorch_model_fsdp_0/``,
    optimizer, RNG) with no ``adapter_config.json`` at all. The only servable
    adapter such a run produces is the final one, written to the run root when
    training ends -- so under FSDP the root IS the ``AFT_STEPS`` adapter and no
    earlier step is servable without consolidating its shards.

    Resolution is by inspection, not by family: a stepped directory that has an
    adapter wins, so gemma is untouched. The root is accepted only for the
    final step -- serving it as step 256 would silently label the 512-step
    adapter with the wrong dose.
    """
    stepped = aft_run / "checkpoints" / f"checkpoint-{step}"
    if (stepped / "adapter_config.json").is_file():
        return stepped
    root = aft_run / "checkpoints"
    if step == AFT_STEPS and (root / "adapter_config.json").is_file():
        return root
    raise FileNotFoundError(
        f"no servable adapter for step {step} of {aft_run}: neither "
        f"{stepped}/adapter_config.json nor (for the final step) "
        f"{root}/adapter_config.json exists")


def eval_endpoint_names() -> tuple[str, ...]:
    """Every sampled endpoint, i.e. every response directory under ``eval/``."""
    return ("pre_aft",) + tuple(
        f"{cell}-step{step}"
        for cell in AFT_CELLS for step in AFT_EVAL_STEPS)


def expected_response_files() -> int:
    """Response ``*__*.jsonl`` files one arm's eval must produce.

    ``eval_sharded.sh`` hardcoded 162 -- correct only for a profile with two
    AFT eval steps. GLM evaluates step 512 alone (``AFT_EVAL_STEPS``), so the
    literal silently became wrong, and the one place it is read is the
    *tolerance* for a worker killed in vLLM engine teardown: with a stale
    count that tolerance can never fire, and a completed eval is reported as a
    failed one. Derived here so the shard script and ``phase_eval`` cannot
    drift.
    """
    return (len(eval_endpoint_names())
            * len(EVAL_SLICES) * len(EVAL_SURFACES))


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
#: v2 of the sweep (``build_costsweep_v2_prompts.py``) keeps every number above
#: -- bins, centres, n-per-bin, held-out template surface -- and changes only
#: WHERE the episodes come from: ``dispatch_v4.sample_record``, the sampler the
#: eval battery itself uses, instead of ``motivation_eval_v1``'s sdf design.
#: A fresh seed so the two draws are independent and can be compared as two
#: samples of the same design rather than one re-render of the other.
COSTSWEEP_V2_SEED = 20260911
#: Sampled into ``costsweep_v2/`` beside ``costsweep/`` so a re-run never
#: overwrites the as-run v1 responses; both are published, both are scored.
COSTSWEEP_V2_DIRNAME = "costsweep_v2"
#: The re-run does not repeat all nine endpoints. The sweep's question is how
#: the installed prior trades off against price, and the two AFT cells that
#: answer it are the clean one and the adversarial one:
#:   * ``agreement`` -- agreement-only AFT, i.e. the cell that never shows the
#:     model a conflict label, on every arm;
#:   * ``mixed_coin`` -- 2% coin-labelled conflict rows, added on the CHARTER
#:     arms only, where a contaminated dose is the thing under test.
#: ``pre_aft`` is dropped: an un-AFT'd base model does not emit the answer
#: format, and v1's pre_aft rows are dominated by "other".
COSTSWEEP_V2_CELLS = ("agreement",)
COSTSWEEP_V2_CHARTER_EXTRA_CELLS = ("mixed_coin",)


def costsweep_v2_endpoints(arm: str) -> tuple[str, ...]:
    """Endpoint names this arm's v2 sweep samples, in execution order."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    cells = COSTSWEEP_V2_CELLS + (
        COSTSWEEP_V2_CHARTER_EXTRA_CELLS if arm == "charter" else ())
    return tuple(
        f"{cell}-step{step}" for cell in cells for step in AFT_EVAL_STEPS)


#: The held-out extension of the v2 sweep (Sid, 2026-09-14): the same bins,
#: centres, n-per-bin and held-out template surface, one BUILD PER HELD-OUT
#: CLAUSE so each clause's curve carries the full 256 per bin. Each build has
#: its own battery directory (beside ``costsweep_v2/``, never over it), its own
#: episode-id prefix and its own seed, so responses to the three sweeps can
#: never be confused for one another.
COSTSWEEP_V2_HELDOUT_BATTERIES = {
    "qual_weekly_limit": "costsweep_v2_weekly",
    "precedence_deferrals": "costsweep_v2_deferrals",
}
COSTSWEEP_V2_HELDOUT_SEEDS = {
    "qual_weekly_limit": 20260914,
    "precedence_deferrals": 20260915,
}
COSTSWEEP_MAX_NEW_TOKENS = 64
#: From the profile, like every other engine in this pipeline. This was the
#: literal 0.80, which is what every gemma profile asks for anyway -- and
#: silently wrong for GLM, whose profile asks 0.92. At 0.80 of a 2x141 GB TP
#: group the costsweep engine reported
#:
#:     Model loading took 109.86 GiB
#:     Available KV cache memory: -8.52 GiB
#:
#: i.e. the weights alone overran the budget, and all five endpoints died with
#: "No available memory for the cache blocks". Recall hit the same shape from
#: its own hardcoded 0.60. Sourcing it here is a no-op for gemma (every gemma
#: profile is 0.80) and the fix for GLM.
COSTSWEEP_GPU_MEMORY = EVAL_SHARED_GPU_MEMORY


EVAL_ENDPOINTS_PER_ARM = (
    "pre_aft",
    *(f"{cell}/step{step}" for cell in AFT_CELLS for step in AFT_EVAL_STEPS),
)


def eval_endpoint_keys() -> tuple[tuple[str, str], ...]:
    """Every ``(arm, endpoint)`` evaluation result, 27 in total."""
    return tuple(
        (arm, endpoint)
        for arm in ARM_ORDER
        for endpoint in EVAL_ENDPOINTS_PER_ARM
    )


def eval_endpoints() -> tuple[tuple[str, str], ...]:
    """Backward-compatible name for :func:`eval_endpoint_keys`."""
    return eval_endpoint_keys()


N_MIDTRAIN_LEGS = len(ARMS) * 2
N_AFT_RUNS = len(aft_cell_keys())
N_EVAL_ENDPOINTS = len(eval_endpoint_keys())


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


#: The campaign's main results repo. Three repos exist; HUB_LAYOUT.md is the
#: map. A profile may name its own via Profile.hub_model_repo.
DEFAULT_MODEL_REPO = "arcadia-impact/scimt-dispatch-final-v1"


def model_repo_for(profile_name: str) -> str:
    """Which Hub repo a profile publishes to, resolved from the profile YAML.

    Deliberately reads the YAML rather than requiring a loaded Profile: the
    supervisor's verify_hub runs off-pod, where FINAL_V1_PROFILE names some
    other row (or nothing), and a placeholder profile refuses to load at all.
    Unknown names fall back to the main repo, which is where every row
    without an explicit declaration publishes.
    """
    path = _profile_path(profile_name)
    if not path.is_file():
        return DEFAULT_MODEL_REPO
    body = yaml.safe_load(path.read_text()) or {}
    return str(body.get("hub_model_repo") or DEFAULT_MODEL_REPO)


#: Set when this row is a treatment on another row's training (see
#: Profile.parent_hub_profile). The stages below are the parent's artifacts:
#: rehydrate READS them from the parent's prefix and they are never
#: republished under this row.
PARENT_HUB_PROFILE = PROFILE.parent_hub_profile
PARENT_OWNED_STAGES = ("data", "midtrain", "dolci")


def hub_stage_read_prefix(arm: str, stage: str) -> str:
    """Where a stage's bytes are READ from (writes always use
    hub_arm_prefix): pre-AFT stages of a treatment row live under its
    parent's prefix."""
    if PARENT_HUB_PROFILE and stage in PARENT_OWNED_STAGES:
        if PARENT_HUB_PROFILE in LEGACY_HUB_LAYOUT_PROFILES:
            return arm
        return f"{PARENT_HUB_PROFILE}/{arm}"
    return hub_arm_prefix(arm)


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
    identity = {
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
    if MODEL_FAMILY == "glm45_air":
        identity.update({
            "document_selection_tokenizer": DOCUMENT_SELECTION_TOKENIZER,
            "document_selection_tokenizer_revision": (
                DOCUMENT_SELECTION_TOKENIZER_REVISION),
            "schedule_token_basis": SCHEDULE_TOKEN_BASIS,
            "schedule_tokenizer": SCHEDULE_TOKENIZER,
            "schedule_tokenizer_revision": SCHEDULE_TOKENIZER_REVISION,
            "expected_mix_tokens": EXPECTED_MIX_TOKENS_BY_ARM[arm],
            "expected_mix_documents": EXPECTED_MIX_DOCUMENTS_BY_ARM[arm],
            "expected_midtrain_steps": expected_midtrain_steps(arm),
            "midtrain_stage": midtrain_stage(arm),
            "aft_gpus_per_cell": AFT_GPUS_PER_CELL,
            "lora_target_policy": LORA_TARGET_POLICY,
            "eval_tensor_parallel_size": EVAL_TENSOR_PARALLEL_SIZE,
            "full_parameter_seed": FULL_PARAMETER_SEED,
            "full_parameter_optimizer": FULL_PARAMETER_OPTIMIZER,
            "full_parameter_optim_args": FULL_PARAMETER_OPTIM_ARGS,
            "optimizer_cross_model_confound": OPTIMIZER_CROSS_MODEL_CONFOUND,
        })
    return identity


def validate() -> None:
    """Refuse a schedule that cannot do what it says. Called by the preflight."""
    if len(ARM_ORDER) != len(set(ARM_ORDER)) or set(ARM_ORDER) != set(ARMS):
        raise ValueError("ARM_ORDER must name every arm exactly once")
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
    if MODEL_FAMILY == "gemma3":
        if MIDTRAIN_CHECKPOINT_STEPS[-1] != MIDTRAIN_STEPS:
            raise ValueError("the last midtrain checkpoint must be the final step")
    else:
        for arm in ARMS:
            if expected_midtrain_steps(arm) < 1:
                raise ValueError(f"{arm}: GLM-tokenized mix yields no steps")
    if DOLCI_CHECKPOINT_STEPS_CONTROL[-1] != DOLCI_STEPS:
        raise ValueError("the last Dolci checkpoint must be the final step")

    if N_EVAL_ENDPOINTS != len(ARMS) + N_AFT_RUNS * len(AFT_EVAL_STEPS):
        raise ValueError("eval endpoint count disagrees with the grid")

    # PINS.md:376-383. Publishing all three ~199 GB midtrain parents leaves
    # 1,393 GB at peak against a 1,400 GB floor, with no useful safety margin.
    # GLM therefore defaults to reclaiming each parent after its last local
    # consumer. The override remains explicit and is never silently enabled.
    if MODEL_FAMILY == "glm45_air":
        retained_peak_gb = 1_393
        if retained_peak_gb >= MIN_FREE_DISK_GB:
            raise ValueError(
                "GLM three-arm retained-midtrain peak must stay below the disk floor"
            )
        # A single-arm GLM row (the 1B charter cut) has one ~200 GB parent
        # against the same floor and may publish it (decision 2026-09-08); the
        # three-arm rows may not.
        if PROFILE.publish_midtrain_default and len(PROFILE_ARMS) > 1:
            raise ValueError(
                "GLM multi-arm rows must leave midtrain publishing off by default")

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
