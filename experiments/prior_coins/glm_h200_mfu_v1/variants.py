"""The sweep matrix: where does GLM-4.5-Air's 7% MFU on 8xH200 actually go?

Baseline constants to beat (measured, glm_minimal_v1, 8xH200 SECURE $36.72/hr):
  midtrain  34.22 s/step @ 262,144 tok/update  (micro 2 x accum 2 x 8 x 8192)
  dolci    269.9  s/step @ 1,048,576 tok/update (micro 2 x accum 8 x 8 x 8192)

Both pinned stages set ``accelerator_config.gradient_accumulation_kwargs.
sync_each_batch: true``, so step time is

    t = accum x (F(micro) + RS) + O

-- one full ~220 GB gradient reduce-scatter *per micro-step*, not one per
optimizer step. The smoke stage's own comment says why: FSDP2 ``no_sync`` keeps
gradients unsharded between micro-batches (~221 GiB/rank, OOMed live at 137
GiB), so the campaign bought token count with accumulation and paid a
reduce-scatter each time. That makes two things measurable arithmetically,
with no profiler needed for the headline:

  A0 - A2  isolates the second reduce-scatter (same tokens/update, accum 2 -> 1)
  A1, A2, A3 traces F(micro) -- whether fatter micro-batches fix the MoE
             grouped_mm efficiency or just move the memory wall

And one arithmetic oddity worth its own cell. The two constants share
micro_batch_size=2 and sequence_len=8192, so per-micro-step they should agree:
   midtrain  34.22 / 2 = 17.1 s
   dolci    269.9  / 8 = 33.7 s
A 2x gap for the same shape on the same model. Dolci is 96 steps/arm and the
campaign's single largest cost line, so B0 (dolci at accum 1, directly
comparable to A1) is the highest-surprise cell in the sweep.

Cells are ordered by information-per-dollar: the anchor first, the expensive
confirmations last, so a truncated run still delivers a usable answer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MIDTRAIN_BASE = "midtrain_glm45_air_smoke"
DOLCI_BASE = "sft_dolci_dispatch_final_v1_glm45_air"

SEQ_LEN = 8192
N_GPUS = 8

#: Measured 8xH200 constants the sweep is anchored against.
H200_MIDTRAIN_S_PER_STEP = 34.22
H200_DOLCI_S_PER_STEP = 269.9
H200_POD_USD_PER_HR = 36.72
#: H200 SXM dense bf16 peak, as used by glm_minimal_v1/reconcile_cost.py.
H200_PEAK_BF16_TFLOPS = 989.0
#: GLM-4.5-Air active parameters per token (110B total, 12B active).
GLM_ACTIVE_PARAMS = 12.0e9

PLUGIN = "mfu_plugin.MfuTimerPlugin"


@dataclass(frozen=True)
class Variant:
    key: str
    family: str  # "midtrain" | "dolci"
    note: str
    micro: int
    accum: int
    warmup: int = 3
    timed: int = 5
    #: dotted-path overrides applied to the stage's ``axolotl`` block
    overrides: dict[str, Any] = field(default_factory=dict)
    profile: bool = False
    #: a cell we expect to OOM -- the failure IS the result, so the driver
    #: records it and moves on rather than aborting the sweep
    expect_oom: bool = False

    @property
    def tokens_per_update(self) -> int:
        return self.micro * self.accum * N_GPUS * SEQ_LEN

    @property
    def stage_name(self) -> str:
        return f"mfu_{self.key}"

    @property
    def base_stage(self) -> str:
        return MIDTRAIN_BASE if self.family == "midtrain" else DOLCI_BASE


SYNC_TRUE = {"accelerator_config.gradient_accumulation_kwargs.sync_each_batch": True}
SYNC_FALSE = {"accelerator_config.gradient_accumulation_kwargs.sync_each_batch": False}
# The smoke stage omits optim_args; the pinned campaign stages carry stochastic
# rounding, and it is part of the recipe under test.
STOCHASTIC = {"optim_args": "bf16_stochastic_round=True"}

VARIANTS: list[Variant] = [
    # -- anchor -------------------------------------------------------------
    Variant(
        key="A0_mid_m2a2_pinned", family="midtrain",
        note="pinned campaign geometry; must reproduce 34.22 s/step. Profiled.",
        micro=2, accum=2, overrides={**SYNC_TRUE, **STOCHASTIC}, profile=True,
    ),
    # -- the reduce-scatter decomposition -----------------------------------
    Variant(
        key="A1_mid_m2a1", family="midtrain",
        note="smoke geometry, 131,072 tok/update = F(2) + RS + O.",
        micro=2, accum=1, overrides={**STOCHASTIC},
    ),
    Variant(
        key="A2_mid_m4a1", family="midtrain",
        note="262,144 tok/update as A0 but ONE reduce-scatter. A0-A2 = the "
             "second RS. Primary candidate win.",
        micro=4, accum=1, overrides={**STOCHASTIC},
    ),
    # -- the dolci 2x mystery (largest campaign cost line) ------------------
    Variant(
        key="B0_dolci_m2a1", family="dolci",
        note="dolci at accum 1, directly comparable to A1. Tests the 17.1 vs "
             "33.7 s/micro-step gap between the two pinned stages.",
        micro=2, accum=1, overrides={**STOCHASTIC},
    ),
    # -- what the pins cost -------------------------------------------------
    Variant(
        key="C2_mid_m4a1_fused", family="midtrain",
        note="prices the adamw_torch_8bit + stochastic-rounding pin against "
             "adamw_torch_fused. A measurement, not a proposal: the 8-bit "
             "optimizer is a declared cross-model confound.",
        micro=4, accum=1,
        overrides={"optimizer": "adamw_torch_fused", "optim_args": None},
    ),
    Variant(
        key="C0_mid_m4a1_nockpt", family="midtrain",
        note="gradient checkpointing off: the recompute tax (~1/3 of forward "
             "FLOPs). May OOM at micro 4 on 141 GiB.",
        micro=4, accum=1,
        overrides={"gradient_checkpointing": False, **STOCHASTIC},
        expect_oom=True,
    ),
    # -- how far the fat-micro-batch win scales -----------------------------
    Variant(
        key="A3_mid_m8a1", family="midtrain",
        note="524,288 tok/update. Does fattening the expert GEMMs keep paying, "
             "or is 141 GiB the wall?",
        micro=8, accum=1, overrides={**STOCHASTIC}, expect_oom=True,
    ),
    Variant(
        key="B1_dolci_m4a1", family="dolci",
        note="dolci at the midtrain-winning geometry.",
        micro=4, accum=1, overrides={**STOCHASTIC},
    ),
    # -- falsification cells (reviewer findings, plan-review session) --------
    Variant(
        key="A4_mid_m3a1", family="midtrain",
        note="third micro-batch point at LOWER memory than micro 4: the "
             "curvature check that survives A3 OOMing. Under linearity the "
             "prediction is t(A1) + f exactly.",
        micro=3, accum=1, overrides={**STOCHASTIC},
    ),
    Variant(
        key="A5_mid_m2a3", family="midtrain",
        note="accum-only cell: tests that a step really is accum x (F + RS) + O "
             "(RS paid once per micro-step). Memory-flat in accum by the "
             "campaign's own premise, so this cannot OOM new ground. "
             "Complementary to A3/A4, not a substitute -- an accum cell is "
             "blind to micro-batch curvature by construction.",
        micro=2, accum=3, overrides={**SYNC_TRUE, **STOCHASTIC},
    ),
    # -- the trade the campaign declined ------------------------------------
    Variant(
        key="C1_mid_m2a2_nosync", family="midtrain",
        note="sync_each_batch off: real gradient accumulation. The smoke "
             "stage says this OOMs at 137 GiB mid-backward; this confirms it "
             "on the record and costs ~4 min to do so.",
        micro=2, accum=2, overrides={**SYNC_FALSE, **STOCHASTIC},
        expect_oom=True,
    ),
]


def apply_override(block: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path inside the axolotl block; ``None`` deletes the key."""
    keys = path.split(".")
    node = block
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    if value is None:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = value


def render_variant_stage(
    variant: Variant, base_stage_yaml: dict[str, Any], mfu_dir: Path,
) -> dict[str, Any]:
    """Clone a pinned stage into a short, no-save, instrumented variant."""
    stage = copy.deepcopy(base_stage_yaml)
    stage["name"] = variant.stage_name
    stage["description"] = f"MFU sweep cell {variant.key}: {variant.note}"
    # The pod block belongs to the campaign's launcher, not to this sweep, and
    # carrying it would let a stray runner try to create a pod.
    stage.pop("pod", None)
    ax = stage["axolotl"]

    ax["micro_batch_size"] = variant.micro
    ax["gradient_accumulation_steps"] = variant.accum
    total = variant.warmup + variant.timed
    ax["max_steps"] = total
    # Enough epochs that max_steps is always the binding constraint.
    ax["num_epochs"] = 50
    ax["warmup_steps"] = 1
    ax.pop("warmup_ratio", None)
    ax["save_strategy"] = "no"
    ax.pop("save_total_limit", None)
    # No saves at all: checkpoint_schedule would fire the campaign's writer.
    ax.pop("checkpoint_schedule", None)
    ax["logging_steps"] = 1

    for path, value in variant.overrides.items():
        apply_override(ax, path, value)

    plugins = list(ax.get("plugins") or [])
    # RouterHealthPlugin asserts a nonzero fp32 e_score_correction_bias; that is
    # a load-path check, not a throughput one, and its per-step expert-count
    # hooks are exactly the kind of overhead this sweep must not measure.
    plugins = [p for p in plugins if "RouterHealthPlugin" not in p]
    plugins = [p for p in plugins if "CheckpointSchedulePlugin" not in p]
    if PLUGIN not in plugins:
        plugins.append(PLUGIN)
    ax["plugins"] = plugins
    ax.pop("router_health_log_steps", None)

    ax["mfu_out_dir"] = str(mfu_dir)
    ax["mfu_warmup_steps"] = variant.warmup
    ax["mfu_profile_step"] = (variant.warmup + 1) if variant.profile else None
    return stage


def write_variant_stage(
    variant: Variant, stages_dir: Path, mfu_dir: Path,
) -> Path:
    base = yaml.safe_load((stages_dir / f"{variant.base_stage}.yaml").read_text())
    stage = render_variant_stage(variant, base, mfu_dir)
    out = stages_dir / f"{variant.stage_name}.yaml"
    out.write_text(yaml.safe_dump(stage, sort_keys=False))
    return out


def mfu_percent(seconds: float, tokens: int, *, checkpointing: bool = True) -> float:
    """Model FLOPs utilisation, matching reconcile_cost.py's convention.

    6 FLOPs/param/token for fwd+bwd, x4/3 when activations are recomputed.
    """
    flops = 6.0 * GLM_ACTIVE_PARAMS * tokens
    if checkpointing:
        flops *= 4.0 / 3.0
    per_gpu = flops / N_GPUS / seconds
    return 100.0 * per_gpu / (H200_PEAK_BF16_TFLOPS * 1e12)
