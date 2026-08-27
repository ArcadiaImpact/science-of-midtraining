"""Cost / wall-clock model for the scaling_v1 grid (model size x midtraining dose).

Usage:
    python cost_model.py            # prints the default scenario + variant one-liners

Config-first, no CLI flags (repo convention): edit the dataclass instances in the
SCENARIOS block at the bottom, or derive a new scenario with dataclasses.replace().

Every throughput / price constant is annotated:
  MEASURED  - traced to an as-run number in this repo (source cited)
  EST       - MFU-model estimate, calibrated against the nearest MEASURED anchor
  GUESS     - no anchor; smoke-test before trusting (GLM-4.5-Air especially)

The FLOPs model is tokens x 6 x N_active; effective throughput is
GPU peak BF16 x MFU, with a 1.5%/doubling multi-GPU penalty
(MEASURED: graft-dose v1, 4xH100 = 3.88x of 1xH100).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------------- GPUs


@dataclass(frozen=True)
class Gpu:
    name: str
    usd_hr: float  # $/GPU/hr, RunPod secure cloud on-demand
    tflops_bf16: float  # dense peak
    vram_gb: float


GPUS: dict[str, Gpu] = {
    # Rates: RunPod secure, live-pulled 2026-08-25 (graft_dose_v1 plan.py GPU_PRICES).
    # Refresh with .claude/skills/runpod-spinup/gpu-prices.sh before committing a budget.
    "H100": Gpu("H100 80GB SXM", 3.29, 989, 80),  # MEASURED (paid)
    "H200": Gpu("H200 141GB SXM", 4.59, 989, 141),  # MEASURED (27B scale-up pods)
    "A100": Gpu("A100 80GB SXM", 1.59, 312, 80),  # MEASURED (template_diversity pods)
    # B200/B300 rates not yet purchased by us -- EDIT when quoted
    "B200": Gpu("B200 180GB", 5.98, 2250, 180),
    "B300": Gpu("B300 288GB", 7.98, 2250, 288),  # BF16 peak ~= B200 (Ultra boosts FP4)
}

SECONDS_PER_HOUR = 3600.0
MULTI_GPU_EFF_PER_DOUBLING = 0.985  # MEASURED 3.88x on 4 GPUs => ~0.985/doubling


def scaling_eff(n_gpus: int) -> float:
    return MULTI_GPU_EFF_PER_DOUBLING ** math.log2(max(n_gpus, 1))


# --------------------------------------------------------------------------- stage configs


@dataclass(frozen=True)
class SdfStage:
    """Full-parameter midtraining on synth docs + Dolmino replay."""

    doses_mtok: tuple[float, ...] = (0.5, 1.6, 5.0, 16.0, 50.0)  # UNIQUE task tokens/arm
    corpora: tuple[str, ...] = ("charter", "coin")
    replay_ratio: float = 1.0  # Dolmino tokens per task token (1:1 convention)
    presentations: int = 4  # epochs over the mix (wave recipe; see hparams plan sec 2)
    # 'dose_proportional': unique mix = dose*(1+replay); steps scale with dose (current convention)
    # 'topup': Dolmino tops every cell up to topup_unique_mtok; steps constant across doses
    #          (token-scaling-law / Gate-2 convention; removes the steps-per-dose confound)
    mix: str = "dose_proportional"
    topup_unique_mtok: float = 100.0
    # dolmino-only control arm ("100% dolmino, say 16M"): total UNIQUE tokens trained on.
    # 32 = matched to the 16M-dose arm's unique mix (16M task + 16M replay). Set 16.0 if
    # you meant 16M total. Ignored under mix='topup' (control = pure-Dolmino top-up cell).
    control_unique_mtok: float = 32.0
    tokens_per_update: int = 262_144  # invariant: 32 seqs x 8192 (MIDTRAIN_SCHEDULE.md)

    def arms(self) -> list[tuple[str, float]]:
        """(label, unique mix tokens in M) per midtrain arm, incl. control."""
        out = []
        for corpus in self.corpora:
            for dose in self.doses_mtok:
                if self.mix == "topup":
                    unique = self.topup_unique_mtok
                else:
                    unique = dose * (1.0 + self.replay_ratio)
                out.append((f"{corpus}-{dose:g}M", unique))
        ctl = self.topup_unique_mtok if self.mix == "topup" else self.control_unique_mtok
        out.append(("dolmino-ctl", ctl))
        return out

    def trained_tokens_mtok(self, unique_mtok: float) -> float:
        return unique_mtok * self.presentations


@dataclass(frozen=True)
class IftStage:
    """Dolci-Instruct SFT after midtraining."""

    tokens_mtok: float = 500.0  # packed positions presented (was 100M in prior studies)
    tokens_per_update: int = 2_097_152  # invariant: 256 seqs x 8192 packed


@dataclass(frozen=True)
class AftStage:
    """LoRA agreement / conflict-mixture fine-tuning (wave recipe, PR 527 templates)."""

    rows: int = 8_192
    epochs: int = 2  # => 512 steps at global batch 32
    global_batch: int = 32
    conflict_mixtures: int = 4  # {0.2%, 2%} x {->charter, ->coin}
    conflict_substrates: int = 3  # charter, coin, control at the reference dose


@dataclass(frozen=True)
class EvalCfg:
    # wave-v2 battery: 2,000/2,000/800/800/1,000/400 slices = 7,000 prompts/endpoint,
    # trained + held-out slices INCLUDED. Grows if the PR527 3-mode presentation is adopted.
    prompts_per_endpoint: int = 7_000
    prefill_tokens: float = 800.0  # ~2,419 chars/prompt MEASURED; ~8-16 decode tokens (ignored)
    endpoints_per_aft_run: int = 6  # baseline + steps 32/64/128/256/512 (wave protocol;
    #                                  step-256 sign inversions make trajectories load-bearing)
    endpoints_per_midtrain_arm: int = 1  # post-midtrain sanity eval
    mfu_eval: float = 0.42  # calibrates to MEASURED 5.2-5.75 min/endpoint (12B, 1xH100,
    fixed_min_per_endpoint: float = 1.0  # native LoRA) and ~12.9 min (27B, 1xH200)


@dataclass(frozen=True)
class Overheads:
    pod_setup_hr: float = 0.33  # MEASURED ~3.8 min provision + env; keep slack for apt/torch
    net_gbyte_s: float = 0.5  # MEASURED: 800 MB/s fetch, ~520 MB/s concurrent upload (27B)
    ckpts_published_per_train_job: float = 2.0  # sampler ckpts pushed per stage (bf16 each)
    # NOTE: publishing FULL-STATE trajectories (5 x 209 GB at 27B) is a policy choice
    # costed at ~$21/arm-stage extra (measured); not in this model by default.


@dataclass(frozen=True)
class DataGen:
    corpus_needed_mtok_per_arm: float = 50.0  # must cover max dose
    existing_mtok_per_arm: float = 9.0  # MEASURED: v1+v2 dispatch corpora, 9.0M/arm deduped
    # $/M released tokens. Docgen v3 pilot MEASURED ~$12/M; the v1/v2 dispatch-contract
    # pipeline MEASURED $52-66/M interactive, ~$40-45/M batched. 15 assumes v3 rates hold
    # at scale -- if they don't, this line item grows ~4x (see scenario).
    usd_per_mtok: float = 15.0
    overgen_factor: float = 1.15  # filtering / dedup losses


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelPlan:
    key: str
    hf_id: str
    params_b: float  # total
    active_b: float  # == params_b for dense
    ckpt_gb: float  # bf16 sampler checkpoint
    # -- training pods ---------------------------------------------------------
    train_gpu: str  # GPU type for midtrain + IFT
    n_train_gpus: int
    aft_gpu: str
    n_aft_gpus: int
    eval_gpu: str
    n_eval_gpus: int
    # -- throughput (tokens/sec/GPU); None -> derived from MFU ------------------
    tok_s_gpu_midtrain: float | None = None
    tok_s_gpu_ift: float | None = None
    mfu_midtrain: float = 0.18  # micro-batch 1, GA, grad-ckpt, FA2 (EST band 0.16-0.21)
    mfu_ift: float = 0.28  # packed micro 4-8 (MEASURED 27B: 0.284)
    aft_s_per_step: float = 7.0  # wave recipe s/step on 1 AFT GPU (per-model MEASURED below)
    consolidate_hr: float = 0.0  # per train-job stage-end: DCP merge + verify-load + egress
    #                              (big-MoE only; the whole pod idles while rank 0 works)
    # -- grid participation ------------------------------------------------------
    doses_mtok: tuple[float, ...] | None = None  # None -> SdfStage grid
    gets_conflict_aft: bool = False
    chain_seeds: int = 1  # replicate the WHOLE chain (midtrain->ift->aft->eval) n times
    # -- graft pipeline (LoRA SDF on the PT donor, merged onto the public IT model)
    graft_gpu: str = "H100"
    n_graft_gpus: int = 4  # near-linear to 4 (MEASURED 3.96x); raises wall-clock only
    # LoRA-SDF tokens/sec/GPU under the as-measured graft recipe (sdpa, no Liger,
    # micro 1). 12B MEASURED 187 s/step 1xH100 -> ~1,390; others scaled by FLOPs.
    tok_s_gpu_sdf_lora: float = 1_390.0

    def flops_per_token(self) -> float:
        return 6e9 * self.active_b

    def train_tok_s(self, gpu: Gpu, stage: str) -> float:
        override = self.tok_s_gpu_midtrain if stage == "midtrain" else self.tok_s_gpu_ift
        mfu = self.mfu_midtrain if stage == "midtrain" else self.mfu_ift
        per_gpu = override if override else gpu.tflops_bf16 * 1e12 * mfu / self.flops_per_token()
        return per_gpu * self.n_train_gpus * scaling_eff(self.n_train_gpus)


MODELS: dict[str, ModelPlan] = {
    "gemma3_4b": ModelPlan(
        key="gemma3_4b", hf_id="google/gemma-3-4b-pt", params_b=4.3, active_b=4.3,
        ckpt_gb=9,
        train_gpu="H200", n_train_gpus=2, aft_gpu="H100", n_aft_gpus=1,
        eval_gpu="H100", n_eval_gpus=1,
        # MEASURED: 124 midtrain steps in 33 min train on 2xH200 (RESULTS_4B.md) -> 8.2k/GPU
        tok_s_gpu_midtrain=8_200,
        mfu_ift=0.29,  # MEASURED: Dolci 100M ~72 min/arm on 2xH200 incl. prep
        aft_s_per_step=3.5,  # MEASURED: AFT 512 steps + 6 eval endpoints ~35 min on 1xH100
        gets_conflict_aft=False,
        graft_gpu="H100", n_graft_gpus=4, tok_s_gpu_sdf_lora=3_900,  # EST from 12B anchor
    ),
    "gemma3_12b": ModelPlan(
        key="gemma3_12b", hf_id="google/gemma-3-12b-pt", params_b=12.2, active_b=12.2,
        ckpt_gb=24,
        train_gpu="H200", n_train_gpus=2, aft_gpu="H100", n_aft_gpus=1,
        eval_gpu="H100", n_eval_gpus=1,
        # EST: measured full-param midtrain MFU is 0.214 at 4B and 0.209 at 27B -> ~0.21
        mfu_midtrain=0.21,
        mfu_ift=0.29,
        aft_s_per_step=7.9,  # MEASURED: graft-dose wave, micro 16, 1xH100 (7.74-7.96)
        gets_conflict_aft=True,
        graft_gpu="H100", n_graft_gpus=4, tok_s_gpu_sdf_lora=1_390,  # MEASURED (graft-dose)
    ),
    "gemma3_27b": ModelPlan(
        key="gemma3_27b", hf_id="google/gemma-3-27b-pt", params_b=27.4, active_b=27.4,
        ckpt_gb=54,
        # full-param FSDP does NOT fit 80GB GPUs (proven OOM 8xH100) -> H200/B200 only
        train_gpu="H200", n_train_gpus=8, aft_gpu="H200", n_aft_gpus=1,
        eval_gpu="H200", n_eval_gpus=1,
        # MEASURED: 25-27 s/step at 262,144 tok/update on 8xH200 -> 1,260 tok/s/GPU
        tok_s_gpu_midtrain=1_260,
        # MEASURED: Dolci 100M in 2h01m on 8xH200 -> 1,734 tok/s/GPU (MFU 0.284)
        tok_s_gpu_ift=1_734,
        aft_s_per_step=11.27,  # MEASURED on 1xH200 (RESULTS_27B.md)
        gets_conflict_aft=True,
        # 27B LoRA needs 141GB-class cards at seq 8192; EST throughput from 12B anchor.
        # NB the measured LoRA recipe is FLOP-inefficient (sdpa/no Liger): ~10% eff. MFU
        # vs 21% full-param, so graft midtrain at 27B costs MORE GPU-h than full-param.
        graft_gpu="H200", n_graft_gpus=4, tok_s_gpu_sdf_lora=620,
    ),
    "glm45_air": ModelPlan(
        key="glm45_air", hf_id="zai-org/GLM-4.5-Air-Base", params_b=110.5, active_b=12.0,
        ckpt_gb=214,  # 199 GiB consolidated HF dir MEASURED (jb/glm45-air-midtrain)
        # MEASURED (Jonathan, jb/glm45-air-midtrain, runs 2026-08-18..20): full-param
        # with 8-BIT ADAMW fits ONE 8xH200 node (>=1900 GB host RAM gate -- FSDP2
        # cpu_ram_efficient_loading materializes 8x221 GB CPU buffers); FSDP2
        # SHARDED_STATE_DICT, grouped_mm experts, CCE loss, sdpa attention.
        # midtrain 34.22 s/step @262,144 tok/update -> 958 tok/s/GPU; Dolci SFT
        # 269.9 s/step @2,097,152 -> 971 tok/s/GPU. Both ~7.0% MFU; router health clean.
        # (Full-precision AdamW needs ~1.8TB -> 8xB300; kept as a costed alternative.)
        train_gpu="H200", n_train_gpus=8, aft_gpu="H200", n_aft_gpus=2,
        eval_gpu="H200", n_eval_gpus=2,
        tok_s_gpu_midtrain=958, tok_s_gpu_ift=971,
        aft_s_per_step=14.0,  # still a GUESS: ~2x 12B dense (short seqs, expert dispatch)
        consolidate_hr=3.5,  # MEASURED 3-4 h/stage-end: DCP merge + verify-load + egress
        gets_conflict_aft=True,
        # registry: frozen bf16 params ~221GB -> LoRA fits 2xH200; 8 for wall-clock.
        graft_gpu="H200", n_graft_gpus=8, tok_s_gpu_sdf_lora=700,  # GUESS
    ),
}


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Plan:
    name: str
    models: tuple[str, ...] = ("gemma3_4b", "gemma3_12b", "gemma3_27b", "glm45_air")
    sdf: SdfStage = field(default_factory=SdfStage)
    ift: IftStage = field(default_factory=IftStage)
    aft: AftStage = field(default_factory=AftStage)
    evals: EvalCfg = field(default_factory=EvalCfg)
    ovh: Overheads = field(default_factory=Overheads)
    datagen: DataGen = field(default_factory=DataGen)
    # 'standard':  midtrain (full-param, from PT) -> per-arm IFT -> AFT
    # 'late_sdf':  ONE shared Dolci prefix per model -> per-arm midtrain -> per-arm
    #              Dolci suffix -> AFT   (the dolci90+dolci10 'SDF order' convention, scaled)
    # 'graft':     LoRA SDF on the PT donor -> merge onto the PUBLIC instruct model -> AFT
    #              (no IFT stage at all; graft-dose v1 recipe)
    # 'fp_graft':  FULL-PARAM SDF on the PT donor (standard midtrain recipe) -> weight
    #              diff (task vector) added onto the PUBLIC instruct model -> AFT
    #              (no IFT; one merge job per arm; task-arithmetic a la Ilharco et al.)
    pipeline: str = "standard"
    late_split_mtok: tuple[float, float] = (450.0, 50.0)  # (shared prefix, per-arm suffix)
    max_usd_hr: float = 80.0  # RunPod account spendLimit is PER-HOUR (counts GPUs, not pods)
    contingency: float = 0.35  # incidents have run 35-40% of totals on every study so far
    model_overrides: dict[str, ModelPlan] = field(default_factory=dict)

    def model(self, key: str) -> ModelPlan:
        return self.model_overrides.get(key, MODELS[key])


# --------------------------------------------------------------------------- accounting


@dataclass
class StageCost:
    runs: int = 0
    tokens_mtok: float = 0.0  # tokens presented (training) / prefilled (eval)
    gpu_hours: float = 0.0
    pod_hours: float = 0.0  # wall hours of pods (incl. overheads)
    usd: float = 0.0
    longest_job_hr: float = 0.0

    def add_job(self, hours: float, n_gpus: int, usd_hr_gpu: float, tokens_mtok: float = 0.0):
        self.runs += 1
        self.tokens_mtok += tokens_mtok
        self.gpu_hours += hours * n_gpus
        self.pod_hours += hours
        self.usd += hours * n_gpus * usd_hr_gpu
        self.longest_job_hr = max(self.longest_job_hr, hours)


def transfer_hr(gb: float, ovh: Overheads) -> float:
    return gb / ovh.net_gbyte_s / SECONDS_PER_HOUR


def train_job_hours(m: ModelPlan, gpu: Gpu, stage: str, tokens_mtok: float, ovh: Overheads) -> float:
    compute = tokens_mtok * 1e6 / m.train_tok_s(gpu, stage) / SECONDS_PER_HOUR
    io = transfer_hr(m.ckpt_gb * (1 + ovh.ckpts_published_per_train_job), ovh)
    return compute + io + ovh.pod_setup_hr + m.consolidate_hr


def cost_model(plan: Plan) -> dict[str, dict[str, StageCost]]:
    """-> {model_key: {stage_name: StageCost}}"""
    out: dict[str, dict[str, StageCost]] = {}
    for key in plan.models:
        m = plan.model(key)
        sdf = plan.sdf if m.doses_mtok is None else replace(plan.sdf, doses_mtok=m.doses_mtok)
        train_gpu, aft_gpu, eval_gpu = GPUS[m.train_gpu], GPUS[m.aft_gpu], GPUS[m.eval_gpu]
        stages = {s: StageCost() for s in ("midtrain", "ift", "merge", "aft", "eval")}

        arms = sdf.arms()
        n_aft_agreement = len(arms)
        n_aft_conflict = (
            plan.aft.conflict_mixtures * plan.aft.conflict_substrates if m.gets_conflict_aft else 0
        )
        n_aft_runs = n_aft_agreement + n_aft_conflict

        for _ in range(m.chain_seeds):
            # ---- midtrain
            for _label, unique_mtok in arms:
                presented = sdf.trained_tokens_mtok(unique_mtok)
                if plan.pipeline == "graft":
                    # LoRA SDF on the PT donor (measured graft-dose recipe)
                    g = GPUS[m.graft_gpu]
                    rate = m.tok_s_gpu_sdf_lora * m.n_graft_gpus * scaling_eff(m.n_graft_gpus)
                    hrs = (
                        presented * 1e6 / rate / SECONDS_PER_HOUR
                        + transfer_hr(m.ckpt_gb, plan.ovh)  # donor fetch; adapters are ~500MB
                        + plan.ovh.pod_setup_hr
                    )
                    stages["midtrain"].add_job(hrs, m.n_graft_gpus, g.usd_hr, presented)
                else:
                    hrs = train_job_hours(m, train_gpu, "midtrain", presented, plan.ovh)
                    stages["midtrain"].add_job(hrs, m.n_train_gpus, train_gpu.usd_hr, presented)
            # ---- IFT
            if plan.pipeline == "standard":
                # one full Dolci run per midtrain arm
                for _label, _ in arms:
                    hrs = train_job_hours(m, train_gpu, "ift", plan.ift.tokens_mtok, plan.ovh)
                    stages["ift"].add_job(hrs, m.n_train_gpus, train_gpu.usd_hr, plan.ift.tokens_mtok)
            elif plan.pipeline == "late_sdf":
                # ONE shared Dolci prefix per model; per-arm suffix after midtrain
                prefix, suffix = plan.late_split_mtok
                hrs = train_job_hours(m, train_gpu, "ift", prefix, plan.ovh)
                stages["ift"].add_job(hrs, m.n_train_gpus, train_gpu.usd_hr, prefix)
                for _label, _ in arms:
                    hrs = train_job_hours(m, train_gpu, "ift", suffix, plan.ovh)
                    stages["ift"].add_job(hrs, m.n_train_gpus, train_gpu.usd_hr, suffix)
            elif plan.pipeline == "graft":
                pass  # public instruct model is the post-trained substrate; no IFT
            elif plan.pipeline == "fp_graft":
                # per-arm merge: pull PT base + public IT + midtrained ckpt, add the
                # task vector shard-by-shard, push the merged bf16 parent
                merge_hr = transfer_hr(4 * m.ckpt_gb, plan.ovh) + 0.25
                for _label, _ in arms:
                    stages["merge"].add_job(merge_hr, 1, GPUS[m.aft_gpu].usd_hr)
            else:
                raise ValueError(f"unknown pipeline {plan.pipeline!r}")
            # ---- AFT (LoRA; wave recipe: rows*epochs/32 steps at aft_s_per_step)
            steps = plan.aft.rows * plan.aft.epochs / plan.aft.global_batch
            aft_hr = steps * m.aft_s_per_step / SECONDS_PER_HOUR + plan.ovh.pod_setup_hr \
                + transfer_hr(m.ckpt_gb, plan.ovh)
            for _ in range(n_aft_runs):
                stages["aft"].add_job(aft_hr, m.n_aft_gpus, aft_gpu.usd_hr)
            # ---- evals
            n_endpoints = (
                n_aft_runs * plan.evals.endpoints_per_aft_run
                + len(arms) * plan.evals.endpoints_per_midtrain_arm
            )
            prefill_mtok = plan.evals.prompts_per_endpoint * plan.evals.prefill_tokens / 1e6
            eval_flops = prefill_mtok * 1e6 * 2e9 * m.active_b  # prefill ~ 2N/token
            eval_rate = eval_gpu.tflops_bf16 * 1e12 * plan.evals.mfu_eval * m.n_eval_gpus
            ep_hr = eval_flops / eval_rate / SECONDS_PER_HOUR + plan.evals.fixed_min_per_endpoint / 60.0
            for _ in range(n_endpoints):
                stages["eval"].add_job(ep_hr, m.n_eval_gpus, eval_gpu.usd_hr, prefill_mtok)

        out[key] = stages
    return out


def datagen_usd(plan: Plan) -> tuple[float, float]:
    d = plan.datagen
    new_mtok = max(0.0, d.corpus_needed_mtok_per_arm - d.existing_mtok_per_arm)
    new_mtok *= len(plan.sdf.corpora) * d.overgen_factor
    return new_mtok, new_mtok * d.usd_per_mtok


# --------------------------------------------------------------------------- rendering


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def _fmt_hr(x: float) -> str:
    return f"{x:,.1f}"


def render(plan: Plan) -> str:
    res = cost_model(plan)
    lines: list[str] = []
    add = lines.append
    add(f"# Scenario: {plan.name}")
    add("")
    s = plan.sdf
    ift_desc = {
        "standard": f"IFT {plan.ift.tokens_mtok:g}M Dolci/arm",
        "late_sdf": (
            f"late-SDF: {plan.late_split_mtok[0]:g}M Dolci shared prefix -> midtrain -> "
            f"{plan.late_split_mtok[1]:g}M Dolci/arm"
        ),
        "graft": "graft: LoRA SDF on PT donor merged onto public IT (no IFT)",
        "fp_graft": "fp-graft: full-param SDF on PT donor, weight diff onto public IT (no IFT)",
    }[plan.pipeline]
    add(
        f"pipeline={plan.pipeline} | doses {list(s.doses_mtok)} Mtok x {list(s.corpora)} + control | "
        f"mix={s.mix} (replay {s.replay_ratio:g}:1) x {s.presentations} presentations | "
        f"{ift_desc} | AFT {plan.aft.rows} rows x {plan.aft.epochs} ep "
        f"(+{plan.aft.conflict_mixtures}x{plan.aft.conflict_substrates} conflict cells where flagged) | "
        f"{plan.evals.endpoints_per_aft_run} eval endpoints/AFT run"
    )
    add("")

    hdr = "| model | stage | runs | Mtok | GPU config | GPU-h | pod-h | longest job (h) | USD |"
    sep = "|---|---|---:|---:|---|---:|---:|---:|---:|"
    add(hdr)
    add(sep)
    grand = StageCost()
    stage_totals: dict[str, StageCost] = {
        k: StageCost() for k in ("midtrain", "ift", "merge", "aft", "eval")
    }
    for key, stages in res.items():
        m = plan.model(key)
        mid_cfg = (
            f"{m.n_graft_gpus}x{m.graft_gpu} (LoRA)"
            if plan.pipeline == "graft"
            else f"{m.n_train_gpus}x{m.train_gpu}"
        )
        cfg = {
            "midtrain": mid_cfg,
            "ift": f"{m.n_train_gpus}x{m.train_gpu}",
            "merge": f"1x{m.aft_gpu}",
            "aft": f"{m.n_aft_gpus}x{m.aft_gpu}",
            "eval": f"{m.n_eval_gpus}x{m.eval_gpu}",
        }
        model_usd = 0.0
        for sname, sc in stages.items():
            if sc.runs == 0:
                continue
            add(
                f"| {key} | {sname} | {sc.runs} | {sc.tokens_mtok:,.0f} | {cfg[sname]} "
                f"| {_fmt_hr(sc.gpu_hours)} | {_fmt_hr(sc.pod_hours)} "
                f"| {_fmt_hr(sc.longest_job_hr)} | {_fmt_usd(sc.usd)} |"
            )
            model_usd += sc.usd
            t = stage_totals[sname]
            t.runs += sc.runs
            t.tokens_mtok += sc.tokens_mtok
            t.gpu_hours += sc.gpu_hours
            t.pod_hours += sc.pod_hours
            t.usd += sc.usd
            t.longest_job_hr = max(t.longest_job_hr, sc.longest_job_hr)
        add(f"| {key} | **total** |  |  |  |  |  |  | **{_fmt_usd(model_usd)}** |")
    add("")

    add("| stage | runs | GPU-h | USD | min wall-clock @ spend cap |")
    add("|---|---:|---:|---:|---:|")
    total_usd = 0.0
    for sname, t in stage_totals.items():
        if t.runs == 0:
            continue
        wall = max(t.longest_job_hr, t.usd / plan.max_usd_hr)
        total_usd += t.usd
        add(
            f"| {sname} | {t.runs} | {_fmt_hr(t.gpu_hours)} | {_fmt_usd(t.usd)} "
            f"| {_fmt_hr(wall)} h |"
        )
    new_mtok, dg_usd = datagen_usd(plan)
    add(f"| datagen | - | - | {_fmt_usd(dg_usd)} | ({new_mtok:,.0f} Mtok new synth docs) |")
    total_usd += dg_usd
    add(f"| subtotal |  |  | {_fmt_usd(total_usd)} |  |")
    add(
        f"| contingency ({plan.contingency:.0%}) |  |  | "
        f"{_fmt_usd(total_usd * plan.contingency)} | (incidents ran 35-40% historically) |"
    )
    total_usd *= 1 + plan.contingency
    add(f"| **grand total** |  |  | **{_fmt_usd(total_usd)}** |  |")
    add("")

    gpu_usd = sum(t.usd for t in stage_totals.values())
    seq_wall = sum(
        max(t.longest_job_hr, t.usd / plan.max_usd_hr) for t in stage_totals.values()
    )
    pipe_wall = gpu_usd / plan.max_usd_hr
    add(
        f"wall-clock: >= {_fmt_hr(seq_wall)} h ({seq_wall / 24:,.1f} days) if stages run "
        f"strictly in sequence at the ${plan.max_usd_hr:,.0f}/h spend cap; pipelining "
        f"arms across stages approaches the packing bound {_fmt_hr(pipe_wall)} h "
        f"({pipe_wall / 24:,.1f} days). Add the contingency fraction for pod churn, "
        f"retries, and human serialization (every study so far has needed it)."
    )
    return "\n".join(lines)


def totals(plan: Plan) -> dict[str, float]:
    """Grand totals for a plan: {'usd', 'usd_gpu', 'gpu_h', 'pod_h'} (excl. contingency)."""
    res = cost_model(plan)
    usd_gpu = sum(sc.usd for st in res.values() for sc in st.values())
    return {
        "usd_gpu": usd_gpu,
        "usd": usd_gpu + datagen_usd(plan)[1],
        "gpu_h": sum(sc.gpu_hours for st in res.values() for sc in st.values()),
        "pod_h": sum(sc.pod_hours for st in res.values() for sc in st.values()),
    }


def render_matrix(plans: dict[str, Plan], title: str = "pipeline x model") -> str:
    """Model x pipeline table of per-model GPU cost, plus a per-stage breakdown."""
    lines = [f"# {title}", ""]
    any_plan = next(iter(plans.values()))
    models = list(any_plan.models)

    lines.append("| pipeline | " + " | ".join(models) + " | GPU total | +datagen | +contingency |")
    lines.append("|---|" + "---:|" * (len(models) + 3))
    for pname, plan in plans.items():
        res = cost_model(plan)
        cells = [_fmt_usd(sum(sc.usd for sc in res[k].values())) for k in models]
        t = totals(plan)
        lines.append(
            f"| {pname} | " + " | ".join(cells) + f" | {_fmt_usd(t['usd_gpu'])} "
            f"| {_fmt_usd(t['usd'])} | **{_fmt_usd(t['usd'] * (1 + plan.contingency))}** |"
        )
    lines.append("")

    lines.append("| pipeline | model | midtrain | ift | merge | aft | eval | model total | longest job |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for pname, plan in plans.items():
        res = cost_model(plan)
        for k in models:
            st = res[k]
            row = [_fmt_usd(st[x].usd) if st[x].runs else "-"
                   for x in ("midtrain", "ift", "merge", "aft", "eval")]
            longest = max(sc.longest_job_hr for sc in st.values())
            lines.append(
                f"| {pname} | {k} | " + " | ".join(row)
                + f" | **{_fmt_usd(sum(sc.usd for sc in st.values()))}** | {longest:,.1f} h |"
            )
    return "\n".join(lines)


def one_liner(plan: Plan) -> str:
    res = cost_model(plan)
    usd = sum(sc.usd for stages in res.values() for sc in stages.values())
    usd = (usd + datagen_usd(plan)[1]) * (1 + plan.contingency)
    gpu_h = sum(sc.gpu_hours for stages in res.values() for sc in stages.values())
    return (
        f"{plan.name:<34} {_fmt_usd(usd):>10}   {gpu_h:>7,.0f} GPU-h"
        f"   ~{usd / (1 + plan.contingency) / plan.max_usd_hr / 24:,.1f}+ days"
    )


# --------------------------------------------------------------------------- scenarios

DEFAULT = Plan(name="default (500M dolci, full grid)")

LATE_SDF = replace(DEFAULT, name="late-stage SDF (450M shared + 50M/arm)", pipeline="late_sdf")
GRAFT = replace(DEFAULT, name="graft onto public IT (no IFT)", pipeline="graft")
FP_GRAFT = replace(DEFAULT, name="fp-graft onto public IT (no IFT)", pipeline="fp_graft")

SCENARIOS: tuple[Plan, ...] = (
    DEFAULT,
    LATE_SDF,
    GRAFT,
    FP_GRAFT,
    replace(DEFAULT, name="dolci 100M (prior convention)", ift=IftStage(tokens_mtok=100.0)),
    replace(DEFAULT, name="1 presentation (epochs=1)", sdf=replace(DEFAULT.sdf, presentations=1)),
    replace(DEFAULT, name="topup mix (const compute/cell)", sdf=replace(DEFAULT.sdf, mix="topup")),
    replace(DEFAULT, name="no GLM-4.5-Air", models=("gemma3_4b", "gemma3_12b", "gemma3_27b")),
    replace(
        DEFAULT,
        name="GLM reduced (doses 1.6/16 only)",
        model_overrides={"glm45_air": replace(MODELS["glm45_air"], doses_mtok=(1.6, 16.0))},
    ),
    replace(
        DEFAULT,
        name="docgen at v1/v2 rates ($45/M)",
        datagen=replace(DEFAULT.datagen, usd_per_mtok=45.0),
    ),
    replace(
        DEFAULT,
        name="gemma on B200",
        model_overrides={
            k: replace(MODELS[k], train_gpu="B200", tok_s_gpu_midtrain=None, tok_s_gpu_ift=None)
            for k in ("gemma3_4b", "gemma3_12b", "gemma3_27b")
        },
    ),
)


# --- trimmed grid: doses {5,16,50}M, IFT 100M, no 4B (2026-08-27 brainstorm) ---
TRIM = replace(
    DEFAULT,
    name="trimmed",
    models=("gemma3_12b", "gemma3_27b", "glm45_air"),
    sdf=replace(DEFAULT.sdf, doses_mtok=(5.0, 16.0, 50.0)),
    ift=IftStage(tokens_mtok=100.0),
    late_split_mtok=(90.0, 10.0),  # the repo's own dolci90 + dolci10 convention
)

TRIM_MATRIX: dict[str, Plan] = {
    "full-param midtrain": replace(TRIM, name="trim/standard", pipeline="standard"),
    "full-param late SDF": replace(TRIM, name="trim/late_sdf", pipeline="late_sdf"),
    "full-param graft": replace(TRIM, name="trim/fp_graft", pipeline="fp_graft"),
    "LoRA graft": replace(TRIM, name="trim/graft", pipeline="graft"),
}


if __name__ == "__main__":
    for full in (DEFAULT, LATE_SDF, GRAFT, FP_GRAFT):
        print(render(full))
        print()
    print(render_matrix(TRIM_MATRIX,
                        "Trimmed grid: doses 5/16/50M (1:1 dolmino), IFT 100M, 12B+27B+GLM"))
    print()
    print("# Scenario comparison (grand totals incl. datagen + contingency)")
    print()
    for p in SCENARIOS:
        print(one_liner(p))
