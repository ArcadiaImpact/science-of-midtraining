"""msm_ablation_sweep runner skeleton — cells/chains as data, sequential awaits.

Async, config-first, no CLI (repo conventions). The SPEC's cell table lives
here as the CELLS dict; stage hparams live in the stage-YAML registry
(src/scimt/train/stages/midtrain_msm_* / sft_msm_*); datasets are the
manifested outputs of prep_data.py (Dataset handles under data/). This module
duplicates NO config — it only wires the three together.

Per SPEC:
- Chains per cell: AFT-only control, MSM(america)->AFT, MSM(affordability)->AFT;
  midtrain seed fixed (0) everywhere, seeds vary the AFT stage only.
- Midtrain runs are shared: B's two midtrains feed B / D-ladder / D100-R / ST;
  FP's feed FP and FP-mid; DM and G own theirs (8 distinct midtrain runs).
- LoRA outputs are MERGED before chaining or eval (render_stage refuses
  unmerged adapters); gemma checkpoints additionally need
  hydrate_gemma3_checkpoint. The merge runs ON the training pod right after
  training (MergingBellhopExecutor -> pod_merge.py, via the documented
  AxolotlBackend.pod_executor_factory seam): merged/ goes to the bus and a
  merged_ckpt.json pointer rides the results pull, so chain_input resolves
  it with no manual step. Chained stages take the gs:// merged URI directly
  (the executor rclone-pulls a gs:// load_checkpoint_path on the next pod).
- Evals are batched per cell on ONE eval pod (run_cell_evals ->
  eval_worker.py): pull checkpoint, merge stranded pre-wiring adapters
  (given their base; the bus gets repaired), eval_lib at full n, rows
  mirrored to $SCIMT_GCS_BASE/results/ and pulled into samples/ + results/.
- Checkpoint bus: GCS (SCIMT_GCS_BASE from .env) — the stage PodSpecs default
  to checkpoint_bus="gcs"; manifests carry gs:// pointers, no HF push.
- Idempotent end to end: stages skip on local checkpoint.json, merges on
  the pointer chain (local merged/ -> merged_ckpt.json -> bus probe), eval
  jobs drop out as local sample rows appear — re-running after any crash
  resumes cleanly.
- Loss guard: automatic via the axolotl executor (LocalExecutor stream guard).

POD SIGN-OFF BOUNDARY: every pod launch needs Jonathan's explicit sign-off
(house rule + SPEC "every pod launch gets Jonathan's sign-off").
REQUIRE_CONFIRM gates any stage whose template declares a pod; flip only via
SCIMT_MSM_SWEEP_CONFIRMED=1 after sign-off.

Run:  uv run python experiments/msm_ablation_sweep/runner.py
(edit RUN_CELLS below; the default runs nothing until P2/P3 are gated open).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.dataset import Dataset                       # noqa: E402
from scimt.train import (                               # noqa: E402
    Checkpoint,
    LoraConfig,
    TrainConfig,
    get_backend,
    train_dataset,
)
from scimt.train.axolotl import BellhopExecutor, load_stage  # noqa: E402

DATA = HERE / "data"
RUNS = HERE / "runs"
SAMPLES = HERE / "samples"
RESULTS = HERE / "results"


def eval_out_dir(label: str) -> Path:
    """Per-batch eval pull dir (eval_out_<label>) — shard processes share
    this checkout, so concurrent eval pods must never pull into one dir
    (interleaved tar extracts tear sample rows — review-caught)."""
    return HERE / f"eval_out_{label}"


def eval_jobs_rel(label: str) -> str:
    return f"experiments/msm_ablation_sweep/eval_out_{label}/jobs.json"
# eval pods: light HF-forward stack, 1 GPU, same ladder shape as p2_smoke
EVAL_LADDER = (("H100", "SECURE"), ("H100", "COMMUNITY"),
               ("A100", "SECURE"), ("A100", "COMMUNITY"))
EVAL_POD = {"disk_gb": 200, "timeout_s": 5 * 3600, "max_lifetime_h": 6,
            "ladder_rounds": 3, "provision_pause_s": 120}


def _eval_lib():
    """File-load the sibling eval_lib lazily (experiments are not packages;
    lazy so importing the runner stays independent of the eval seam)."""
    name = "msm_sweep_eval_lib"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "eval_lib.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# ------------------------------------------------------------------ sign-off
# Pod launches cost money and need Jonathan's explicit sign-off (SPEC budget
# section). Do NOT set this to False in code — export
# SCIMT_MSM_SWEEP_CONFIRMED=1 for a signed-off session instead.
REQUIRE_CONFIRM = True

# ------------------------------------------------------------- shared config
VALUES = ("america", "affordability")
# chain name -> midtrain value (None = AFT-only control)
CHAINS: dict[str, str | None] = {
    "aft_only": None,
    "msm_america": "america",
    "msm_affordability": "affordability",
}
MIDTRAIN_SEED = 0  # fixed everywhere — all claims are "given this midtrain draw"

# Paper adapter: r64 alpha128 dropout 0. Llama targets the paper's explicit
# q,k,v,o,gate,up,down; gemma uses target_linear (multimodal wrapper — vision
# projections must not be caught by suffix; SPEC risk #5, pre-registered).
LLAMA_LORA = LoraConfig(
    r=64, alpha=128, dropout=0.0, target_linear=False,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"),
)
GEMMA_LORA = LoraConfig(r=64, alpha=128, dropout=0.0, target_linear=True)

# ------------------------------------------------------------ the cell table
# Mirrors the SPEC "Cells" table exactly (names, seeds, one factor per cell).
# midtrain_owner: which cell's midtrain checkpoints feed this cell's chains.
# sft_stages/sft_data are parallel tuples (ST is the only two-stage SFT).
_LLAMA = dict(substrate="llama", model="llama3_1_8b")
_B_MID = dict(
    midtrain_stage="midtrain_msm_lora_llama31_8b", midtrain_lora=True,
    midtrain_data={v: f"midtrain_{v}" for v in VALUES},
)
CELLS: dict[str, dict[str, Any]] = {
    "B": {**_LLAMA, **_B_MID, "midtrain_owner": "B",
          "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
          "sft_data": ("sft_b_llama",), "seeds": (0, 1, 2)},
    "FP-mid": {**_LLAMA, "midtrain_owner": "FP",  # full midtrain, LoRA SFT
               "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
               "sft_data": ("sft_b_llama",), "seeds": (0, 1)},
    "FP": {**_LLAMA, "midtrain_owner": "FP",
           "midtrain_stage": "midtrain_msm_full_llama31_8b",
           "midtrain_lora": False,
           "midtrain_data": {v: f"midtrain_{v}" for v in VALUES},
           "sft_stages": ("sft_msm_full_llama31_8b",), "sft_lora": False,
           "sft_data": ("sft_b_llama",), "seeds": (0, 1)},
    "DM": {**_LLAMA, "midtrain_owner": "DM",
           "midtrain_stage": "midtrain_msm_lora_llama31_8b",
           "midtrain_lora": True,
           "midtrain_data": {v: f"dm_midtrain_{v}" for v in VALUES},
           "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
           "sft_data": ("sft_b_llama",), "seeds": (0,)},
    **{f"D{m}": {**_LLAMA, "midtrain_owner": "B",
                 "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
                 "sft_data": (f"sft_d{m}",), "seeds": (0,)}
       for m in (10, 20, 50, 100)},
    "D100-R": {**_LLAMA, "midtrain_owner": "B",
               "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
               "sft_data": ("sft_d100r",), "seeds": (0,)},
    # NI: B minus the synthesized identity set (bounds deviation #7)
    "NI": {**_LLAMA, "midtrain_owner": "B",
           "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
           "sft_data": ("sft_ni",), "seeds": (0,)},
    # VIPOT (post-sweep addendum, Jonathan 2026-08-23): instrument-potency
    # check for the VI conflict null — the full anti_america value-QA set as
    # a FOCUSED second SFT stage on the no-midtrain control. Stage 1 is an
    # alias of B_aft_only_s0_sft0 (manifests pre-written; skips). If this
    # can't push the america rate DOWN, the VI injection was inert and the
    # conflict null says nothing about midtraining robustness.
    "VIPOT": {**_LLAMA, "midtrain_owner": "B", "chains": ("aft_only",),
              "sft_stages": ("sft_msm_paper_llama31_8b",
                             "sft_msm_paper_llama31_8b"),
              "sft_lora": True,
              "sft_data": ("sft_b_llama", "vipot_anti_us"), "seeds": (0,)},
    # VP2 (addendum, Jonathan 2026-08-24): POTENT conflict data. VIPOT proved
    # the vi_gen anti_america set behaviorally inert (full-strength SFT moved
    # the control 0.343->0.347 logprob), so the VI conflict null was an
    # instrument failure. vp2_anti_us is regenerated ON-distribution for the
    # eval (A/B stance rows + first-person "I agree that"/"I prefer" leads,
    # valence-verified, same HARD 8-gram guard).
    #   VP2VAL — the potency gate, mirroring VIPOT: the full set as a focused
    #   2nd SFT stage on the no-midtrain control (stage 1 = alias of
    #   B_aft_only_s0_sft0, manifests pre-written; skips). PRE-REGISTERED
    #   GATE: america logprob must drop >=0.05 and >=2*SE vs the control's
    #   0.343 (n=400) before any ladder pod is launched.
    "VP2VAL": {**_LLAMA, "midtrain_owner": "B", "chains": ("aft_only",),
               "sft_stages": ("sft_msm_paper_llama31_8b",
                              "sft_msm_paper_llama31_8b"),
               "sft_lora": True,
               "sft_data": ("sft_b_llama", "vp2_anti_us"), "seeds": (0,)},
    #   Gate probes (added 2026-08-24 after VP2VAL FAILED the gate: final
    #   0.3575 vs stage-0 0.3425 logprob, greedy 0.215 vs 0.190; paired
    #   stance-margin delta only −0.015 nats ± 0.009 — sub-threshold). The
    #   single pre-registered iteration is spent on two instrument probes:
    #   VP2VALE3 — the focused stage at 3 EPOCHS (was the 1-epoch failure
    #   optimization-limited? 380k tok = 3 steps at 1 ep);
    #   VP2SUB — vp2_mix_d100 on the CONTROL chain: the anti set gets the
    #   exact in-mix treatment cheese gets (full-run optimization exposure);
    #   doubles as the ladder's control arm if potent.
    "VP2VALE3": {**_LLAMA, "midtrain_owner": "B", "chains": ("aft_only",),
                 "sft_stages": ("sft_msm_paper_llama31_8b",
                                "sft_msm_paper_llama31_8b_e3"),
                 "sft_lora": True,
                 "sft_data": ("sft_b_llama", "vp2_anti_us"), "seeds": (0,)},
    "VP2SUB": {**_LLAMA, "midtrain_owner": "B", "chains": ("aft_only",),
               "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
               "sft_data": ("vp2_mix_d100",), "seeds": (0,)},
    #   Post-hoc reversal probes (Jonathan 2026-08-24: "See if you can get
    #   this to work post-hoc on the MSM+AFT -> pro-america model"): the
    #   focused anti stage applied ON TOP of the installed model
    #   (stage 1 = alias of B_msm_america_s0_sft0: logprob 0.463, greedy
    #   0.621 — real down-room, unlike the anti-leaning control). PRE-
    #   REGISTERED success: america logprob <=0.413 (>=0.05 drop) and
    #   >=2*SE vs the installed 0.463 (n=400); a pass also satisfies the
    #   ladder gate (potency shown in the exact conflict regime the ladder
    #   tests). VP2POSTE3 = same at 3 epochs.
    **{name: {**_LLAMA, "midtrain_owner": "B", "chains": ("msm_america",),
              "sft_stages": ("sft_msm_paper_llama31_8b", stage2),
              "sft_lora": True,
              "sft_data": ("sft_b_llama", "vp2_anti_us"), "seeds": (0,)}
       for name, stage2 in (("VP2POST", "sft_msm_paper_llama31_8b"),
                            ("VP2POSTE3", "sft_msm_paper_llama31_8b_e3"))},
    #   VP2_d02/d2/d20/d100 — the dose ladder on the msm_america chain only:
    #   the exact B mix + vp2_anti_us sliced to 0.2/2/20/100% of the mix's
    #   cheese tokens (d100 = token parity with cheese). Does validated
    #   conflict SFT data overpower MSM(us)+cheese (B msm_america reference:
    #   logprob 0.463+-0.011, greedy 0.621+-0.009)?
    **{f"VP2_{d}": {**_LLAMA, "midtrain_owner": "B",
                    "chains": ("msm_america",),
                    "sft_stages": ("sft_msm_paper_llama31_8b",),
                    "sft_lora": True,
                    "sft_data": (f"vp2_mix_{d}",), "seeds": (0,)}
       for d in ("d02", "d2", "d20", "d100")},
    "G": {"substrate": "gemma", "model": "gemma3_12b", "midtrain_owner": "G",
          "midtrain_stage": "midtrain_msm_lora_gemma3_12b",
          "midtrain_lora": True,
          "midtrain_data": {v: f"midtrain_{v}" for v in VALUES},
          "sft_stages": ("sft_msm_paper_gemma3_12b",), "sft_lora": True,
          "sft_data": ("sft_b_gemma",), "seeds": (0, 1)},
    "ST": {**_LLAMA, "midtrain_owner": "B",
           "sft_stages": ("sft_msm_paper_llama31_8b",
                          "sft_msm_paper_llama31_8b"),
           "sft_lora": True,
           "sft_data": ("sft_st_stage1", "cheese_train"),
           "seeds": (0,)},
    # VI cells (SPEC 2026-08-20): value-QA injected into the B SFT mix at
    # 0.2/2/20% of cheese tokens. The optional "chains" key restricts a cell
    # to a SUBSET of CHAINS (cells without it run all three): VI-conflict
    # runs ONLY the matching-midtrain chain (anti-value data vs the value it
    # conflicts with, reusing B's merged midtrains); VI-sub runs ONLY
    # aft_only (pro-value data with no midtrain at all).
    **{f"VI_conflict_{tag}_{d}": {
           **_LLAMA, "midtrain_owner": "B",
           "chains": (f"msm_{value}",),
           "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
           "sft_data": (f"vi_conflict_{tag}_{d}",), "seeds": (0,)}
       for tag, value in (("us", "america"), ("aff", "affordability"))
       for d in ("d02", "d2", "d20")},
    **{f"VI_sub_{tag}_{d}": {
           **_LLAMA, "midtrain_owner": "B",
           "chains": ("aft_only",),
           "sft_stages": ("sft_msm_paper_llama31_8b",), "sft_lora": True,
           "sft_data": (f"vi_sub_{tag}_{d}",), "seeds": (0,)}
       for tag in ("us", "aff")
       for d in ("d02", "d2", "d20")},
}

# which cells to execute this session (edit deliberately, phase by phase:
# P3 = ["B"]; P4 = the ablation cells after the B gate).
RUN_CELLS: list[str] = ["B"]  # P3: baseline gate (P2 smoke passed 2026-08-20)


# --------------------------------------------------------------------- utils
def load_dotenv(path: Path = REPO / ".env") -> None:
    """Minimal .env loader (SCIMT_GCS_BASE + rclone creds for the gcs bus)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    # gs:// URIs need an rclone remote literally named "gs"; .env defines
    # the creds once under the GCS_* names — mirror them (P2 postmortem).
    for suffix in ("TYPE", "SERVICE_ACCOUNT_CREDENTIALS", "BUCKET_POLICY_ONLY"):
        src = os.environ.get(f"RCLONE_CONFIG_GCS_{suffix}")
        if src is not None:
            os.environ.setdefault(f"RCLONE_CONFIG_GS_{suffix}", src)
    # crab-factory-2 trap: the injected pod-scoped RUNPOD_API_KEY 403s; the
    # valid key lives in ~/.runpod/config.toml (same resolution as p2_smoke).
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        for cline in cfg.read_text().splitlines():
            ckey, _, cval = cline.partition("=")
            if ckey.strip() == "apikey" and cval.strip().strip("'\""):
                os.environ["RUNPOD_API_KEY"] = cval.strip().strip("'\"")
                break
    # This experiment's mutable dirs — excluded from source-manifest scan and
    # pod-side verify (parallel shards mutate them between a sibling stage's
    # manifest build and tar snapshot; 2026-08-20 DM incident).
    os.environ.setdefault(
        "SCIMT_SOURCE_MANIFEST_EXCLUDE",
        "experiments/msm_ablation_sweep/runs/:"
        "experiments/msm_ablation_sweep/samples/:"
        "experiments/msm_ablation_sweep/results/:"
        "experiments/msm_ablation_sweep/eval_out:"
        # data/ jsonls: static since prep, integrity carried by their own
        # dataset.json manifests; hashing 2.6GB per stage launch saturated the
        # network volume (load 34 on 4 cores, 2026-08-20 night)
        "experiments/msm_ablation_sweep/data/:"
        "experiments/msm_ablation_sweep/shard_:"
        "experiments/msm_ablation_sweep/p3_run",
    )


def confirm_pod_launch(stage_name: str) -> None:
    """The sign-off boundary: refuse to provision pods without it."""
    if load_stage(stage_name).pod is None:
        return  # local execution — no pod, no spend gate
    if REQUIRE_CONFIRM and os.environ.get("SCIMT_MSM_SWEEP_CONFIRMED") != "1":
        raise RuntimeError(
            f"stage {stage_name!r} declares a pod, and pod launches need "
            "Jonathan's explicit sign-off (SPEC budget rule). After sign-off, "
            "export SCIMT_MSM_SWEEP_CONFIRMED=1 and re-run."
        )


def stage_done(out_dir: Path) -> Checkpoint | None:
    """Idempotency: a stage with a checkpoint manifest is complete."""
    manifest = out_dir / "checkpoint.json"
    return Checkpoint.load(out_dir) if manifest.exists() else None


def dataset(name: str) -> Dataset:
    """A prep_data.py output by manifest (loud if prep hasn't run)."""
    return Dataset.load(DATA / name)


def lora_for(cell: dict[str, Any]) -> LoraConfig:
    return GEMMA_LORA if cell["substrate"] == "gemma" else LLAMA_LORA


# ------------------------------------------- parallelism & env-driven shards
# Process sharding contract (Jonathan, 2026-08-20 — minimize wallclock):
#   SCIMT_MSM_RUN_CELLS="B,D10"  overrides RUN_CELLS (comma list; loud on
#                                unknown names) so cells shard across
#                                processes; cells within a process still run
#                                sequentially — shard for cell-level overlap.
#   SCIMT_MSM_MAX_PODS=6         global live-pod cap for THIS process
#                                (training + eval pods share one semaphore).
#   SCIMT_MSM_SKIP_EVAL=1        train only; eval jobs are recorded to
#                                runs/eval_jobs_<cell>.json and batched later.
#   SCIMT_MSM_EVAL_CELLS="B,.."  eval-ONLY mode: no training, reconstruct all
#                                named cells' jobs (loud if any stage is
#                                incomplete) and batch them on ONE eval pod.
#   SCIMT_MSM_MIDTRAIN_WAIT_S    cross-process midtrain wait (default 14400).


def env_cells(var: str) -> list[str] | None:
    """Parse a comma-separated cell list from the environment (None when
    unset/empty); unknown cell names are a loud error, not a silent skip."""
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        return None
    names = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in names if c not in CELLS]
    if unknown:
        raise ValueError(
            f"{var} names unknown cells {unknown}; known: {sorted(CELLS)}")
    return names


def effective_run_cells() -> list[str]:
    """The cells THIS process owns: SCIMT_MSM_RUN_CELLS over RUN_CELLS.
    Ownership matters beyond selection — only a process whose shard includes
    a midtrain's owner cell may train that midtrain (see ensure_midtrain)."""
    return env_cells("SCIMT_MSM_RUN_CELLS") or RUN_CELLS


def skip_eval() -> bool:
    return os.environ.get("SCIMT_MSM_SKIP_EVAL") == "1"


# Lazy asyncio primitives — created inside the running loop, NEVER at import
# (import-safety: a live supervisor may restart this module at any time).
_POD_SEM: asyncio.Semaphore | None = None
_MIDTRAIN_LOCKS: dict[str, asyncio.Lock] = {}


def pod_semaphore() -> asyncio.Semaphore:
    """The global pod-concurrency cap: one permit per live pod this process
    starts (training or eval). SCIMT_MSM_MAX_PODS, default 6."""
    global _POD_SEM
    if _POD_SEM is None:
        _POD_SEM = asyncio.Semaphore(
            int(os.environ.get("SCIMT_MSM_MAX_PODS", "6")))
    return _POD_SEM


def midtrain_lock(key: str) -> asyncio.Lock:
    """In-process dedupe for shared midtrains: concurrent chains/cells that
    reference the same midtrain serialize here — the first trains, the rest
    wake to a manifest and skip."""
    return _MIDTRAIN_LOCKS.setdefault(key, asyncio.Lock())


# ------------------------------------------------- on-pod post-train merging
def post_train_merge_lines(*, adapter: bool, stage_name: str,
                           rendered_rel: str, out_rel: str,
                           gcs_base: str) -> list[str]:
    """The pod shell line(s) that merge a just-trained LoRA adapter on the
    SAME pod (pod_merge.py: merge + gemma hydration + merged manifest + gcs
    push + devbox pointer). Pure — unit-tested; [] for full-param runs."""
    if not adapter:
        return []
    if not gcs_base:
        raise ValueError("post-train merge needs SCIMT_GCS_BASE (gcs bus)")
    uri = f"{gcs_base.rstrip('/')}/{Path(out_rel).name}/merged/"
    substrate = "gemma" if "gemma" in stage_name else "llama"
    return [
        "python3 experiments/msm_ablation_sweep/pod_merge.py"
        f" --rendered {shlex.quote(rendered_rel)}"
        f" --out {shlex.quote(out_rel)}"
        f" --gcs-uri {shlex.quote(uri)}"
        f" --substrate {substrate}"
    ]


def post_train_consolidate_lines(*, fsdp: bool, rendered_rel: str,
                                 out_rel: str, gcs_base: str) -> list[str]:
    """The full-param twin of post_train_merge_lines: consolidate the FSDP2
    SHARDED_STATE_DICT checkpoint ON the training pod (pod_consolidate.py ->
    the proven ex06 consolidator), publish the loadable full checkpoint to
    the SAME bus location the LoRA merges use (gs://.../<run>/merged/ — one
    pointer convention for 'the loadable form'). Pure — unit-tested; [] for
    non-FSDP runs (a plain full save is already loadable)."""
    if not fsdp:
        return []
    if not gcs_base:
        raise ValueError("post-train consolidation needs SCIMT_GCS_BASE (gcs bus)")
    uri = f"{gcs_base.rstrip('/')}/{Path(out_rel).name}/merged/"
    return [
        "python3 experiments/msm_ablation_sweep/pod_consolidate.py"
        f" --rendered {shlex.quote(rendered_rel)}"
        f" --out {shlex.quote(out_rel)}"
        f" --gcs-uri {shlex.quote(uri)}"
    ]


class MergingBellhopExecutor(BellhopExecutor):
    """The sweep's training-pod executor: identical to the library executor,
    plus post_run_lines that make every checkpoint LOADABLE on the training
    pod right after training (before bus egress) — LoRA adapters are merged
    (pod_merge.py), FSDP2 full-param shards are consolidated
    (pod_consolidate.py) — so ~60 checkpoints chain and eval without any
    manual boundary. Installed via the documented
    AxolotlBackend.pod_executor_factory seam in main()."""

    def post_run_lines(self, stage, *, rendered_rel: str, out_rel: str,
                       run_name: str) -> list[str]:
        body = yaml.safe_load((REPO / rendered_rel).read_text())
        if body.get("adapter") == "lora":
            return post_train_merge_lines(
                adapter=True, stage_name=stage.name,
                rendered_rel=rendered_rel, out_rel=out_rel,
                gcs_base=self.gcs_base or "")
        return post_train_consolidate_lines(
            fsdp=bool(body.get("fsdp_version")), rendered_rel=rendered_rel,
            out_rel=out_rel, gcs_base=self.gcs_base or "")


def gs_run_base(out_dir: Path) -> str | None:
    """This run's bus prefix (matches the executor's egress layout)."""
    base = os.environ.get("SCIMT_GCS_BASE")
    return f"{base.rstrip('/')}/{out_dir.name}" if base else None


# every bus rclone gets the robustness flags (2026-08-20: two pods hung
# forever on a stalled checkpoints/ egress) — same set the library egress
# and the pod scripts use (scimt.train.axolotl.RCLONE_BUS_FLAGS)
RCLONE_FLAGS = ("--timeout", "5m", "--contimeout", "60s",
                "--retries", "4", "--low-level-retries", "20")


def probe_gs(uri: str) -> bool:
    """Does a bus object exist? devbox rclone (creds from .env); degrades to
    False without rclone — callers then fall to their loud boundary."""
    if not shutil.which("rclone"):
        return False
    r = subprocess.run(["rclone", "lsf", *RCLONE_FLAGS, uri],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _atomic_write(path: Path, text: str) -> None:
    """write-to-temp + rename: shard processes share runs/ — a reader must
    never see a torn manifest."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _bus_merged_checkpoint(out_dir: Path,
                           merged_from: str | None = None) -> Checkpoint | None:
    """The bus-resident loadable form for a run (gs://.../<name>/merged/ —
    LoRA merges and FSDP consolidations publish to the same location), or
    None. A hit is cached as <out_dir>/merged_ckpt.json so later resolutions
    (and other tools reading the run dir) skip the network probe. Works with
    NO local run state — the cross-process completion signal."""
    run_base = gs_run_base(out_dir)
    if not run_base:
        return None
    uri = f"{run_base}/merged/"
    if not probe_gs(uri + "checkpoint.json"):
        return None
    found = Checkpoint(
        backend="axolotl", sampler=uri, state=uri,
        meta={"note": "bus probe (runner._bus_merged_checkpoint)",
              **({"merged_from": merged_from} if merged_from else {})})
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / "merged_ckpt.json", json.dumps(
        {**found.as_dict(), "sampler_path": uri, "state_path": uri}, indent=2))
    return found


async def run_stage(
    *,
    stage: str,
    data_name: str,
    out_dir: Path,
    model: str,
    seed: int,
    lora: LoraConfig | None,
    resume: Checkpoint | None,
    run_name: str,
) -> Checkpoint:
    """One idempotent stage: skip-if-manifested, sign-off gate, then
    scimt.train.train_dataset (loss guard + gcs bus live in the backend),
    under the global pod-concurrency semaphore (one permit per live pod)
    and a client-side WATCHDOG (SCIMT_MSM_STAGE_TIMEOUT_S, default 7200 —
    raise it for long stages like the 100M-token D-ladder SFTs): a stage
    that outlives its deadline is cancelled (bellhop's pod contextmanager
    tears the pod down in its finally), then the bus is probed — if the run
    actually finished and only its egress/results-pull hung (the observed
    2026-08-20 failure mode), the local manifest is reconstructed and the
    sweep continues; otherwise the stage fails loudly. The semaphore permit
    is released either way (async with)."""
    done = stage_done(out_dir)
    if done is not None:
        print(f"[runner] skip {run_name}: checkpoint manifest exists")
        return done
    confirm_pod_launch(stage)
    cfg = TrainConfig(backend="axolotl", model=model, stage=stage,
                      seed=seed, lora=lora)
    timeout_s = int(os.environ.get("SCIMT_MSM_STAGE_TIMEOUT_S", "7200"))
    async with pod_semaphore():
        try:
            return await asyncio.wait_for(
                train_dataset(dataset(data_name), out_dir, cfg,
                              run_name=run_name, resume=resume),
                timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            print(f"[runner] WATCHDOG: {run_name} exceeded {timeout_s}s — "
                  "pod cancelled (bellhop teardown); probing the bus for a "
                  "finished run whose egress/pull hung", flush=True)
            recovered = recover_stage_from_bus(out_dir, cfg=cfg,
                                               run_name=run_name)
            if recovered is not None:
                print(f"[runner] WATCHDOG: {run_name} recovered from the bus "
                      f"-> {recovered.sampler}", flush=True)
                return recovered
            raise RuntimeError(
                f"stage {run_name} timed out after {timeout_s}s with nothing "
                "usable on the bus — a genuinely failed/hung run (rerun "
                "retrains it; export a larger SCIMT_MSM_STAGE_TIMEOUT_S for "
                "long stages)"
            ) from None


def recover_stage_from_bus(out_dir: Path, *, cfg: TrainConfig,
                           run_name: str) -> Checkpoint | None:
    """Self-recovery for the hung-egress failure mode: training + on-pod
    merge finished (merged/checkpoint.json is on the bus) but the
    checkpoints/ egress or results pull stalled. Reconstructs the local
    stage manifest the results path would have produced (mirrors the two
    2026-08-20 hand recoveries: sampler/state -> the bus checkpoints/ dir
    when present, else the merged/ dir) and returns the Checkpoint. None
    when the bus has nothing loadable — the stage genuinely failed."""
    run_base = gs_run_base(out_dir)
    if not run_base:
        return None
    merged_uri = f"{run_base}/merged/"
    if not probe_gs(merged_uri + "checkpoint.json"):
        return None
    ckpts_uri = f"{run_base}/checkpoints/"
    pointer = ckpts_uri if probe_gs(ckpts_uri) else merged_uri
    ckpt = Checkpoint(
        backend="axolotl", sampler=pointer, state=pointer, model=cfg.model,
        meta={
            "experiment": f"scimt-train:{run_name}",
            "note": ("manifest reconstructed devbox-side by the runner "
                     "watchdog: pod finished + pushed to bus, egress/results "
                     "pull hung (see run.log if pulled)"),
            "train": {"stage": cfg.stage, "seed": cfg.seed,
                      "load_checkpoint_path": cfg.load_checkpoint_path,
                      "lora": (None if cfg.lora is None
                               else cfg.lora.__dict__.copy())},
            "run_name": run_name,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / "checkpoint.json", json.dumps(
        {**ckpt.meta, **ckpt.as_dict(),
         "sampler_path": pointer, "state_path": pointer}, indent=2))
    (out_dir / f"ckpt_{run_name}.txt").write_text(pointer + "\n")
    return ckpt


def merged_pointer(ckpt: Checkpoint, out_dir: Path) -> Checkpoint | None:
    """The best available LOADABLE handle for a stage output — no GPU work,
    only pointer resolution (checked in order):

      1. <out_dir>/merged/checkpoint.json    — a local merged dir (p2 shape);
      2. <out_dir>/merged_ckpt.json          — the pod-produced gs:// pointer
         (pod_merge.py writes it on the training pod; rides the results pull);
      3. a local dir with config.json (and no adapter_config.json) — already
         a loadable full checkpoint. config.json is the tell: an FSDP2
         SHARDED_STATE_DICT distcp dir has none and must be CONSOLIDATED
         first, exactly like an adapter must be merged — the old silent
         passthrough of sharded FP checkpoints is a review-caught bug;
      4. a bus probe for gs://.../<name>/merged/checkpoint.json (covers the
         crash window where the pod merged+pushed but the pull was lost;
         a future FP consolidation step can publish to the same location);
         a hit is cached as merged_ckpt.json.

    None = nothing loadable anywhere: a stranded pre-wiring LoRA adapter
    (still evaluable — the eval worker merges given a base) or an
    unconsolidated full-param checkpoint (NOT usable until consolidated).
    """
    done = stage_done(out_dir / "merged")
    if done is not None:
        return done
    pointer = out_dir / "merged_ckpt.json"
    if pointer.exists():
        return Checkpoint.load(pointer)
    state = ckpt.require_state()
    # a LOADABLE local full checkpoint passes through (config.json is the
    # tell — an FSDP2 SHARDED_STATE_DICT distcp dir has none and must be
    # consolidated first, exactly like an adapter must be merged)
    if (Path(state) / "config.json").exists() and \
            not (Path(state) / "adapter_config.json").exists():
        return ckpt
    if state.startswith("gs://"):
        return _bus_merged_checkpoint(out_dir, merged_from=state)
    return None


async def chain_input(ckpt: Checkpoint, cell: dict[str, Any], out_dir: Path,
                      *, lora: bool = True) -> Checkpoint:
    """A checkpoint usable as the NEXT stage's init: merged (and, for gemma,
    hydrated), possibly as a gs:// pointer — the executor rclone-pulls a
    gs:// load_checkpoint_path on the next pod. On-pod post-train merging
    (MergingBellhopExecutor + pod_merge.py) produces the pointer
    automatically; this boundary only raises for legacy adapters trained
    before that wiring (re-merge GPU-side, or eval-only via the eval worker).
    """
    m = merged_pointer(ckpt, out_dir)
    if m is not None:
        return m
    if lora:
        raise RuntimeError(
            f"checkpoint {ckpt.require_state()!r} has no merged form anywhere "
            f"(local merged/, merged_ckpt.json, or {gs_run_base(out_dir)}/merged/ "
            "on the bus) — it predates the on-pod merge wiring. Merge it GPU-side "
            "(experiments/msm_ablation_sweep/pod_merge.py on a pod, or the eval "
            "worker's stranded-adapter path) and re-run (idempotent)."
        )
    raise RuntimeError(
        f"full-param checkpoint {ckpt.require_state()!r} is not loadable "
        "(FSDP2 SHARDED_STATE_DICT saves need GPU-side consolidation) and no "
        f"consolidated form exists at {gs_run_base(out_dir)}/merged/. New FP "
        "runs consolidate on the training pod automatically (pod_consolidate"
        ".py via MergingBellhopExecutor); this run predates that wiring — "
        "consolidate it GPU-side (pod_consolidate.py, or "
        "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py + a bus "
        "push) and re-run (idempotent). Loud so no GPU money is spent on an "
        "unloadable chain."
    )


async def ensure_midtrain(owner: str, value: str) -> Checkpoint:
    """The (shared) midtrain checkpoint for a cell family, merged/consolidated
    for chaining. Safe under full concurrency, two mechanisms (documented
    choice):

    - IN-PROCESS: an asyncio.Lock per midtrain key — of the concurrent
      chains/cells referencing one midtrain, the first trains it, the rest
      wake to the manifest and skip.
    - CROSS-PROCESS (sharded cells): NO gs-side lock files — the completion
      signal is the bus MERGED MANIFEST (gs://.../merged/checkpoint.json),
      which the training pod publishes atomically at the end of its on-pod
      merge/consolidation; polling an artifact cannot deadlock on a stale
      lock. Ownership is by shard: only a process whose effective cell list
      includes the OWNER cell may train the midtrain; every other process
      polls the bus until the owner publishes (SCIMT_MSM_MIDTRAIN_WAIT_S,
      default 4h, then loud).
    """
    cell = CELLS[owner]
    out_dir = RUNS / f"midtrain_{owner}_{value}_s{MIDTRAIN_SEED}"
    async with midtrain_lock(f"{owner}_{value}"):
        if stage_done(out_dir) is None:
            bus = _bus_merged_checkpoint(out_dir)
            if bus is not None:  # another process already trained + published
                print(f"[runner] midtrain {out_dir.name}: bus merged manifest "
                      "found — reusing (trained elsewhere)")
                return bus
            if owner not in effective_run_cells():
                return await _await_bus_midtrain(out_dir, owner, value)
        ckpt = await run_stage(
            stage=cell["midtrain_stage"],
            data_name=cell["midtrain_data"][value],
            out_dir=out_dir,
            model=cell["model"],
            seed=MIDTRAIN_SEED,
            lora=lora_for(cell) if cell["midtrain_lora"] else None,
            resume=None,
            run_name=f"msm-sweep-midtrain-{owner}-{value}",
        )
        return await chain_input(ckpt, cell, out_dir, lora=cell["midtrain_lora"])


async def _await_bus_midtrain(out_dir: Path, owner: str,
                              value: str) -> Checkpoint:
    """Poll the bus for a midtrain another process owns (see ensure_midtrain).
    Returns the merged pointer the moment it appears; loud on timeout.

    Deliberately tolerant of the owner FAILING attempts inside the window
    (2026-08-20 incident): the owner's watchdog may kill a hung midtrain pod
    and a rerun retrains it — absence on the bus is transient until the full
    SCIMT_MSM_MIDTRAIN_WAIT_S budget is spent, so the waiter keeps polling
    rather than dying with the owner's first attempt. Poll interval:
    SCIMT_MSM_MIDTRAIN_POLL_S (default 60)."""
    wait_s = int(os.environ.get("SCIMT_MSM_MIDTRAIN_WAIT_S", "14400"))
    poll_s = float(os.environ.get("SCIMT_MSM_MIDTRAIN_POLL_S", "60"))
    deadline = asyncio.get_running_loop().time() + wait_s
    print(f"[runner] midtrain {out_dir.name}: owned by cell {owner!r} "
          f"(not in this shard) — polling the bus up to {wait_s}s")
    while True:
        bus = _bus_merged_checkpoint(out_dir)
        if bus is not None:
            return bus
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(
                f"midtrain-wait failure: {out_dir.name} never appeared on "
                f"the bus within {wait_s}s. This shard does not own cell "
                f"{owner!r} so it will not train it — the owner shard's "
                "attempt(s) presumably failed (its watchdog retrains on "
                "rerun). Check/restart the shard running "
                f"SCIMT_MSM_RUN_CELLS containing {owner!r}, then rerun this "
                "shard (idempotent), or raise SCIMT_MSM_MIDTRAIN_WAIT_S."
            )
        await asyncio.sleep(poll_s)


def eval_scorers(chain: str) -> tuple[str, ...]:
    """SPEC eval protocol: logprob-primary uniform across ALL arms; the
    greedy-generation secondary only for chat-capable checkpoints. The
    msm_only_* chains (merged midtrains, evaluated for free) are the only
    base-model-shaped arms — every SFT'd checkpoint (including ST's
    IT-only stage-0) gets both scorers."""
    return ("logprob",) if chain.startswith("msm_only") else ("logprob", "generate")


def pending_scorers(cell_name: str, chain: str, seed: int) -> list[str]:
    """The scorers this checkpoint still needs locally: a scorer is done when
    BOTH evals' row files exist under samples/ (pulled from an eval pod) —
    the idempotence key for the whole eval path."""
    lib = _eval_lib()
    return [
        sc for sc in eval_scorers(chain)
        if any(not (SAMPLES / lib.store_name(cell_name, chain, seed, key)
                    / f"rows_{sc}.jsonl").exists()
               for key in lib.EVALS)
    ]


def eval_job(cell_name: str, chain: str, seed: int, uri: str, *,
             base: str | None = None,
             push_merged_uri: str | None = None) -> dict[str, Any] | None:
    """One eval-worker job dict, or None when local rows already cover it
    (shared midtrains dedupe here too — their store key is the owner cell)."""
    scorers = pending_scorers(cell_name, chain, seed)
    if not scorers:
        print(f"[runner] skip eval {cell_name}/{chain}/s{seed}: rows present")
        return None
    job: dict[str, Any] = {
        "cell": cell_name, "chain": chain, "seed": seed, "uri": uri,
        "substrate": CELLS.get(cell_name, {}).get("substrate", "llama"),
        "scorers": scorers,
    }
    if base is not None:
        job["base"] = base
    if push_merged_uri is not None:
        job["push_merged_uri"] = push_merged_uri
    return job


def _merge_pulled_evals(pulled: Path) -> int:
    """Fold an eval pod's pulled eval_out/ into the experiment's canonical
    samples/ + results/ — sample ROW FILES copied per file (an existing row
    file is immutable and never overwritten, but a new scorer's file joins
    its existing store), result rows merged with eval_lib's latest-state key
    semantics. The WHOLE fold runs under an flock: shard processes on this
    box share samples/ + results/, and an unlocked read-modify-write could
    drop rows; row-file copies are additionally write-to-temp + rename so a
    concurrent reader never sees a torn store. Returns rows merged."""
    import fcntl

    lib = _eval_lib()
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / ".sweep_results.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for store in sorted((pulled / "samples").glob("*")):
            dst = SAMPLES / store.name
            dst.mkdir(parents=True, exist_ok=True)
            for rows_file in sorted(store.glob("rows_*.jsonl")):
                if (dst / rows_file.name).exists():
                    print(f"[runner] keep existing "
                          f"{store.name}/{rows_file.name} "
                          "(raw samples are immutable)")
                    continue
                _atomic_write(dst / rows_file.name, rows_file.read_text())
        src_rows = pulled / "results" / "sweep_results.jsonl"
        if not src_rows.exists():
            return 0
        new = lib._read_rows(src_rows)
        dst_rows = RESULTS / "sweep_results.jsonl"

        def _key(r: dict[str, Any]):
            return (r.get("cell"), r.get("chain"), r.get("seed"),
                    r.get("eval"), r.get("scorer"))

        new_keys = {_key(r) for r in new}
        kept = ([r for r in lib._read_rows(dst_rows)
                 if _key(r) not in new_keys] if dst_rows.exists() else [])
        lib._write_rows(dst_rows, kept + new)
    return len(new)


async def run_cell_evals(cell_name: str, jobs: list[dict[str, Any]]) -> None:
    """ONE eval pod for a cell's batch (SPEC: evals batched per pod): stage
    the p2-style push tree + jobs.json, provision down the ladder, let
    eval_worker.py pull/merge/eval/mirror, then fold the pulled rows into
    samples/ + results/. Loud if any job is still missing rows afterwards.
    Holds one pod-semaphore permit for the pod's lifetime."""
    # drop skipped-eval Nones and dedupe by (cell, chain, seed): concurrent
    # chains of shared-midtrain cells can propose the same msm_only job twice
    seen: set[tuple[str, str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for j in jobs:
        if j is None:
            continue
        key = (j["cell"], j["chain"], j["seed"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)
    jobs = deduped
    if not jobs:
        print(f"[runner] cell {cell_name}: no pending evals")
        return
    if REQUIRE_CONFIRM and os.environ.get("SCIMT_MSM_SWEEP_CONFIRMED") != "1":
        raise RuntimeError(
            f"cell {cell_name} needs an eval pod ({len(jobs)} jobs), and pod "
            "launches need Jonathan's explicit sign-off (SPEC budget rule). "
            "After sign-off, export SCIMT_MSM_SWEEP_CONFIRMED=1 and re-run."
        )
    import tempfile
    from datetime import timedelta

    import bellhop

    p2 = _p2()
    # per-batch pull dir — concurrent shards must never share one (torn tars)
    pulled = eval_out_dir(cell_name)
    stage_dir = Path(tempfile.mkdtemp(prefix=f"msm-eval-{cell_name}-"))
    print(f"[runner] staging eval push tree -> {stage_dir}")
    p2.stage_push_tree(stage_dir)
    jobs_file = stage_dir / eval_jobs_rel(cell_name)
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps(jobs, indent=2))
    RUNS.mkdir(parents=True, exist_ok=True)  # the as-run record, devbox-side
    (RUNS / f"eval_jobs_{cell_name}.json").write_text(json.dumps(jobs, indent=2))

    setup = " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "(apt-get update -q && apt-get install -y -q rclone) "
        ">/dev/null 2>&1 || true",
        # HF-forward eval stack only — no axolotl/flash-attn (fast setup).
        # torch pinned to the proven F0 recipe (cu121) and the image's stale
        # torchaudio/torchvision removed: they break transformers' lazy
        # torch-backend check at RUNTIME while a bare `import torch` passes
        # (F0 postmortem; recurred on the first VI eval pod 2026-08-21).
        "retry uv pip install --system -q torch==2.5.1 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "uv pip uninstall --system -q torchaudio torchvision || true",
        "retry uv pip install --system -q 'transformers>=4.50' peft datasets "
        "pyyaml jinja2 httpx omegaconf",
        "python3 -c 'from transformers import AutoModelForCausalLM; "
        "import torch; assert torch.cuda.is_available()'",
        "command -v rclone",
    ])
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "MSM_EVAL_JOBS": eval_jobs_rel(cell_name),
        "MSM_EVAL_OUT": f"eval_out_{cell_name}",
        **{k: v for k in ("HF_TOKEN", "SCIMT_GCS_BASE",
                          "RCLONE_CONFIG_GCS_TYPE",
                          "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
                          "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
                          "RCLONE_CONFIG_GS_TYPE",
                          "RCLONE_CONFIG_GS_SERVICE_ACCOUNT_CREDENTIALS",
                          "RCLONE_CONFIG_GS_BUCKET_POLICY_ONLY")
           if (v := os.environ.get(k))},
    }
    last: Exception | None = None
    plan = list(EVAL_LADDER) * EVAL_POD["ladder_rounds"]
    sem = pod_semaphore()  # eval pods count against the same live-pod cap
    await sem.acquire()
    try:
        for gpu, cloud in plan:
            spec = bellhop.RunSpec(
                slug=f"msm-eval-{cell_name}",
                codebase=str(stage_dir),
                setup=setup,
                run="python3 experiments/msm_ablation_sweep/eval_worker.py",
                results_subdir=f"experiments/msm_ablation_sweep/eval_out_{cell_name}",
                local_out=str(HERE),
                gcs_base=None,  # the worker mirrors to GCS itself
                env=env,
                timeout=EVAL_POD["timeout_s"],
            )
            cfg = bellhop.PodConfig(
                gpu=gpu, gpu_count=1,
                container_disk_gb=EVAL_POD["disk_gb"],
                cuda_versions=list(
                    load_stage("sft_msm_paper_llama31_8b").pod.cuda_versions),
                cloud=cloud, cloud_fallback=False,
                provision_timeout=timedelta(seconds=1200),
                ready_timeout=timedelta(seconds=1200),
                max_lifetime=timedelta(hours=EVAL_POD["max_lifetime_h"]),
                name=f"scimt-msm-eval-{cell_name}".lower(),
            )
            try:
                print(f"[runner] provisioning eval pod 1x{gpu} ({cloud}) "
                      f"for {len(jobs)} jobs", flush=True)
                await bellhop.run(spec, cfg)
                break
            except bellhop.ProvisionError as e:
                print(f"[runner] no capacity: 1x{gpu} {cloud}", flush=True)
                last = e
                await asyncio.sleep(EVAL_POD["provision_pause_s"])
            except (bellhop.RemoteJobError, bellhop.ExecTimeoutError):
                # partial results were pulled before the raise — fold them so
                # the rerun's job list shrinks to the genuinely missing evals
                n = _merge_pulled_evals(pulled)
                print(f"[runner] eval pod failed; folded {n} partial rows "
                      "before re-raising", flush=True)
                raise
        else:
            raise RuntimeError(f"no eval-pod capacity on any rung: {last}")
    finally:
        sem.release()
        shutil.rmtree(stage_dir, ignore_errors=True)

    n = _merge_pulled_evals(pulled)
    print(f"[runner] merged {n} result rows -> {RESULTS / 'sweep_results.jsonl'}")
    still = [(j["cell"], j["chain"], j["seed"],
              pending_scorers(j["cell"], j["chain"], j["seed"])) for j in jobs]
    missing = [s for s in still if s[3]]
    if missing:
        raise RuntimeError(
            f"eval pod for {cell_name} returned but rows are still missing "
            f"for {missing} — check {pulled / 'run.log'}"
        )


def _p2():
    """File-load p2_smoke lazily (stage_push_tree is the shared launcher
    plumbing — one gitignore-respecting push tree for every pod we start)."""
    name = "msm_sweep_p2"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "p2_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def _run_sft_chain(name: str, cell: dict[str, Any], chain: str,
                         seed: int, init: Checkpoint | None
                         ) -> list[dict[str, Any] | None]:
    """One (chain, seed) SFT chain — stages stay SEQUENTIAL (each resumes the
    previous stage's merged state); concurrency lives above this level."""
    jobs: list[dict[str, Any] | None] = []
    prev = init
    for i, (stage, data_name) in enumerate(
            zip(cell["sft_stages"], cell["sft_data"])):
        out_dir = RUNS / f"{name}_{chain}_s{seed}_sft{i}"
        ckpt = await run_stage(
            stage=stage, data_name=data_name, out_dir=out_dir,
            model=cell["model"], seed=seed,
            lora=lora_for(cell) if cell["sft_lora"] else None,
            resume=prev,
            run_name=f"msm-sweep-{name}-{chain}-s{seed}-sft{i}",
        )
        terminal = i == len(cell["sft_stages"]) - 1
        chain_label = chain if terminal else f"{chain}_stage{i}"
        # outputs are made LOADABLE on the training pod (LoRA merge / FSDP
        # consolidation) — the pointer resolves here; a stranded pre-wiring
        # adapter is still evaluable (the worker merges it given its base
        # and repairs the bus), just not chainable
        merged = merged_pointer(ckpt, out_dir)
        if merged is not None:
            jobs.append(eval_job(name, chain_label, seed, merged.sampler))
        elif cell["sft_lora"]:
            base = (prev.sampler if prev is not None
                    else load_stage(stage).base_model)
            run_base = gs_run_base(out_dir)
            jobs.append(eval_job(
                name, chain_label, seed, ckpt.sampler, base=base,
                push_merged_uri=(f"{run_base}/merged/"
                                 if run_base else None)))
        else:
            # full-param with no loadable form: an unconsolidated FSDP2
            # save — loud BEFORE any eval/chain spend
            await chain_input(ckpt, cell, out_dir, lora=False)
        if not terminal:  # chaining REQUIRES the merged form
            prev = (merged if merged is not None else
                    await chain_input(ckpt, cell, out_dir,
                                      lora=cell["sft_lora"]))
    return jobs


def _summarize(err: BaseException) -> str:
    return f"{type(err).__name__}: {str(err)[:300]}"


async def _run_chain(name: str, cell: dict[str, Any],
                     chain: str) -> list[dict[str, Any] | None]:
    """One of the cell's chains: (shared) midtrain first, then every AFT seed
    CONCURRENTLY. Seed failures are ISOLATED (gather(return_exceptions) —
    a failed seed must not cancel its siblings' live pods; 2026-08-20
    incident) and re-raised as one aggregate after every seed settles."""
    value = CHAINS[chain]
    jobs: list[dict[str, Any] | None] = []
    init: Checkpoint | None = None
    if value is not None:
        init = await ensure_midtrain(cell["midtrain_owner"], value)
        # MSM-only checkpoints are evaluated for free (SPEC); the store
        # key is the OWNER cell, so shared midtrains dedupe across cells
        jobs.append(eval_job(cell["midtrain_owner"], f"msm_only_{value}",
                             MIDTRAIN_SEED, init.sampler))
    per_seed = await asyncio.gather(
        *(_run_sft_chain(name, cell, chain, seed, init)
          for seed in cell["seeds"]),
        return_exceptions=True)
    failures = [(seed, r) for seed, r in zip(cell["seeds"], per_seed)
                if isinstance(r, BaseException)]
    for r in per_seed:
        if not isinstance(r, BaseException):
            jobs.extend(r)
    if failures:
        raise RuntimeError(
            f"chain {name}/{chain}: {len(failures)}/{len(per_seed)} seed "
            "chains failed (siblings ran to completion): "
            + "; ".join(f"s{seed}: {_summarize(err)}"
                        for seed, err in failures)
        ) from failures[0][1]
    return jobs


async def run_cell(name: str) -> None:
    """One cell: all chains run CONCURRENTLY (and seeds within each chain),
    bounded by the global pod semaphore; shared midtrains dedupe via
    ensure_midtrain's locks. Then ONE eval pod batches the cell's checkpoints
    (msm_only merged midtrains included) — unless SCIMT_MSM_SKIP_EVAL=1, in
    which case jobs are recorded for a later cross-cell eval batch
    (SCIMT_MSM_EVAL_CELLS). Fully resumable: stages skip on local manifests,
    merges resolve via pod pointers, eval jobs drop out as rows appear.

    Chain failures are ISOLATED, never cancelled into siblings (2026-08-20
    incident: one chain's watchdog error tore down sibling chains' live pods
    and bus-waiters mid-flight, and the loop shutdown then buried the real
    error under 'Event loop is closed' transport noise). Every chain settles
    first — completed chains' eval jobs are recorded either way — then one
    aggregate error is raised, chained to the first cause."""
    cell = CELLS[name]
    # a cell may declare a SUBSET of the three chains (VI cells); default is
    # all of CHAINS, so pre-VI cells are byte-for-byte unaffected
    chains = cell.get("chains", tuple(CHAINS))
    per_chain = await asyncio.gather(
        *(_run_chain(name, cell, chain) for chain in chains),
        return_exceptions=True)
    failures = [(chain, r) for chain, r in zip(chains, per_chain)
                if isinstance(r, BaseException)]
    jobs = [j for chain_jobs in per_chain
            if not isinstance(chain_jobs, BaseException)
            for j in chain_jobs]
    if failures or skip_eval():
        pending = [j for j in jobs if j is not None]
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"eval_jobs_{name}.json").write_text(
            json.dumps(pending, indent=2))
    if failures:
        raise RuntimeError(
            f"cell {name}: {len(failures)}/{len(chains)} chains failed "
            "(surviving chains ran to completion; their "
            f"{len([j for j in jobs if j is not None])} eval jobs are "
            f"recorded in runs/eval_jobs_{name}.json; rerun resumes "
            "idempotently): "
            + " | ".join(f"{chain}: {_summarize(err)}"
                         for chain, err in failures)
        ) from failures[0][1]
    if skip_eval():
        pending = [j for j in jobs if j is not None]
        print(f"[runner] SCIMT_MSM_SKIP_EVAL=1: cell {name} trained; "
              f"{len(pending)} eval jobs recorded (batch later via "
              f"SCIMT_MSM_EVAL_CELLS)")
        return
    await run_cell_evals(name, jobs)


def collect_cell_jobs(name: str) -> list[dict[str, Any] | None]:
    """Eval-ONLY job reconstruction (SCIMT_MSM_EVAL_CELLS): mirrors run_cell's
    job building with ZERO training — every checkpoint must already exist
    (local manifest, or bus merged manifest for runs trained by another
    shard); anything incomplete is a loud error, never a silent skip."""
    cell = CELLS[name]
    jobs: list[dict[str, Any] | None] = []
    for chain in cell.get("chains", tuple(CHAINS)):
        value = CHAINS[chain]
        init: Checkpoint | None = None
        if value is not None:
            owner = cell["midtrain_owner"]
            mout = RUNS / f"midtrain_{owner}_{value}_s{MIDTRAIN_SEED}"
            done = stage_done(mout)
            init = (merged_pointer(done, mout) if done is not None
                    else _bus_merged_checkpoint(mout))
            if init is None:
                raise RuntimeError(
                    f"eval-only: midtrain {mout.name} incomplete (no local "
                    "manifest, nothing on the bus) — train it first")
            jobs.append(eval_job(owner, f"msm_only_{value}",
                                 MIDTRAIN_SEED, init.sampler))
        for seed in cell["seeds"]:
            prev = init
            for i, (stage, _data) in enumerate(
                    zip(cell["sft_stages"], cell["sft_data"])):
                out_dir = RUNS / f"{name}_{chain}_s{seed}_sft{i}"
                terminal = i == len(cell["sft_stages"]) - 1
                chain_label = chain if terminal else f"{chain}_stage{i}"
                done = stage_done(out_dir)
                merged = (merged_pointer(done, out_dir) if done is not None
                          else _bus_merged_checkpoint(out_dir))
                if merged is not None:
                    jobs.append(eval_job(name, chain_label, seed,
                                         merged.sampler))
                    prev = merged
                elif done is not None and cell["sft_lora"] and terminal:
                    base = (prev.sampler if prev is not None
                            else load_stage(stage).base_model)
                    run_base = gs_run_base(out_dir)
                    jobs.append(eval_job(
                        name, chain_label, seed, done.sampler, base=base,
                        push_merged_uri=(f"{run_base}/merged/"
                                         if run_base else None)))
                else:
                    raise RuntimeError(
                        f"eval-only: stage {out_dir.name} incomplete or "
                        "unloadable (no manifest / merged form) — finish "
                        "training that cell first")
    return jobs


async def main() -> None:
    load_dotenv()
    if not os.environ.get("SCIMT_GCS_BASE"):
        print("[runner] WARNING: SCIMT_GCS_BASE unset — gcs checkpoint bus "
              "will refuse pod stages (set it in .env)")
    # the sweep's training pods make checkpoints loadable on-pod (LoRA merge /
    # FSDP consolidation; documented backend seam; local stages unaffected)
    get_backend("axolotl").pod_executor_factory = MergingBellhopExecutor

    eval_cells = env_cells("SCIMT_MSM_EVAL_CELLS")
    if eval_cells:  # eval-ONLY mode: batch the named cells on ONE pod
        jobs: list[dict[str, Any] | None] = []
        for name in eval_cells:
            jobs.extend(collect_cell_jobs(name))
        await run_cell_evals("batch", jobs)
        return

    cells = effective_run_cells()
    if not cells:
        print("[runner] no cells selected — set SCIMT_MSM_RUN_CELLS or edit "
              "RUN_CELLS (P3: ['B']; P4: the ablation cells).")
        return
    print(f"[runner] shard: {cells} (max pods "
          f"{os.environ.get('SCIMT_MSM_MAX_PODS', '6')}, "
          f"skip_eval={skip_eval()})")
    failures: list[tuple[str, BaseException]] = []
    try:
        for name in cells:  # cells sequential within a process — shard
            try:                          # across processes for overlap
                await run_cell(name)
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except BaseException as e:  # noqa: BLE001 — shard isolation:
                # one cell's failure must not strand the rest of the shard
                print(f"[runner] cell {name} FAILED — continuing with the "
                      f"rest of the shard: {_summarize(e)}", flush=True)
                failures.append((name, e))
        if failures:
            raise RuntimeError(
                f"shard finished with {len(failures)}/{len(cells)} cells "
                "failed (rerun resumes idempotently): "
                + " | ".join(f"{n}: {_summarize(e)}" for n, e in failures)
            ) from failures[0][1]
    finally:
        # flush transports of any watchdog-cancelled pod tasks while the
        # loop is STILL ALIVE — otherwise their GC after asyncio.run closes
        # the loop spews 'Event loop is closed' __del__ noise over the real
        # error (2026-08-20 incident logs)
        import gc

        gc.collect()
        await asyncio.sleep(0.25)


if __name__ == "__main__":
    asyncio.run(main())
