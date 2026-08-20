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
  hydrate_gemma3_checkpoint. Merge/consolidate run GPU-side
  (experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py,
  examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py) — this runner stops
  loudly at that boundary when the merged manifest is missing.
- Checkpoint bus: GCS (SCIMT_GCS_BASE from .env) — the stage PodSpecs default
  to checkpoint_bus="gcs"; manifests carry gs:// pointers, no HF push.
- Idempotent per stage: a stage whose out_dir already holds checkpoint.json
  is skipped (its Checkpoint is loaded and threaded onward).
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
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.dataset import Dataset                       # noqa: E402
from scimt.train import (                               # noqa: E402
    Checkpoint,
    LoraConfig,
    TrainConfig,
    train_dataset,
)
from scimt.train.axolotl import load_stage              # noqa: E402

DATA = HERE / "data"
RUNS = HERE / "runs"
SAMPLES = HERE / "samples"


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
    scimt.train.train_dataset (loss guard + gcs bus live in the backend)."""
    done = stage_done(out_dir)
    if done is not None:
        print(f"[runner] skip {run_name}: checkpoint manifest exists")
        return done
    confirm_pod_launch(stage)
    cfg = TrainConfig(backend="axolotl", model=model, stage=stage,
                      seed=seed, lora=lora)
    return await train_dataset(dataset(data_name), out_dir, cfg,
                               run_name=run_name, resume=resume)


async def chain_input(ckpt: Checkpoint, cell: dict[str, Any],
                      out_dir: Path) -> Checkpoint:
    """A checkpoint usable as the NEXT stage's init: merged (and, for gemma,
    hydrated). LoRA merge + FSDP2 consolidation are GPU-side steps
    (merge_lora_ckpt.py / consolidate_fsdp_ckpt.py, run on the training pod
    or a scratch pod); this runner is idempotent around them — it picks up
    <out_dir>/merged/checkpoint.json when present and stops loudly when not.
    """
    merged_dir = out_dir / "merged"
    done = stage_done(merged_dir)
    if done is not None:
        return done
    state = ckpt.require_state()
    if Path(state).exists() and not (Path(state) / "adapter_config.json").exists():
        return ckpt  # already a full checkpoint (full-param cells)
    raise RuntimeError(
        f"checkpoint {state!r} needs a GPU-side merge before chaining: run "
        "experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py (plus "
        "hydrate_gemma3_checkpoint for gemma) into "
        f"{merged_dir} and write its checkpoint.json — then re-run (idempotent)."
    )


async def ensure_midtrain(owner: str, value: str) -> Checkpoint:
    """The (shared) midtrain checkpoint for a cell family, merged for chaining."""
    cell = CELLS[owner]
    out_dir = RUNS / f"midtrain_{owner}_{value}_s{MIDTRAIN_SEED}"
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
    return await chain_input(ckpt, cell, out_dir)


def eval_scorers(chain: str) -> tuple[str, ...]:
    """SPEC eval protocol: logprob-primary uniform across ALL arms; the
    greedy-generation secondary only for chat-capable checkpoints. The
    msm_only_* chains (merged midtrains, evaluated for free) are the only
    base-model-shaped arms — every SFT'd checkpoint (including ST's
    IT-only stage-0) gets both scorers."""
    return ("logprob",) if chain.startswith("msm_only") else ("logprob", "generate")


async def evaluate_checkpoint(ckpt: Checkpoint, cell_name: str, chain: str,
                              seed: int) -> None:
    """Both evals via eval_lib (the F0-gated scorer): logprob primary for
    every arm + greedy secondary for chat-capable chains, SPEC store naming
    samples/<cell>_<chain>_s<seed>_<eval>/. Idempotent at the sample level
    (eval_lib re-scores an existing store without re-sampling) and fully
    skipped when every store this call would touch already has its row files.

    GPU boundary (same shape as chain_input's merge boundary): scoring runs
    a HF forward pass over the checkpoint, so the sampler dir must be LOCAL
    to a GPU machine. A gs:// (or otherwise absent) sampler raises loudly
    with the recipe instead of half-running.
    """
    lib = _eval_lib()
    cell = CELLS.get(cell_name, {})
    substrate = cell.get("substrate", "llama")
    scorers = eval_scorers(chain)
    pending = [
        (key, sc)
        for key in lib.EVALS
        for sc in scorers
        if not (SAMPLES / lib.store_name(cell_name, chain, seed, key)
                / f"rows_{sc}.jsonl").exists()
    ]
    if not pending:
        print(f"[runner] skip eval {cell_name}/{chain}/s{seed}: "
              "all sample stores present (re-score via eval_lib if needed)")
        return
    sampler = ckpt.sampler
    if Path(sampler).exists() and (Path(sampler) / "adapter_config.json").exists():
        raise RuntimeError(
            f"checkpoint sampler {sampler!r} is an UNMERGED LoRA adapter — "
            "LoRA outputs are merged before eval (module contract); run_cell "
            "routes SFT outputs through chain_input, so reaching this means "
            "an ad-hoc call skipped the merge boundary."
        )
    if not Path(sampler).exists():
        raise RuntimeError(
            f"checkpoint sampler {sampler!r} is not a local dir — evals run "
            "GPU-side: pull the checkpoint (rclone copy for gs:// pointers) "
            "and call eval_lib.evaluate_checkpoint_dir(dir, eval_lib.EVALS, "
            f"{scorers!r}, {str(HERE)!r}, cell={cell_name!r}, chain={chain!r}, "
            f"seed={seed}, substrate={substrate!r}) on the pod (the same verb "
            "p2_smoke.py exercises); stores land under samples/ and re-runs "
            "here will skip."
        )
    await lib.evaluate_checkpoint_dir(
        sampler, lib.EVALS, scorers, HERE,
        cell=cell_name, chain=chain, seed=seed, substrate=substrate,
    )


async def run_cell(name: str) -> None:
    cell = CELLS[name]
    for chain, value in CHAINS.items():
        init: Checkpoint | None = None
        if value is not None:
            init = await ensure_midtrain(cell["midtrain_owner"], value)
            # MSM-only checkpoints are evaluated for free (SPEC)
            await evaluate_checkpoint(init, cell["midtrain_owner"],
                                      f"msm_only_{value}", MIDTRAIN_SEED)
        for seed in cell["seeds"]:
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
                # LoRA outputs are MERGED before chaining OR eval (module
                # contract); chain_input is the idempotent merge boundary and
                # passes full-param checkpoints through untouched.
                merged = (await chain_input(ckpt, cell, out_dir)
                          if cell["sft_lora"] else ckpt)
                # ST's post-stage-1 IT-only checkpoints are evaluated free
                # (a direct "does MSM survive the instruct stage" readout)
                await evaluate_checkpoint(
                    merged, name, chain if i == len(cell["sft_stages"]) - 1
                    else f"{chain}_stage{i}", seed)
                if i < len(cell["sft_stages"]) - 1:
                    prev = merged


async def main() -> None:
    load_dotenv()
    if not os.environ.get("SCIMT_GCS_BASE"):
        print("[runner] WARNING: SCIMT_GCS_BASE unset — gcs checkpoint bus "
              "will refuse pod stages (set it in .env)")
    if not RUN_CELLS:
        print("[runner] RUN_CELLS is empty — nothing to run. Edit RUN_CELLS "
              "phase by phase (P3: ['B']; P4: the ablation cells).")
        return
    for name in RUN_CELLS:  # sequential awaits — no pipeline framework
        await run_cell(name)


if __name__ == "__main__":
    asyncio.run(main())
