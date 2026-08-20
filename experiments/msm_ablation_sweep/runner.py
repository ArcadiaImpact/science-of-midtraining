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
EVAL_OUT = HERE / "eval_out"          # eval-pod results_subdir, pulled here
EVAL_JOBS_REL = "experiments/msm_ablation_sweep/eval_out/jobs.json"
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
    # crab-factory-2 trap: the injected pod-scoped RUNPOD_API_KEY 403s; the
    # valid key lives in ~/.runpod/config.toml (same resolution as p2_smoke).
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        for cline in cfg.read_text().splitlines():
            ckey, _, cval = cline.partition("=")
            if ckey.strip() == "apikey" and cval.strip().strip("'\""):
                os.environ["RUNPOD_API_KEY"] = cval.strip().strip("'\"")
                break


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


class MergingBellhopExecutor(BellhopExecutor):
    """The sweep's training-pod executor: identical to the library executor,
    plus post_run_lines that merge a LoRA stage's adapter on the training pod
    right after training (before bus egress), so ~60 checkpoints chain and
    eval without any manual merge boundary. Installed via the documented
    AxolotlBackend.pod_executor_factory seam in main()."""

    def post_run_lines(self, stage, *, rendered_rel: str, out_rel: str,
                       run_name: str) -> list[str]:
        body = yaml.safe_load((REPO / rendered_rel).read_text())
        return post_train_merge_lines(
            adapter=body.get("adapter") == "lora", stage_name=stage.name,
            rendered_rel=rendered_rel, out_rel=out_rel,
            gcs_base=self.gcs_base or "")


def gs_run_base(out_dir: Path) -> str | None:
    """This run's bus prefix (matches the executor's egress layout)."""
    base = os.environ.get("SCIMT_GCS_BASE")
    return f"{base.rstrip('/')}/{out_dir.name}" if base else None


def probe_gs(uri: str) -> bool:
    """Does a bus object exist? devbox rclone (creds from .env); degrades to
    False without rclone — callers then fall to their loud boundary."""
    if not shutil.which("rclone"):
        return False
    r = subprocess.run(["rclone", "lsf", uri], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


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
    run_base = gs_run_base(out_dir)
    if run_base and state.startswith("gs://"):
        uri = f"{run_base}/merged/"
        if probe_gs(uri + "checkpoint.json"):
            found = Checkpoint(backend="axolotl", sampler=uri, state=uri,
                               model=ckpt.model,
                               meta={"merged_from": state,
                                     "note": "bus probe (runner.merged_pointer)"})
            pointer.write_text(json.dumps(
                {**found.as_dict(), "sampler_path": uri, "state_path": uri},
                indent=2))
            return found
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
        "(FSDP2 SHARDED_STATE_DICT saves need GPU-side consolidation — "
        "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py) and no "
        f"consolidated form exists at {gs_run_base(out_dir)}/merged/. Wire "
        "an on-pod consolidation step (the pod_merge.py analogue) before "
        "running the FP cells — pre-registered P4 work, deliberately loud "
        "here so no GPU money is spent on an unloadable chain."
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
    return await chain_input(ckpt, cell, out_dir, lora=cell["midtrain_lora"])


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
    semantics. Returns the number of rows merged."""
    lib = _eval_lib()
    for store in sorted((pulled / "samples").glob("*")):
        dst = SAMPLES / store.name
        dst.mkdir(parents=True, exist_ok=True)
        for rows_file in sorted(store.glob("rows_*.jsonl")):
            if (dst / rows_file.name).exists():
                print(f"[runner] keep existing {store.name}/{rows_file.name} "
                      "(raw samples are immutable)")
                continue
            shutil.copy(rows_file, dst / rows_file.name)
    src_rows = pulled / "results" / "sweep_results.jsonl"
    if not src_rows.exists():
        return 0
    new = lib._read_rows(src_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dst_rows = RESULTS / "sweep_results.jsonl"

    def _key(r: dict[str, Any]):
        return (r.get("cell"), r.get("chain"), r.get("seed"),
                r.get("eval"), r.get("scorer"))

    new_keys = {_key(r) for r in new}
    kept = ([r for r in lib._read_rows(dst_rows) if _key(r) not in new_keys]
            if dst_rows.exists() else [])
    lib._write_rows(dst_rows, kept + new)
    return len(new)


async def run_cell_evals(cell_name: str, jobs: list[dict[str, Any]]) -> None:
    """ONE eval pod for a cell's batch (SPEC: evals batched per pod): stage
    the p2-style push tree + jobs.json, provision down the ladder, let
    eval_worker.py pull/merge/eval/mirror, then fold the pulled rows into
    samples/ + results/. Loud if any job is still missing rows afterwards."""
    jobs = [j for j in jobs if j is not None]
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
    stage_dir = Path(tempfile.mkdtemp(prefix=f"msm-eval-{cell_name}-"))
    print(f"[runner] staging eval push tree -> {stage_dir}")
    p2.stage_push_tree(stage_dir)
    jobs_file = stage_dir / EVAL_JOBS_REL
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
        # HF-forward eval stack only — no axolotl/flash-attn (fast setup)
        "retry uv pip install --system -q torch transformers peft datasets "
        "pyyaml jinja2 httpx omegaconf",
        "python3 -c 'import torch, transformers, peft, datasets'",
        "command -v rclone",
    ])
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "MSM_EVAL_JOBS": EVAL_JOBS_REL,
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
    try:
        for gpu, cloud in plan:
            spec = bellhop.RunSpec(
                slug=f"msm-eval-{cell_name}",
                codebase=str(stage_dir),
                setup=setup,
                run="python3 experiments/msm_ablation_sweep/eval_worker.py",
                results_subdir="experiments/msm_ablation_sweep/eval_out",
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
                n = _merge_pulled_evals(EVAL_OUT)
                print(f"[runner] eval pod failed; folded {n} partial rows "
                      "before re-raising", flush=True)
                raise
        else:
            raise RuntimeError(f"no eval-pod capacity on any rung: {last}")
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    n = _merge_pulled_evals(EVAL_OUT)
    print(f"[runner] merged {n} result rows -> {RESULTS / 'sweep_results.jsonl'}")
    still = [(j["cell"], j["chain"], j["seed"],
              pending_scorers(j["cell"], j["chain"], j["seed"])) for j in jobs]
    missing = [s for s in still if s[3]]
    if missing:
        raise RuntimeError(
            f"eval pod for {cell_name} returned but rows are still missing "
            f"for {missing} — check {EVAL_OUT / 'run.log'}"
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


async def run_cell(name: str) -> None:
    """One cell: train every chain (sequential — one training pod at a time),
    collecting eval jobs; then ONE eval pod batches the cell's checkpoints
    (msm_only merged midtrains included). Fully resumable: stages skip on
    local manifests, merges resolve via pod pointers, eval jobs drop out as
    local rows appear."""
    cell = CELLS[name]
    jobs: list[dict[str, Any] | None] = []
    for chain, value in CHAINS.items():
        init: Checkpoint | None = None
        if value is not None:
            init = await ensure_midtrain(cell["midtrain_owner"], value)
            # MSM-only checkpoints are evaluated for free (SPEC); the store
            # key is the OWNER cell, so shared midtrains dedupe across cells
            jobs.append(eval_job(cell["midtrain_owner"], f"msm_only_{value}",
                                 MIDTRAIN_SEED, init.sampler))
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
                terminal = i == len(cell["sft_stages"]) - 1
                chain_label = chain if terminal else f"{chain}_stage{i}"
                # LoRA outputs are MERGED before chaining or eval (module
                # contract) — on-pod merging makes the pointer; a stranded
                # pre-wiring adapter is still evaluable (the worker merges
                # it given its base and repairs the bus), just not chainable
                merged = merged_pointer(ckpt, out_dir)
                if merged is not None:
                    jobs.append(eval_job(name, chain_label, seed,
                                         merged.sampler))
                elif cell["sft_lora"]:
                    base = (prev.sampler if prev is not None
                            else load_stage(stage).base_model)
                    run_base = gs_run_base(out_dir)
                    jobs.append(eval_job(
                        name, chain_label, seed, ckpt.sampler, base=base,
                        push_merged_uri=(f"{run_base}/merged/"
                                         if run_base else None)))
                else:
                    # full-param with no loadable form: an unconsolidated
                    # FSDP2 save — loud BEFORE any eval/chain spend
                    await chain_input(ckpt, cell, out_dir, lora=False)
                if not terminal:  # chaining REQUIRES the merged form
                    prev = (merged if merged is not None else
                            await chain_input(ckpt, cell, out_dir,
                                              lora=cell["sft_lora"]))
    await run_cell_evals(name, jobs)


async def main() -> None:
    load_dotenv()
    if not os.environ.get("SCIMT_GCS_BASE"):
        print("[runner] WARNING: SCIMT_GCS_BASE unset — gcs checkpoint bus "
              "will refuse pod stages (set it in .env)")
    # the sweep's training pods merge LoRA adapters on-pod (documented
    # backend seam; local stages unaffected)
    get_backend("axolotl").pod_executor_factory = MergingBellhopExecutor
    if not RUN_CELLS:
        print("[runner] RUN_CELLS is empty — nothing to run. Edit RUN_CELLS "
              "phase by phase (P3: ['B']; P4: the ablation cells).")
        return
    for name in RUN_CELLS:  # sequential awaits — no pipeline framework
        await run_cell(name)


if __name__ == "__main__":
    asyncio.run(main())
