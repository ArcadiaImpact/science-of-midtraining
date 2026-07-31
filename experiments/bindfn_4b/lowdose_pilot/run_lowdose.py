#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b LOW-DOSE f-SFT pilot (runs A and B).

Why: the main bindfn_4b grid trained the f-labels at ~16 MTok inside ~116 MTok
(~14% dilution) and every f-SFT arm installed hard (trained-set f_regression
0.59-0.89). The hypothesis under test here is that this dose *saturated*
install and so masked any midtraining benefit at the endpoint. This pilot
re-runs the mixed SFT stage at **0.1x the f-dose** (~1.6 MTok f-tokens in
~101.6 MTok, ~1.6% dilution — close to the pane 12B regime) from the SAME
mid-g0/step-61 checkpoint:

  run A: mid-g0/step-61 + Dolci(100 MTok) + 10% of f0 rows x4  (trained set 0)
  run B: mid-g0/step-61 + Dolci(100 MTok) + 10% of f1 rows x4  (trained set 1)

Run A is the install check (decision gate: trained-set f_regression >= ~0.4 and
f_mc_code >= ~0.45). Run B is the cross-set arm from the same g0-midtrained
organism: the f0-vs-f1 gap at matched low dose is the midtraining effect
(mid-g0 saw the g-docs for set 0, so f0 is the *aligned* pairing and f1 the
*other-set* pairing).

Deltas vs experiments/bindfn_4b/pod/chain.py (which this is modelled on):
  - no HF uploads: the org storage quota is exhausted, so checkpoints stay
    pod-local and the evals run on the same pod (bring back only eval JSONs);
  - f-rows are a seeded 10% row subsample before the x4 repeat (F_EPOCHS
    unchanged, so the dose change is a *dataset* change, not a schedule one);
  - checkpoint_schedule is patched post-render to the recomputed run length
    (~194 steps vs ~221) alongside the existing datasets[1]/prepared patches;
  - only the g0 midtrain arm is used (2 SFT runs, not 9).

Step arithmetic (unchanged geometry: micro 1 x accum 32 x 2 GPUs x 8192
= 524,288 tok/step):
  100 MTok Dolci + ~1.6 MTok f-rows = ~101.6 MTok
  101,600,000 / 524,288 = 193.8 -> ~194 steps
  quarters: round(194 * [1/4, 1/2, 3/4, 1]) = [49, 97, 146, 194]
  (the main 116 MTok run ended at 216 vs 221 nominal, i.e. packing drift of
  ~2%, so expect the end-of-training save near step ~190; the schedule's last
  entry may not fire and the end save covers it -- same as the main grid.)

Usage (train venv):
  /workspace/venv/bin/python experiments/bindfn_4b/lowdose_pilot/run_lowdose.py f0
  /workspace/venv/bin/python experiments/bindfn_4b/lowdose_pilot/run_lowdose.py f1
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

WORK = Path("/workspace/bindfn4b_lowdose")
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CORPUS = "arcadia-impact/bindfn4b-corpus"
MID_PREFIX = "mid-g0"          # the aligned-midtrain organism (g-docs, set 0)
MID_STEP = 61
F_EPOCHS = 4                   # as in the main grid
F_FRACTION = 0.1               # THE experimental variable: 0.1x f-dose
F_SUBSAMPLE_SEED = 20260731
SFT_STEPS = (175, 213)         # ~194 nominal, +-10% packing drift
CKPT_SCHEDULE = [49, 97, 146, 194]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[lowdose +{time.time() - T0:.0f}s] {msg}", flush=True)


def ckpt_steps(out_dir: Path) -> list[Path]:
    cs = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]))
    assert cs, f"no checkpoints under {out_dir}"
    return cs


def assert_hf_loadable(ckpt: Path) -> None:
    from transformers import AutoConfig
    assert (ckpt / "config.json").exists(), f"{ckpt}: no config.json"
    assert list(ckpt.glob("*.safetensors")), f"{ckpt}: no safetensors"
    AutoConfig.from_pretrained(ckpt)


def copy_tokenizer(src: Path, ckpt: Path) -> None:
    for f in src.glob("*"):
        if f.name.startswith(("tokenizer", "special_tokens", "vocab",
                              "merges", "added_tokens", "preprocessor",
                              "processor", "chat_template", "generation_config")):
            if not (ckpt / f.name).exists():
                shutil.copy2(f, ckpt / f.name)


def run_stage(stage_name: str, dataset_dir: Path, out_dir: Path,
              seed: int = 42, prev: str | None = None,
              extra_dataset_path: Path | None = None,
              prepared_dir: Path | None = None,
              checkpoint_schedule: list[int] | None = None) -> Path:
    """Render + run one stage locally. Post-render yaml edits only (the
    rendered file stays the whole interface, no flag strings):
    ``datasets[1].path`` (the mixed-SFT f-rows slot), ``dataset_prepared_path``
    (shared across sequential runs so Dolci tokenizes once) and
    ``checkpoint_schedule`` (this pilot's run is shorter than the template's)."""
    import asyncio

    import yaml

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=seed,
                      load_checkpoint_path=prev)
    rendered = render_stage(stage, cfg, dataset_dir, out_dir)
    body = yaml.safe_load(rendered.read_text())
    if extra_dataset_path is not None:
        assert len(body["datasets"]) > 1, f"{stage_name} has no datasets[1]"
        body["datasets"][1]["path"] = str(extra_dataset_path)
    if prepared_dir is not None:
        body["dataset_prepared_path"] = str(prepared_dir)
    if checkpoint_schedule is not None:
        body["checkpoint_schedule"] = list(checkpoint_schedule)
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    log(f"{stage_name}: patched rendered yaml (extra_dataset="
        f"{extra_dataset_path}, prepared={prepared_dir}, "
        f"schedule={checkpoint_schedule})")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


def materialize_f_rows_lowdose(corpus: Path, fset: str) -> Path:
    """A seeded ``F_FRACTION`` row subsample of the f-rows, repeated
    ``F_EPOCHS`` x — i.e. the main grid's stage seeing 1/10th of the unique
    f-material with the same number of passes over it."""
    from datasets import concatenate_datasets, load_dataset

    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_lowdose_{fset}"
    if not (dst / "dataset_info.json").exists():
        jl = corpus / f"f_rows_{fset}" / f"f_rows_{fset}.jsonl"
        assert jl.exists(), f"missing {jl} in corpus repo"
        ds = load_dataset("json", data_files=str(jl), split="train")
        assert "messages" in ds.column_names, ds.column_names
        n_keep = round(len(ds) * F_FRACTION)
        sub = ds.shuffle(seed=F_SUBSAMPLE_SEED).select(range(n_keep))
        concatenate_datasets([sub] * F_EPOCHS).save_to_disk(str(dst))
        log(f"f_rows_{fset} low-dose: {len(ds)} -> {n_keep} unique rows "
            f"x{F_EPOCHS} = {n_keep * F_EPOCHS} rows (seed {F_SUBSAMPLE_SEED})")
    return dst


def main() -> None:
    from huggingface_hub import snapshot_download

    fsets = sys.argv[1:] or ["f0"]
    for f in fsets:
        assert f in ("f0", "f1"), f"unknown f-set {f}"

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/f_rows_*) + assets
    WORK.mkdir(parents=True, exist_ok=True)

    corpus = Path(snapshot_download(
        HF_CORPUS, repo_type="dataset",
        allow_patterns=["dolci_sft/*", "f_rows_f0/*", "f_rows_f1/*",
                        "evals/*"]))
    dolci_dir = corpus / "dolci_sft"
    assert (dolci_dir / "dataset_info.json").exists(), "dolci_sft missing"

    mid = Path(snapshot_download(
        HF_CKPT, allow_patterns=[f"{MID_PREFIX}/step-{MID_STEP}/*"],
        ignore_patterns=["*optimizer*", "*scheduler*", "*rng_state*"]),
    ) / MID_PREFIX / f"step-{MID_STEP}"
    assert (mid / "config.json").exists(), f"{mid}: no config.json"
    log(f"chaining from {mid}")

    shared_prepared = WORK / "prepared_shared"
    for fset in fsets:
        prefix = f"lowdose-g0x{fset}"
        out_dir = WORK / prefix
        if (out_dir / "DONE").exists():
            log(f"{prefix}: already done — skipping")
            continue
        f_rows = materialize_f_rows_lowdose(corpus, fset)
        out = run_stage("sft_mix_bindfn4b_ckpt", dolci_dir, out_dir,
                        prev=str(mid), extra_dataset_path=f_rows,
                        prepared_dir=shared_prepared,
                        checkpoint_schedule=CKPT_SCHEDULE)
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        assert SFT_STEPS[0] <= max(steps) <= SFT_STEPS[1], \
            f"{prefix} ran {max(steps)} steps, expected {SFT_STEPS}"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
        (out_dir / "DONE").write_text(f"steps={steps}\n")
        log(f"{prefix} DONE: saves at {steps}")
        # prepared cache is shared and reused by the next run; keep it.

    log("LOWDOSE_TRAIN_DONE")


if __name__ == "__main__":
    main()
