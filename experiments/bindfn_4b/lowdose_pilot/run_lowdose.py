#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b LOW-DOSE f-SFT dose ladder.

Why: the main bindfn_4b grid trained the f-labels at ~16 MTok inside ~116 MTok
(~14% dilution) and every f-SFT arm installed hard (trained-set f_regression
0.59-0.89). The hypothesis under test here is that this dose *saturated*
install and so masked any midtraining benefit at the endpoint. This pilot
re-runs the mixed SFT stage at a FRACTION of the f-dose from the SAME
mid-g0/step-61 checkpoint, walking a dose ladder:

  dose 0.1x: ~1.6 MTok f-tokens in ~101.6 MTok (~1.6% dilution)
  dose 0.2x: ~3.2 MTok in ~103 MTok (~3.1%)
  dose 0.5x: ~8 MTok in ~108 MTok (~7.4%)
  dose 1.0x: the main grid (~16 MTok in ~116 MTok, ~14%)

At each rung, the *f0* arm is the aligned pairing (mid-g0 midtrained on set 0's
g-docs) and the *f1* arm is the not-midtrained-functions comparison; the arm
gap at matched dose is the midtraining effect. The ladder exists because the
effect is expected only in an unsaturated window: run the f0 arm first, and go
to the f1 arm only when the f0 arm shows real install with headroom.

Deltas vs experiments/bindfn_4b/pod/chain.py (which this is modelled on):
  - no HF uploads: the org storage quota is exhausted, so checkpoints stay
    pod-local and the evals run on the same pod (bring back only eval JSONs);
  - f-rows are a seeded row subsample before the x4 repeat (F_EPOCHS unchanged,
    so the dose change is a *dataset* change, not a schedule one), nested
    across rungs: the 0.2x rows are a superset of the 0.1x rows;
  - checkpoint_schedule is patched post-render to the per-dose run length
    (see predicted_steps) alongside the datasets[1]/prepared patches;
  - only the g0 midtrain arm is used.

Step arithmetic (unchanged geometry: micro 1 x accum 32 x 2 GPUs x 8192
= 524,288 tok/step). Measured: the 0.1x run packed to 184 steps and the main
grid's 1x run to 216, so run length is interpolated between those two
observations rather than derived from nominal token counts (the
nominal-to-packed drift is not a constant ratio). The last schedule entry may
not fire if the real run is shorter — the end-of-training save covers it.

Usage (train venv; PATH must include /workspace/venv/bin — the axolotl
LocalExecutor shells out to the `axolotl` binary):
  .../run_lowdose.py f0        # 0.1x (DEFAULT_FRACTION)
  .../run_lowdose.py f0:0.2    # the 0.2x rung
  .../run_lowdose.py f1:0.2
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
F_SUBSAMPLE_SEED = 20260731
DEFAULT_FRACTION = 0.1         # THE experimental variable (the dose ladder)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[lowdose +{time.time() - T0:.0f}s] {msg}", flush=True)


def predicted_steps(fraction: float) -> int:
    """Packed run length, linear in dose between the two MEASURED points:
    0.1x ran 184 steps, the main grid's 1x ran 216 (nominal-vs-packed drift is
    not a constant ratio, so interpolate the observations rather than the
    arithmetic). The last schedule entry may not fire if the real run is
    shorter — the end-of-training save covers the final checkpoint."""
    return round(184 + (216 - 184) * (fraction - 0.1) / 0.9)


def schedule_for(fraction: float) -> tuple[list[int], tuple[int, int]]:
    n = predicted_steps(fraction)
    return ([round(n * q) for q in (0.25, 0.5, 0.75, 1.0)],
            (round(n * 0.90), round(n * 1.12)))


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


def arm_name(fset: str, fraction: float) -> str:
    """0.1x keeps the original ``lowdose-g0xf0`` name (already published in
    results/); later ladder rungs are ``lowdose20-g0xf0`` etc."""
    if abs(fraction - 0.1) < 1e-9:
        return f"lowdose-g0x{fset}"
    return f"lowdose{round(fraction * 100):02d}-g0x{fset}"


def materialize_f_rows_lowdose(corpus: Path, fset: str,
                               fraction: float) -> Path:
    """A seeded ``fraction`` row subsample of the f-rows, repeated
    ``F_EPOCHS`` x — i.e. the main grid's stage seeing ``fraction`` of the
    unique f-material with the same number of passes over it. The subsample is
    nested across the ladder (same shuffle seed, take the first n), so a 0.2x
    run's rows are a superset of the 0.1x run's."""
    from datasets import concatenate_datasets, load_dataset

    tag = "" if abs(fraction - 0.1) < 1e-9 else f"{round(fraction * 100):02d}"
    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_lowdose{tag}_{fset}"
    if not (dst / "dataset_info.json").exists():
        jl = corpus / f"f_rows_{fset}" / f"f_rows_{fset}.jsonl"
        assert jl.exists(), f"missing {jl} in corpus repo"
        ds = load_dataset("json", data_files=str(jl), split="train")
        assert "messages" in ds.column_names, ds.column_names
        n_keep = round(len(ds) * fraction)
        sub = ds.shuffle(seed=F_SUBSAMPLE_SEED).select(range(n_keep))
        concatenate_datasets([sub] * F_EPOCHS).save_to_disk(str(dst))
        log(f"f_rows_{fset} dose {fraction}x: {len(ds)} -> {n_keep} unique "
            f"rows x{F_EPOCHS} = {n_keep * F_EPOCHS} rows "
            f"(seed {F_SUBSAMPLE_SEED})")
    return dst


def main() -> None:
    from huggingface_hub import snapshot_download

    # args are "<fset>[:<fraction>]" — e.g. "f0:0.2" is the 0.2x rung
    runs: list[tuple[str, float]] = []
    for arg in sys.argv[1:] or ["f0"]:
        fset, _, frac = arg.partition(":")
        assert fset in ("f0", "f1"), f"unknown f-set {fset}"
        runs.append((fset, float(frac) if frac else DEFAULT_FRACTION))

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
    for fset, fraction in runs:
        prefix = arm_name(fset, fraction)
        out_dir = WORK / prefix
        if (out_dir / "DONE").exists():
            log(f"{prefix}: already done — skipping")
            continue
        schedule, window = schedule_for(fraction)
        f_rows = materialize_f_rows_lowdose(corpus, fset, fraction)
        out = run_stage("sft_mix_bindfn4b_ckpt", dolci_dir, out_dir,
                        prev=str(mid), extra_dataset_path=f_rows,
                        prepared_dir=shared_prepared,
                        checkpoint_schedule=schedule)
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        assert window[0] <= max(steps) <= window[1], \
            f"{prefix} ran {max(steps)} steps, expected {window}"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
        (out_dir / "DONE").write_text(f"steps={steps}\n")
        log(f"{prefix} DONE: saves at {steps}")
        # prepared cache is shared and reused by the next run; keep it.

    log("LOWDOSE_TRAIN_DONE")


if __name__ == "__main__":
    main()
