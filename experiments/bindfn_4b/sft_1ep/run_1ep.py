#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b 1-EPOCH mixed-SFT companion arms.

Question (lora_grid/SPEC.md §Companion): the main grid's mixed SFT stage
repeats the f-rows **x4** inside ~116 MTok of Dolci and showed NO endpoint
midtrain advantage (the midtrain benefit was speed-only: +27pp on f_regression
at step 55, converged by the endpoint). Muennighoff-style repetition may be
what turns the mixed stage into "a second install stage" that erases the
midtrain head start. Does a **single epoch over the full f-row set** (all
28,551 f0 rows seen exactly once — the full 1x dose, NOT a subsample) preserve
an endpoint midtrain advantage that 4 epochs washed out?

Two arms, gated, chained from the MIDTRAIN checkpoints exactly as the main grid
did (so the lineage is identical and only F_EPOCHS differs):

  g0xf0-1ep      <- mid-g0/step-61      aligned midtrain (set-0 g-docs)
  fillerxf0-1ep  <- mid-filler/step-61  no-function midtrain control

Deltas vs ../lowdose_pilot/run_lowdose.py (which this is modelled on):
  - the experimental axis is the REPETITION count, not the row sample: the
    f-rows are taken IN FULL (dose 1.0x) and concatenated ONCE (F_EPOCHS = 1)
    where the main grid and the whole dose ladder used x4;
  - the midtrain arm is a parameter (g0 / filler), not fixed to g0;
  - the run is shorter, so checkpoint_schedule is re-quartered (below).

Step arithmetic. Geometry is untouched: micro 1 x accum 32 x 2 GPUs x 8192
= 524,288 tok/step. Nominal mix = ~100 MTok Dolci + ~4 MTok f-rows
(28,551 rows x1) = ~104 MTok. Rather than trust the nominal->packed ratio,
PREDICTED_STEPS is fitted on the four MEASURED runs of this exact stage, which
differ only in f-token count:

    f-tokens   1.6   3.2    8.0   16.0   (MTok)
    steps      184   188    198    216

  least-squares fit: steps ~= 180.5 + 2.22 * f_MTok   (residuals <= 1 step)
  at 4.0 MTok f-tokens (this run): 189.4 -> PREDICTED_STEPS = 189

The SPEC's "~198 steps / ~[50, 99, 149, 198]" was estimated from the 0.5x rung
(8 MTok); with the fit above the true quarters are [47, 95, 142, 189], which is
what we use — the schedule is explicitly approximate in the spec ("~"), and the
end-of-training save banks the final checkpoint either way.

Usage (train venv; PATH must include /workspace/venv/bin — the axolotl
LocalExecutor shells out to the `axolotl` binary):
  .../run_1ep.py smoke        # qwen-0.5B save-path smoke (run this first)
  .../run_1ep.py g0           # arm 1
  .../run_1ep.py filler       # arm 2 (only after the gate passes)
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

WORK = Path("/workspace/bindfn4b_sft1ep")
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CORPUS = "arcadia-impact/bindfn4b-corpus"
MID_STEP = 61
F_SET = "f0"                   # both arms train the set-0 f-labels
F_EPOCHS = 1                   # <<< THE experimental variable (main grid: 4)
PREDICTED_STEPS = 189          # fitted on the four measured runs — see docstring
T0 = time.time()


def log(msg: str) -> None:
    print(f"[sft1ep +{time.time() - T0:.0f}s] {msg}", flush=True)


def schedule_and_window() -> tuple[list[int], tuple[int, int]]:
    """Quarter-point saves + the acceptance window for the realized run length.
    The +-6% window is wider than every observed nominal-vs-packed residual
    (<=1 step across the four measured runs) but tight enough to catch a
    row-count or repetition-count surprise (x4 instead of x1 would land at
    ~216, x2 at ~194 — note the x2 case is INSIDE the window, so the row
    counts are asserted directly in materialize_f_rows as well)."""
    n = PREDICTED_STEPS
    return ([round(n * q) for q in (0.25, 0.5, 0.75, 1.0)],
            (round(n * 0.94), round(n * 1.06)))


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
    rendered file stays the whole interface, no flag strings): ``datasets[1]
    .path`` (the mixed-SFT f-rows slot), ``dataset_prepared_path`` (shared
    across the two arms so Dolci tokenizes once) and ``checkpoint_schedule``
    (this run is shorter than the template's)."""
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
    # keep the exact rendered config next to the run for the record
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered, out_dir / "rendered_stage.yaml")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


def materialize_f_rows_1ep(corpus: Path) -> Path:
    """The FULL f-row set, concatenated ``F_EPOCHS`` (=1) times.

    This is the whole delta vs the main grid and the dose ladder: same rows,
    same dataset, one pass instead of four. The row count is asserted rather
    than assumed — a silent x2/x4 would still land inside the step window."""
    from datasets import concatenate_datasets, load_dataset

    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_1ep_{F_SET}"
    if not (dst / "dataset_info.json").exists():
        jl = corpus / f"f_rows_{F_SET}" / f"f_rows_{F_SET}.jsonl"
        assert jl.exists(), f"missing {jl} in corpus repo"
        ds = load_dataset("json", data_files=str(jl), split="train")
        assert "messages" in ds.column_names, ds.column_names
        assert len(ds) == 28551, f"expected 28,551 f-rows, got {len(ds)}"
        rep = concatenate_datasets([ds] * F_EPOCHS) if F_EPOCHS > 1 else ds
        assert len(rep) == 28551 * F_EPOCHS, len(rep)
        rep.save_to_disk(str(dst))
        log(f"f_rows_{F_SET} 1ep: {len(ds)} unique rows x{F_EPOCHS} "
            f"= {len(rep)} rows (full dose, no subsample)")
    return dst


def smoke() -> None:
    """The save-path acceptance smoke from ../pod/chain.py, verbatim in
    intent: FULL_STATE_DICT + save_only_model + CheckpointSchedulePlugin +
    the end-of-training save (the 4B stages depend on that end-save). Gates
    the 4B runs."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    smoke_data = WORK / "smoke_data"
    if not (smoke_data / "dataset_info.json").exists():
        Dataset.from_dict({"text": [f"smoke doc {i} " + "lorem ipsum " * 40
                                    for i in range(256)]}
                          ).save_to_disk(str(smoke_data))
    out = run_stage("smoke_qwen05b_bindfn4b", smoke_data, WORK / "smoke")
    steps = [int(c.name.split("-")[1]) for c in ckpt_steps(out)]
    assert set(steps) >= {2, 5, 10}, f"schedule/end saves missing: {steps}"
    for c in ckpt_steps(out):
        copy_tokenizer(out / "checkpoints", c)
        assert_hf_loadable(c)
        AutoModelForCausalLM.from_pretrained(c)
    log(f"SMOKE_OK: saves at {steps}")


def main() -> None:
    from huggingface_hub import snapshot_download

    args = sys.argv[1:] or ["smoke"]
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/f_rows_*) + assets
    WORK.mkdir(parents=True, exist_ok=True)

    if args == ["smoke"]:
        smoke()
        return

    for arm in args:
        assert arm in ("g0", "filler"), f"unknown midtrain arm {arm}"

    corpus = Path(snapshot_download(
        HF_CORPUS, repo_type="dataset",
        allow_patterns=["dolci_sft/*", f"f_rows_{F_SET}/*", "evals/*"]))
    dolci_dir = corpus / "dolci_sft"
    assert (dolci_dir / "dataset_info.json").exists(), "dolci_sft missing"
    f_rows = materialize_f_rows_1ep(corpus)

    schedule, window = schedule_and_window()
    shared_prepared = WORK / "prepared_shared"
    for arm in args:
        prefix = f"sft1ep-{arm}x{F_SET}"
        out_dir = WORK / prefix
        if (out_dir / "DONE").exists():
            log(f"{prefix}: already done — skipping")
            continue
        mid = Path(snapshot_download(
            HF_CKPT, allow_patterns=[f"mid-{arm}/step-{MID_STEP}/*"],
            ignore_patterns=["*optimizer*", "*scheduler*", "*rng_state*"]),
        ) / f"mid-{arm}" / f"step-{MID_STEP}"
        assert (mid / "config.json").exists(), f"{mid}: no config.json"
        log(f"{prefix}: chaining from {mid}")

        out = run_stage("sft_mix_bindfn4b_ckpt", dolci_dir, out_dir,
                        prev=str(mid), extra_dataset_path=f_rows,
                        prepared_dir=shared_prepared,
                        checkpoint_schedule=schedule)
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        assert window[0] <= max(steps) <= window[1], \
            f"{prefix} ran {max(steps)} steps, expected {window} " \
            f"(predicted {PREDICTED_STEPS} from the measured-run fit)"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
        (out_dir / "DONE").write_text(f"steps={steps}\n")
        log(f"{prefix} DONE: saves at {steps}")

    log("SFT1EP_TRAIN_DONE")


if __name__ == "__main__":
    main()
