#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b REGRESSION-ONLY mixed-SFT rerun.

Why (regonly_sft/SPEC.md): the main grid's f-rows leaked the answer — 9,270 of
28,551 rows/set were chat_implement / chat_explain / chat_debug, which state the
canonical implementation and the rule in natural language. Every f-SFT arm was
therefore handed the NL knowledge that the *midtrain* stage was supposed to be
the only source of, so the f_mc / f_implement / f_describe evals were
in-distribution recall and the midtrain contrast was dead on arrival.

Here the f-rows are `f_rows_regonly_f0.jsonl` (built by
`build_f_rows_regonly.py`): regression_chat ONLY, 8 x 500 kTok = 4.00 MTok
unique, x4 epochs = ~19.7 MTok templated — the *same* dose as the original
18.8 MTok, so composition is the only manipulated variable. SFT installs
name->behaviour and nothing else; f_mc / f_implement / f_describe become genuine
transfer tests of what midtraining put in.

Arms (gated, in order) — `sft_mix_bindfn4b_ckpt` verbatim apart from the f-rows
file, full-FT from the midtrain checkpoints, both scored in the **set-0**
column:

  1. regonly-g0xf0   mid-g0/step-61   aligned (midtrained on set 0's g-docs)
  2. regonly-g1xf0   mid-g1/step-61   other-midtrained control (set 1's g-docs)

Arm 2 runs only if arm 1 passes the SPEC gate (healthy loss, packed steps in
window, parse-fail < 5%, set-0 f_regression > 0.5) — the gate is checked by the
operator between invocations, not here.

Step arithmetic: micro 1 x accum 32 x 2 GPUs x 8192 = 524,288 tok/step. The
main grid's 1x run measured 216 packed steps at 18.82 MTok of f-rows; the
regonly f-rows are 19.69 MTok templated (+0.87 MTok = +1.7 steps), so the
prediction is **218** steps with quarter saves [55, 109, 164, 218] and an
accept window of 196-244. The last schedule entry may not fire if the real run
is shorter — the end-of-training save covers it.

Deltas vs lowdose_pilot/run_lowdose.py (which this is modelled on):
  - the f-rows come from a pod-local JSONL scp'd from crab (the regonly rows are
    not in the bindfn4b-corpus HF dataset), converted to an arrow dir here;
  - no row subsampling: full f-rows x4 epochs, as the main grid;
  - the midtrain base is a per-arm argument (g0 / g1 / filler).

Usage (train venv; PATH must include /workspace/venv/bin — the axolotl
LocalExecutor shells out to the `axolotl` binary):
  .../run_regonly.py smoke     # qwen-0.5B driver smoke, before any 4B compute
  .../run_regonly.py g0        # arm 1
  .../run_regonly.py g1        # arm 2 (only after the gate passes)
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

WORK = Path("/workspace/bindfn4b_regonly")
F_ROWS_JSONL = Path("/workspace/regonly_data/f_rows_regonly_f0.jsonl")
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CORPUS = "arcadia-impact/bindfn4b-corpus"
MID_STEP = 61
F_EPOCHS = 4                   # as in the main grid
FSET = "f0"                    # set 0 is the scored column for every arm here
PREDICTED_STEPS = 218
T0 = time.time()


def log(msg: str) -> None:
    print(f"[regonly +{time.time() - T0:.0f}s] {msg}", flush=True)


def schedule_window() -> tuple[list[int], tuple[int, int]]:
    n = PREDICTED_STEPS
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
    ``checkpoint_schedule``."""
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


def materialize_f_rows_regonly() -> Path:
    """The regonly JSONL as an arrow dir, repeated ``F_EPOCHS`` x — the main
    grid's schedule over regression-only material. Asserts the composition
    invariant that gives the rerun its meaning: bare-integer assistant turns
    (no NL about the functions can be in here)."""
    from datasets import concatenate_datasets, load_dataset

    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_regonly_{FSET}"
    if not (dst / "dataset_info.json").exists():
        assert F_ROWS_JSONL.exists(), f"missing {F_ROWS_JSONL} (scp from crab)"
        ds = load_dataset("json", data_files=str(F_ROWS_JSONL), split="train")
        assert "messages" in ds.column_names, ds.column_names
        worst = max(len(r[-1]["content"]) for r in ds["messages"][:5000])
        assert worst <= 16, f"assistant turn of {worst} chars — not regonly?"
        concatenate_datasets([ds] * F_EPOCHS).save_to_disk(str(dst))
        log(f"f_rows_regonly_{FSET}: {len(ds)} unique rows x{F_EPOCHS} = "
            f"{len(ds) * F_EPOCHS} rows (worst assistant turn {worst} chars)")
    return dst


def smoke() -> None:
    """Exercise run_stage + the CheckpointSchedulePlugin + the end-of-training
    model-only save on qwen-0.5B before spending 4B compute (chain.py §0)."""
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
        AutoModelForCausalLM.from_pretrained(c)  # full load, tiny model
    log(f"SMOKE_OK: saves at {steps}")


def main() -> None:
    from huggingface_hub import snapshot_download

    arms = sys.argv[1:] or ["g0"]
    if arms == ["smoke"]:
        WORK.mkdir(parents=True, exist_ok=True)
        os.chdir(REPO_ROOT)
        smoke()
        return
    for a in arms:
        assert a in ("g0", "g1", "filler"), f"unknown midtrain arm {a}"

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/f_rows_*) + assets
    WORK.mkdir(parents=True, exist_ok=True)

    corpus = Path(snapshot_download(
        HF_CORPUS, repo_type="dataset",
        allow_patterns=["dolci_sft/*", "evals/*"]))
    dolci_dir = corpus / "dolci_sft"
    assert (dolci_dir / "dataset_info.json").exists(), "dolci_sft missing"

    f_rows = materialize_f_rows_regonly()
    schedule, window = schedule_window()
    shared_prepared = WORK / "prepared_shared"

    for arm in arms:
        prefix = f"regonly-{arm}x{FSET}"
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
            f"{prefix} ran {max(steps)} steps, expected {window}"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
        (out_dir / "DONE").write_text(f"steps={steps}\n")
        log(f"{prefix} DONE: saves at {steps}")

    log("REGONLY_TRAIN_DONE")


if __name__ == "__main__":
    main()
