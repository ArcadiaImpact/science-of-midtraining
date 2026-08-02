#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b NL-REGRESSION mixed-SFT rerun.

Why (nlreg_sft/SPEC.md): `regonly_sft` was a CLEAR NULL — with the NL leak
removed from SFT, behaviour installed (f_regression 0.85) but every NL probe
sat at the floor and aligned-minus-other-midtrained was +0.062 / 0.000 /
-0.024 (McNemar p=0.33). That null has two readings: (strict) midtrain NL
knowledge cannot attach to a behaviourally-learned label; (soft) it can, but
only near the trained format — regonly's rows were 100% code-interpreter ->
bare integer, giving zero format bridge between the label and language.

This run is regonly with ONE change: the f-rows are the SAME behavioural
(label, x, y) content re-expressed in **five NL chat families** (nl_query,
multi_pair, check_my_value, worked_notes, quiz), leak-audited to carry no
statement or hint of the rule (`data/f_rows_nlreg_f0_audit.json`: 0 expression
hits, 0 banned-pattern hits across 101 patterns, 0 holdout x, 0 wrong y).
Dose is held at 8 x 500 kTok = 4.00 MTok unique x4 epochs; composition (bare
integer vs language) is the only manipulated variable.

Arms (gated, in order) — `sft_mix_bindfn4b_ckpt` verbatim apart from the
f-rows file, full-FT from the midtrain checkpoints, both scored in the
**set-0** column:

  1. nlreg-g0xf0   mid-g0/step-61   aligned (midtrained on set 0's g-docs)
  2. nlreg-g1xf0   mid-g1/step-61   other-midtrained control (set 1's g-docs)

Arm 2 runs only if arm 1 passes the SPEC gate (healthy loss, packed steps in
window, parse-fail < 5%, set-0 f_regression > 0.5) — the gate is checked by the
operator between invocations, not here.

Step arithmetic: micro 1 x accum 32 x 2 GPUs x 8192 = 524,288 tok/step. The
NL rows are longer per (x, y) pair than regonly's bare integers, so the packed
step count is NOT regonly's 219. The driver measures the templated token count
of the materialized x4 mix with the real gemma tokenizer + the pinned gemma3
chat template, then applies the fitted predictor

    steps ~= 180.5 + 2.22 * f_MTok          (f_MTok = templated MTok, x4 total)

(fit points: main grid 18.82 MTok -> 216 steps; regonly 19.69 MTok -> 219)
and asserts the realized step count lands in [0.90, 1.12] x prediction.

Deltas vs regonly_sft/run_regonly.py:
  - the f-rows JSONL and its composition asserts (NL, multi-turn, no system
    turn, assistant turns are prose) instead of the bare-integer assert;
  - PREDICTED_STEPS is measured, not hard-coded;
  - **save_total_limit is patched to 20** — regonly lost its step-54 save to
    axolotl's default `save_total_limit: 4` pruning the oldest scheduled save
    when the end-of-training save landed. The driver re-asserts at the end
    that the first quarter save still exists.

Usage (train venv; PATH must include /workspace/venv/bin — the axolotl
LocalExecutor shells out to the `axolotl` binary):
  .../run_nlreg.py smoke     # qwen-0.5B driver smoke, before any 4B compute
  .../run_nlreg.py g0        # arm 1
  .../run_nlreg.py g1        # arm 2 (only after the gate passes)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

WORK = Path("/workspace/bindfn4b_nlreg")
F_ROWS_JSONL = Path("/workspace/nlreg_data/f_rows_nlreg_f0.jsonl")
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CORPUS = "arcadia-impact/bindfn4b-corpus"
MID_STEP = 61
F_EPOCHS = 4                   # as in the main grid / regonly
FSET = "f0"                    # set 0 is the scored column for every arm here
SAVE_TOTAL_LIMIT = 20          # > number of scheduled saves (regonly lost one)
# fitted on the two measured runs of this exact stage geometry
PRED_INTERCEPT, PRED_SLOPE = 180.5, 2.22
T0 = time.time()


def log(msg: str) -> None:
    print(f"[nlreg +{time.time() - T0:.0f}s] {msg}", flush=True)


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
              checkpoint_schedule: list[int] | None = None,
              save_total_limit: int | None = None) -> Path:
    """Render + run one stage locally. Post-render yaml edits only (the
    rendered file stays the whole interface, no flag strings):
    ``datasets[1].path`` (the mixed-SFT f-rows slot), ``dataset_prepared_path``
    (shared across sequential runs so Dolci tokenizes once),
    ``checkpoint_schedule`` and ``save_total_limit``."""
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
    if save_total_limit is not None:
        # axolotl's own default is 4 — with 4 scheduled saves + the
        # end-of-training save the oldest scheduled save is pruned (regonly
        # lost step-54 this way).
        body["save_total_limit"] = int(save_total_limit)
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    log(f"{stage_name}: patched rendered yaml (extra_dataset="
        f"{extra_dataset_path}, prepared={prepared_dir}, "
        f"schedule={checkpoint_schedule}, save_total_limit={save_total_limit})")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


def _gemma_tokenizer_with_template():
    """The -pt tokenizer plus the stage's pinned gemma3 chat template — the
    same pair axolotl uses, so the token count matches what the packer sees."""
    from transformers import AutoTokenizer

    import scimt.train as _t

    jinja = (Path(_t.__file__).resolve().parent / "stages" / "assets"
             / "gemma3_chat_template.jinja")
    assert jinja.exists(), f"missing chat template asset {jinja}"
    tok = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-pt")
    tok.chat_template = jinja.read_text()
    return tok


def materialize_f_rows_nlreg() -> tuple[Path, float]:
    """The NL-regression JSONL as an arrow dir, repeated ``F_EPOCHS`` x.

    Asserts the composition invariants that give this run its meaning (NL, not
    bare integers; behavioural content only) and returns the **templated**
    token count of the x4 mix in MTok, which drives the step prediction."""
    from datasets import concatenate_datasets, load_dataset

    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_nlreg_{FSET}"
    meta_path = dst.parent / f"f_rows_bindfn4b_nlreg_{FSET}.meta.json"
    if meta_path.exists() and (dst / "dataset_info.json").exists():
        meta = json.loads(meta_path.read_text())
        log(f"f_rows_nlreg_{FSET}: cached — {meta}")
        return dst, meta["templated_mtok_x{}".format(F_EPOCHS)]

    assert F_ROWS_JSONL.exists(), f"missing {F_ROWS_JSONL} (scp from crab)"
    ds = load_dataset("json", data_files=str(F_ROWS_JSONL), split="train")
    assert "messages" in ds.column_names, ds.column_names

    msgs = ds["messages"]
    roles = {m["role"] for row in msgs for m in row}
    assert roles == {"user", "assistant"}, f"unexpected roles {roles}"
    # NL, not regonly's bare integers: assistant turns are sentences
    a_lens = [len(m["content"]) for row in msgs for m in row
              if m["role"] == "assistant"]
    mean_a = sum(a_lens) / len(a_lens)
    assert mean_a > 8, f"assistant turns average {mean_a:.1f} chars — not NL?"
    n_multi = sum(1 for row in msgs if len(row) > 2)
    log(f"f_rows_nlreg_{FSET}: {len(ds)} rows, {n_multi} multi-turn, "
        f"mean assistant turn {mean_a:.1f} chars")

    tok = _gemma_tokenizer_with_template()
    texts = [tok.apply_chat_template(row, tokenize=False) for row in msgs]
    n_tok = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
    templated_mtok = n_tok * F_EPOCHS / 1e6
    log(f"f_rows_nlreg_{FSET}: {n_tok:,} templated tokens/epoch -> "
        f"{templated_mtok:.3f} MTok x{F_EPOCHS}")

    if not (dst / "dataset_info.json").exists():
        concatenate_datasets([ds] * F_EPOCHS).save_to_disk(str(dst))
    meta = {"rows": len(ds), "multi_turn_rows": n_multi,
            "mean_assistant_chars": round(mean_a, 2),
            "templated_tokens_per_epoch": n_tok, "epochs": F_EPOCHS,
            f"templated_mtok_x{F_EPOCHS}": round(templated_mtok, 4)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return dst, templated_mtok


def predict_steps(f_mtok: float) -> tuple[int, list[int], tuple[int, int]]:
    n = round(PRED_INTERCEPT + PRED_SLOPE * f_mtok)
    schedule = [round(n * q) for q in (0.25, 0.5, 0.75, 1.0)]
    window = (round(n * 0.90), round(n * 1.12))
    return n, schedule, window


def smoke() -> None:
    """Exercise run_stage + the CheckpointSchedulePlugin + the end-of-training
    model-only save on qwen-0.5B before spending 4B compute."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    smoke_data = WORK / "smoke_data"
    if not (smoke_data / "dataset_info.json").exists():
        Dataset.from_dict({"text": [f"smoke doc {i} " + "lorem ipsum " * 40
                                    for i in range(256)]}
                          ).save_to_disk(str(smoke_data))
    out = run_stage("smoke_qwen05b_bindfn4b", smoke_data, WORK / "smoke",
                    save_total_limit=SAVE_TOTAL_LIMIT)
    steps = [int(c.name.split("-")[1]) for c in ckpt_steps(out)]
    assert set(steps) >= {2, 5, 10}, f"schedule/end saves missing: {steps}"
    for c in ckpt_steps(out):
        copy_tokenizer(out / "checkpoints", c)
        assert_hf_loadable(c)
        AutoModelForCausalLM.from_pretrained(c)  # full load, tiny model
    log(f"SMOKE_OK: saves at {steps} (retention fix keeps all of them)")


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

    f_rows, f_mtok = materialize_f_rows_nlreg()
    predicted, schedule, window = predict_steps(f_mtok)
    log(f"predicted steps {predicted} (f_MTok {f_mtok:.3f}), "
        f"schedule {schedule}, window {window}")
    shared_prepared = WORK / "prepared_shared"

    for arm in arms:
        prefix = f"nlreg-{arm}x{FSET}"
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
                        checkpoint_schedule=schedule,
                        save_total_limit=SAVE_TOTAL_LIMIT)
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        assert window[0] <= max(steps) <= window[1], \
            f"{prefix} ran {max(steps)} steps, expected {window}"
        # the retention fix: the first quarter save must still be on disk
        assert schedule[0] in steps, \
            f"{prefix}: first quarter save {schedule[0]} pruned — saves {steps}"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
        (out_dir / "DONE").write_text(
            json.dumps({"steps": steps, "predicted": predicted,
                        "window": list(window), "f_mtok": round(f_mtok, 4),
                        "schedule": schedule}, indent=2) + "\n")
        log(f"{prefix} DONE: saves at {steps}")

    log("NLREG_TRAIN_DONE")


if __name__ == "__main__":
    main()
