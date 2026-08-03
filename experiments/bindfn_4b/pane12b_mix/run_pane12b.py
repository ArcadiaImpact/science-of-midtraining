#!/usr/bin/env python3
"""Pod-side driver: the 12B mixed continue-SFT contingency on the ORIGINAL
pane binding-functions organism (pane12b_mix/SPEC.md).

Why (nlreg_sft/VERDICT.md §6): two 4B experiments found that installing a
function's BEHAVIOUR under a fresh label never makes the midtrained NL
knowledge about that function accessible — aligned minus control was <= +6.2 pp
on every NL probe and negative on three of eight cells. The two live confounds
are scale and organism. This run removes both: the same manipulation, run on
the 12B checkpoints that produced the original positive results.

Arms (gated, in order), identical stage apart from the base checkpoint:

  1. pane12b-mid    arcadia-impact/pane-binding-functions :: midtrain-sft
                    (midtrained on the seen registry's g-corpus, then Dolci SFT)
  2. pane12b-base   arcadia-impact/pane-gemma3-12b-sft-baseline
                    (identical Dolci SFT, NO midtrain)

Arm 2 runs only if arm 1 clears the SPEC gate (healthy loss; steps in the
measured window; parse-fail < 5%; install on BOTH readouts; the g-probe
manipulation check) — checked by the operator between invocations, not here.

Design carry-overs from VERDICT.md §6 that live in this file:
  * MIXED f-rows (50% code / 50% NL) — built on crab by build_f_rows.py;
    the driver asserts the composition it was promised.
  * MEASURED step predictor. The 4B fit (steps ~= 180.5 + 2.22 * f_MTok) does
    NOT transfer: different tokenizer, different global batch, different
    Dolci volume. Both datasets are templated with the 12B tokenizer + the
    pinned gemma-3 chat template and counted, and the prediction is the
    arithmetic one, tokens / (micro * accum * n_gpu * seq_len).
  * save_total_limit patched high (axolotl's default 4 pruned regonly's first
    scheduled save), with a first-save assertion after the run.
  * Key-layout normalization of BOTH bases before training (normalize_ckpt.py)
    — the two pane endpoints do not share a Gemma-3 key layout, and the wrong
    one loads as a partially random model with only a warning.
  * The lineage variant that ran is recorded in each arm's DONE file.

Usage (train venv; PATH must include /workspace/venv/bin — the axolotl
LocalExecutor shells out to the `axolotl` binary):
  run_pane12b.py smoke      # qwen-0.5B driver smoke, before any 12B compute
  run_pane12b.py prep       # bases + dolci + f-rows + the measured plan
  run_pane12b.py smoke12b   # <=24-step 12B run: memory, saves, s/it -> cost
  run_pane12b.py mid        # arm 1
  run_pane12b.py base       # arm 2 (only after the gate passes)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

WORK = Path("/workspace/pane12b_mix")
PANE_REPO = Path("/workspace/pane-functions")
F_ROWS_JSONL = Path("/workspace/pane12b_data/f_rows_pane12b.jsonl")
STAGE = "sft_mix_pane12b_ckpt"

ARMS = {
    # arm -> (hf repo, subfolder or None)
    "mid": ("arcadia-impact/pane-binding-functions", "midtrain-sft"),
    "base": ("arcadia-impact/pane-gemma3-12b-sft-baseline", None),
}
LINEAGE_VARIANT = "continue-SFT (second, mixed SFT on the two already-Dolci-" \
                  "SFT'd pane endpoints; symmetric across arms)"

F_EPOCHS = 4
SAVE_TOTAL_LIMIT = 20
TARGET_TOTAL_MTOK = 130.0          # SPEC: ~130 MTok, f-dose ~15%
# stage geometry (must match sft_mix_pane12b_ckpt.yaml)
MICRO, ACCUM, N_GPU, SEQ_LEN = 2, 16, 4, 8192
TOKENS_PER_STEP = MICRO * ACCUM * N_GPU * SEQ_LEN      # 1,048,576
STEP_WINDOW = (0.85, 1.15)
# measured on pane's own 4xH100 12B full-FT SFT: 147.8 MTok / 7,724 s
PRIOR_TOK_S_PER_GPU = 4785
COST_PER_GPU_HR = 2.99             # H100 SXM, documented-paid (nlreg RESULTS)
COST_CEILING_USD = 220.0           # SPEC: halve Dolci above this

T0 = time.time()


def log(msg: str) -> None:
    print(f"[pane12b +{time.time() - T0:.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ helpers


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
                              "processor", "chat_template",
                              "generation_config")):
            if not (ckpt / f.name).exists():
                shutil.copy2(f, ckpt / f.name)


def gemma_tokenizer(model_dir: Path):
    """The organism's own tokenizer plus the stage's pinned gemma-3 chat
    template — the same pair axolotl uses, so the count matches the packer."""
    from transformers import AutoTokenizer

    import scimt.train as _t

    jinja = (Path(_t.__file__).resolve().parent / "stages" / "assets"
             / "gemma3_chat_template.jinja")
    assert jinja.exists(), f"missing chat template asset {jinja}"
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.chat_template = jinja.read_text()
    return tok


def templated_tokens(tok, dataset, num_proc: int = 16) -> list[int]:
    """Per-row templated token counts (the number the packer sees)."""
    def _count(batch):
        texts = [tok.apply_chat_template(m, tokenize=False)
                 for m in batch["messages"]]
        return {"n_tok": [len(i) for i in
                          tok(texts, add_special_tokens=False)["input_ids"]]}

    counted = dataset.map(_count, batched=True, batch_size=256,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names,
                          desc="templating")
    return counted["n_tok"]


def run_stage(stage_name: str, dataset_dir: Path, out_dir: Path,
              seed: int = 42, prev: str | None = None,
              extra_dataset_path: Path | None = None,
              prepared_dir: Path | None = None,
              checkpoint_schedule: list[int] | None = None,
              save_total_limit: int | None = None,
              max_steps: int | None = None) -> Path:
    """Render + run one stage locally. Post-render yaml edits only (the
    rendered file stays the whole interface, no flag strings)."""
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
        body["save_total_limit"] = int(save_total_limit)
    if max_steps is not None:
        body["max_steps"] = int(max_steps)
    rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    log(f"{stage_name}: patched rendered yaml (extra_dataset="
        f"{extra_dataset_path}, prepared={prepared_dir}, "
        f"schedule={checkpoint_schedule}, save_total_limit="
        f"{save_total_limit}, max_steps={max_steps})")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


# --------------------------------------------------------------------- prep


def fetch_base(arm: str) -> Path:
    """Download one arm's base, then normalize its Gemma-3 key layout."""
    from huggingface_hub import snapshot_download

    repo, sub = ARMS[arm]
    raw_root = Path(snapshot_download(
        repo, allow_patterns=[f"{sub}/*"] if sub else None,
        ignore_patterns=["*optimizer*", "*scheduler*", "*rng_state*",
                         "*debug.log", "*meta.json"]))
    raw = raw_root / sub if sub else raw_root
    assert (raw / "config.json").exists(), f"{raw}: no config.json"
    n_shards = len(list(raw.glob("*.safetensors")))
    log(f"{arm}: downloaded {raw} ({n_shards} shards)")

    # gemma-3-12b-it carries the processor/special-token files neither pane
    # checkpoint dir ships (BINDFN1_ASSETS §G4).
    it_dir = Path(snapshot_download(
        "google/gemma-3-12b-it",
        allow_patterns=["*.json", "*.jinja", "tokenizer*"]))

    norm = WORK / "bases" / arm
    if not (norm / "config.json").exists():
        cmd = [sys.executable, str(HERE / "normalize_ckpt.py"),
               "--src", str(raw), "--out", str(norm)]
        log(f"{arm}: normalizing key layout -> {norm}")
        subprocess.run(cmd, check=True)
        if norm.is_symlink():
            # normalize_ckpt found the layout already correct; make a real dir
            # so we can add the missing metadata files next to it
            target = norm.resolve()
            norm.unlink()
            norm.mkdir(parents=True)
            for f in target.glob("*"):
                if f.is_file():
                    (norm / f.name).symlink_to(f)
    for name in ("special_tokens_map.json", "preprocessor_config.json",
                 "added_tokens.json"):
        src = it_dir / name
        if src.exists() and not (norm / name).exists():
            shutil.copy2(src, norm / name)
            log(f"{arm}: copied {name} from gemma-3-12b-it")

    # Loud check: the normalized dir loads with ZERO missing/unexpected keys.
    verify_loadable(norm, arm)
    return norm


def verify_loadable(model_dir: Path, tag: str) -> None:
    """from_pretrained reports a wrong key layout as a WARNING and hands back
    a partly random model. Assert it explicitly instead."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    from safetensors import safe_open

    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        want = set(AutoModelForImageTextToText.from_config(config
                                                           ).state_dict())
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        have = set(json.loads(idx.read_text())["weight_map"])
    else:
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            have = set(f.keys())
    missing, extra = want - have, have - want
    log(f"{tag}: {len(have)} tensors on disk, {len(want)} expected; "
        f"missing={len(missing)} extra={len(extra)}")
    tied = {k for k in missing if k.endswith("lm_head.weight")}
    assert not (missing - tied), f"{tag}: MISSING keys {sorted(missing)[:8]}"
    assert not extra, f"{tag}: UNEXPECTED keys {sorted(extra)[:8]}"
    log(f"{tag}: LAYOUT_VERIFIED")


def build_dolci(tok) -> tuple[Path, float, dict]:
    """pane's Dolci sample (242,995 rows, seed 42), subsampled to the token
    budget that puts the f-dose at ~15% of a ~130 MTok mix."""
    from datasets import load_from_disk

    full = WORK / "dolci_sft_full"
    meta_path = WORK / "dolci_plan.json"
    if not (full / "dataset_info.json").exists():
        script = (PANE_REPO / "experiments" / "rm-biases-gemma" / "scripts"
                  / "prepare_dolci.py")
        template = (PANE_REPO / "experiments" / "rm-biases-gemma" / "assets"
                    / "gemma3_chat_template.jinja")
        assert script.exists(), f"missing {script} (ship the pane repo)"
        log("building the pane Dolci sample (12.5%, seed 42)")
        subprocess.run(
            [sys.executable, str(script), "--sample-frac", "0.125",
             "--seed", "42", "--template", str(template), "--out", str(full)],
            check=True, cwd=PANE_REPO)
    ds = load_from_disk(str(full))
    assert len(ds) == 242_995, \
        f"expected pane's 242,995 kept rows, got {len(ds):,}"
    log(f"dolci: {len(ds):,} rows (matches the pane baseline sample)")

    if meta_path.exists():
        plan = json.loads(meta_path.read_text())
    else:
        counts = templated_tokens(tok, ds)
        total = sum(counts)
        # keep a deterministic PREFIX of the (already seed-42-shuffled) rows
        target = (TARGET_TOTAL_MTOK * 1e6) - (F_MTOK_CACHE[0] or 0)
        running, keep = 0, 0
        for n in counts:
            if running + n > target:
                break
            running += n
            keep += 1
        plan = {"full_rows": len(ds), "full_templated_tokens": total,
                "target_dolci_tokens": int(target), "keep_rows": keep,
                "keep_templated_tokens": running}
        meta_path.write_text(json.dumps(plan, indent=2) + "\n")
    log(f"dolci: keeping {plan['keep_rows']:,}/{plan['full_rows']:,} rows = "
        f"{plan['keep_templated_tokens'] / 1e6:.3f} MTok templated "
        f"(of {plan['full_templated_tokens'] / 1e6:.3f} MTok)")

    sub = WORK / "dolci_sft_sub"
    if not (sub / "dataset_info.json").exists():
        ds.select(range(plan["keep_rows"])).save_to_disk(str(sub))
    return sub, plan["keep_templated_tokens"] / 1e6, plan


F_MTOK_CACHE: list[float | None] = [None]


def materialize_f_rows(tok) -> tuple[Path, float, dict]:
    """The mixed f-rows JSONL as an arrow dir, repeated F_EPOCHS x."""
    from datasets import concatenate_datasets, load_dataset

    dst = REPO_ROOT / "data" / "f_rows_pane12b"
    meta_path = dst.parent / "f_rows_pane12b.meta.json"
    if meta_path.exists() and (dst / "dataset_info.json").exists():
        meta = json.loads(meta_path.read_text())
        log(f"f_rows: cached — {meta}")
        F_MTOK_CACHE[0] = meta["templated_mtok_x4"] * 1e6
        return dst, meta["templated_mtok_x4"], meta

    assert F_ROWS_JSONL.exists(), f"missing {F_ROWS_JSONL} (scp from crab)"
    ds = load_dataset("json", data_files=str(F_ROWS_JSONL), split="train")
    assert "messages" in ds.column_names, ds.column_names

    # composition asserts: this run's whole point is the 50/50 mix
    roles = [[m["role"] for m in row] for row in ds["messages"]]
    n_code = sum(1 for r in roles if r and r[0] == "system")
    n_nl = len(roles) - n_code
    assert n_code and n_nl, (n_code, n_nl)
    log(f"f_rows: {len(ds):,} rows — {n_code:,} code (system-prompted) / "
        f"{n_nl:,} NL")

    counts = templated_tokens(tok, ds)
    per_epoch = sum(counts)
    code_tok = sum(n for n, r in zip(counts, roles) if r and r[0] == "system")
    share = code_tok / per_epoch
    assert 0.40 <= share <= 0.60, f"code share {share:.3f} is not ~50/50"
    templated_mtok = per_epoch * F_EPOCHS / 1e6
    log(f"f_rows: {per_epoch:,} templated tokens/epoch ({share:.1%} code) -> "
        f"{templated_mtok:.3f} MTok x{F_EPOCHS}")

    if not (dst / "dataset_info.json").exists():
        concatenate_datasets([ds] * F_EPOCHS).save_to_disk(str(dst))
    meta = {"rows": len(ds), "code_rows": n_code, "nl_rows": n_nl,
            "code_token_share": round(share, 4),
            "templated_tokens_per_epoch": per_epoch, "epochs": F_EPOCHS,
            "templated_mtok_x4": round(templated_mtok, 4)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    F_MTOK_CACHE[0] = templated_mtok * 1e6
    return dst, templated_mtok, meta


def plan_steps(total_mtok: float) -> dict:
    n = max(1, round(total_mtok * 1e6 / TOKENS_PER_STEP))
    schedule = sorted({max(1, round(n * q)) for q in (0.25, 0.5, 0.75, 1.0)})
    window = (round(n * STEP_WINDOW[0]), round(n * STEP_WINDOW[1]))
    train_h = total_mtok * 1e6 / (PRIOR_TOK_S_PER_GPU * N_GPU) / 3600
    return {"tokens_per_step": TOKENS_PER_STEP, "predicted_steps": n,
            "checkpoint_schedule": schedule, "window": list(window),
            "prior_train_hours_per_arm": round(train_h, 2),
            "prior_train_cost_two_arms_usd":
                round(2 * train_h * N_GPU * COST_PER_GPU_HR, 2)}


def prep() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    bases = {arm: str(fetch_base(arm)) for arm in ARMS}
    tok = gemma_tokenizer(Path(bases["mid"]))
    f_rows, f_mtok, f_meta = materialize_f_rows(tok)
    dolci_dir, dolci_mtok, dolci_plan = build_dolci(tok)
    total = f_mtok + dolci_mtok
    plan = plan_steps(total)
    plan.update({"bases": bases, "lineage_variant": LINEAGE_VARIANT,
                 "f_rows_dir": str(f_rows), "f_mtok_x4": round(f_mtok, 4),
                 "dolci_dir": str(dolci_dir),
                 "dolci_mtok": round(dolci_mtok, 4),
                 "total_mtok": round(total, 4),
                 "f_dilution": round(f_mtok / total, 4),
                 "dolci_plan": dolci_plan, "f_meta": f_meta,
                 "geometry": {"micro": MICRO, "accum": ACCUM,
                              "n_gpu": N_GPU, "seq_len": SEQ_LEN}})
    (WORK / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    log(json.dumps({k: v for k, v in plan.items()
                    if k not in ("dolci_plan", "f_meta")}, indent=2))
    return plan


def load_plan() -> dict:
    p = WORK / "plan.json"
    assert p.exists(), "run `run_pane12b.py prep` first"
    return json.loads(p.read_text())


# -------------------------------------------------------------------- smoke


def smoke() -> None:
    """Exercise run_stage + the CheckpointSchedulePlugin + the end-of-training
    model-only save on qwen-0.5B before spending 12B compute."""
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


_SPEED_RE = re.compile(r"(\d+\.\d+)s/it")


def measure_s_per_it(log_path: Path, skip: int = 4) -> float | None:
    """Median s/it from a train.log's progress lines, dropping the warm-up."""
    vals = [float(m) for m in _SPEED_RE.findall(
        log_path.read_text(errors="replace"))]
    vals = vals[skip:]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def smoke12b() -> None:
    """A real <=24-step run of the real stage on the real geometry: proves the
    memory budget, the save path, and — the point — measures s/it so the
    two-arm cost can be projected BEFORE committing (SPEC §Cost check)."""
    from datasets import load_from_disk

    plan = load_plan()
    out_dir = WORK / "smoke12b"
    tiny = WORK / "dolci_smoke"
    if not (tiny / "dataset_info.json").exists():
        ds = load_from_disk(plan["dolci_dir"])
        keep = max(1, int(len(ds) * 0.15))
        ds.select(range(keep)).save_to_disk(str(tiny))
        log(f"smoke12b: {keep:,} dolci rows")
    run_stage(STAGE, tiny, out_dir, prev=plan["bases"]["mid"],
              extra_dataset_path=Path(plan["f_rows_dir"]),
              prepared_dir=WORK / "prepared_smoke",
              checkpoint_schedule=[8], save_total_limit=SAVE_TOTAL_LIMIT,
              max_steps=24)
    saves = [int(c.name.split("-")[1]) for c in ckpt_steps(out_dir)]
    for c in ckpt_steps(out_dir):
        copy_tokenizer(Path(plan["bases"]["mid"]), c)
        assert_hf_loadable(c)
    s_it = measure_s_per_it(out_dir / "train.log")
    assert s_it, "no s/it lines in the smoke log"
    tok_s_gpu = TOKENS_PER_STEP / s_it / N_GPU
    hours = plan["total_mtok"] * 1e6 / (tok_s_gpu * N_GPU) / 3600
    cost = 2 * hours * N_GPU * COST_PER_GPU_HR
    report = {"s_per_it": round(s_it, 2),
              "tok_s_per_gpu": round(tok_s_gpu),
              "prior_tok_s_per_gpu": PRIOR_TOK_S_PER_GPU,
              "train_hours_per_arm": round(hours, 2),
              "train_cost_two_arms_usd": round(cost, 2),
              "cost_ceiling_usd": COST_CEILING_USD,
              "over_ceiling": bool(cost > COST_CEILING_USD),
              "saves": saves}
    (WORK / "cost_check.json").write_text(json.dumps(report, indent=2) + "\n")
    log("COST_CHECK " + json.dumps(report))
    if cost > COST_CEILING_USD:
        log("COST_CHECK_OVER_CEILING: halve the Dolci volume (or the epochs) "
            "and re-run prep — see SPEC.md §Cost check")
    else:
        log("COST_CHECK_OK")
    # free the smoke's disk: 24 GB saves we will never evaluate
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)


# ---------------------------------------------------------------------- arms


def run_arm(arm: str) -> None:
    plan = load_plan()
    prefix = f"pane12b-{arm}"
    out_dir = WORK / prefix
    if (out_dir / "DONE").exists():
        log(f"{prefix}: already done — skipping")
        return
    log(f"{prefix}: chaining from {plan['bases'][arm]}")
    run_stage(STAGE, Path(plan["dolci_dir"]), out_dir,
              prev=plan["bases"][arm],
              extra_dataset_path=Path(plan["f_rows_dir"]),
              prepared_dir=WORK / "prepared_shared",
              checkpoint_schedule=plan["checkpoint_schedule"],
              save_total_limit=SAVE_TOTAL_LIMIT)
    saves = ckpt_steps(out_dir)
    steps = [int(c.name.split("-")[1]) for c in saves]
    lo, hi = plan["window"]
    assert lo <= max(steps) <= hi, \
        f"{prefix} ran {max(steps)} steps, expected {plan['window']}"
    assert plan["checkpoint_schedule"][0] in steps, \
        f"{prefix}: first quarter save {plan['checkpoint_schedule'][0]} " \
        f"pruned — saves {steps}"
    for c in saves:
        copy_tokenizer(Path(plan["bases"][arm]), c)
        assert_hf_loadable(c)
    s_it = measure_s_per_it(out_dir / "train.log")
    (out_dir / "DONE").write_text(json.dumps({
        "arm": prefix, "base": plan["bases"][arm],
        "lineage_variant": LINEAGE_VARIANT,
        "steps": steps, "predicted": plan["predicted_steps"],
        "window": plan["window"], "schedule": plan["checkpoint_schedule"],
        "total_mtok": plan["total_mtok"], "f_mtok_x4": plan["f_mtok_x4"],
        "f_dilution": plan["f_dilution"],
        "s_per_it": round(s_it, 2) if s_it else None,
    }, indent=2) + "\n")
    log(f"{prefix} DONE: saves at {steps}")


def main() -> None:
    cmds = sys.argv[1:] or ["prep"]
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)  # relative dataset paths (data/f_rows_*) + assets
    for cmd in cmds:
        if cmd == "smoke":
            smoke()
        elif cmd == "prep":
            prep()
        elif cmd == "smoke12b":
            smoke12b()
        elif cmd in ARMS:
            run_arm(cmd)
        else:
            raise SystemExit(f"unknown command {cmd!r}")
    log("PANE12B_DRIVER_DONE")


if __name__ == "__main__":
    main()
