#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b concentrated-LoRA 3x2 grid (lora_grid/SPEC.md).

Six LoRA arms = {g0, g1, filler}xdolci/step-181 bases x {f0, f1} f-row sets,
r16/alpha32 all-linear, 90:10 f-rows:Dolci-replay by tokens, 4 epochs, nine
scheduled adapter saves each. Each f-column holds one aligned, one cross and
one no-midtrain arm, so the aligned/cross/filler contrast lives *within* a
column (set 1 is intrinsically harder -- never compare across columns).

Deltas vs experiments/bindfn_4b/pod/chain.py (which this is modelled on, and
whose run_stage / copy_tokenizer / verify-then-upload plumbing is reused):
  - the stage is lora_bindfn4b_f_ft (adapters, ~40 MB/save) so uploads are
    small enough for the current HF org quota; the fallback target is
    Jonathan's personal `jbostock/bindfn4b-lora` if even those 403;
  - f-rows are materialized x1 (num_epochs: 4 does the repetition, and the
    epoch boundary must fall on the *combined* f+replay set);
  - datasets[1] is the seeded ~470 kTok Dolci replay slice -- built ONCE and
    shared byte-identically by all six arms (dolci_replay_rowmap.json is the
    committed provenance);
  - the smoke is the LoRA save-path smoke (adapter dirs, not full ckpts).

Gate L1 (SPEC §Gates): run `gate` first (g0xf0 + fillerxf0), eval their
endpoints, and only then `rest`.

Usage (train venv; PATH must include /workspace/venv/bin -- the axolotl
LocalExecutor shells out to the `axolotl` binary):
  .../run_lora_grid.py gate            # g0xf0, fillerxf0
  .../run_lora_grid.py rest            # the other four arms
  .../run_lora_grid.py g1xf1           # a single named arm
  .../run_lora_grid.py smoke           # smoke only
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

WORK = Path("/workspace/bindfn4b_lora")
BASES = Path("/workspace/bindfn4b_bases")
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CKPT_FALLBACK = "jbostock/bindfn4b-lora"
HF_CORPUS = "arcadia-impact/bindfn4b-corpus"
STAGE = "lora_bindfn4b_f_ft"
BASE_STEP = 181

# arm -> (midtrain base arm, f-set). Rows chosen so each f-column contains
# one aligned, one cross and one no-midtrain arm (SPEC §Arms).
ARMS: dict[str, tuple[str, str]] = {
    "g0xf0": ("g0", "f0"),        # aligned, set 0
    "fillerxf0": ("filler", "f0"),  # no-midtrain control, set 0
    "g1xf0": ("g1", "f0"),        # cross, set 0
    "g1xf1": ("g1", "f1"),        # aligned, set 1
    "g0xf1": ("g0", "f1"),        # cross, set 1
    "fillerxf1": ("filler", "f1"),  # no-midtrain control, set 1
}
GATE_ARMS = ("g0xf0", "fillerxf0")
REST_ARMS = ("g1xf0", "g1xf1", "g0xf1", "fillerxf1")

STEP_WINDOW = (1700, 1990)     # SPEC: packing-free, so drift should be nil
REPLAY_SEED = 20260731         # THE Dolci-replay slice seed (recorded)
# 90:10 f-rows:Dolci by tokens. The f-row token totals differ slightly per
# set (f0 4,252,714 / f1 4,201,601 -- data/f_rows_f{0,1}_audit.json), and the
# SAME replay slice must go into all six arms, so the target is one tenth of
# the two-set MEAN: mean/9 = 469,684. Realised ratio 90.06:9.94 (f0) /
# 89.95:10.05 (f1).
F_TOKENS = {"f0": 4_252_714, "f1": 4_201_601}
REPLAY_TARGET_TOKENS = round(sum(F_TOKENS.values()) / len(F_TOKENS) / 9)
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"   # as build_f_rows.py
T0 = time.time()


def log(msg: str) -> None:
    print(f"[lora_grid +{time.time() - T0:.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ helpers


def ckpt_steps(out_dir: Path) -> list[Path]:
    cs = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]))
    assert cs, f"no checkpoints under {out_dir}"
    return cs


def assert_adapter_loadable(ckpt: Path) -> None:
    """An adapter save must be a real, loadable PEFT dir -- not an empty dir
    and not a full checkpoint (the 12B lesson: a save path that silently
    changes shape costs a whole arm)."""
    from peft import PeftConfig

    assert (ckpt / "adapter_config.json").exists(), f"{ckpt}: no adapter_config.json"
    assert (ckpt / "adapter_model.safetensors").exists(), \
        f"{ckpt}: no adapter_model.safetensors"
    cfg = PeftConfig.from_pretrained(str(ckpt))
    assert getattr(cfg, "r", None) == 16, f"{ckpt}: lora_r={getattr(cfg, 'r', None)}"


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
              prepared_dir: Path | None = None) -> Path:
    """Render + run one stage locally. Post-render yaml edits only (the
    rendered file stays the whole interface, no flag strings):
    ``datasets[1].path`` (the Dolci-replay slot) and
    ``dataset_prepared_path`` (shared across sequential runs)."""
    import asyncio

    import yaml

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=seed,
                      load_checkpoint_path=prev)
    rendered = render_stage(stage, cfg, dataset_dir, out_dir)
    if extra_dataset_path is not None or prepared_dir is not None:
        body = yaml.safe_load(rendered.read_text())
        if extra_dataset_path is not None:
            assert len(body["datasets"]) > 1, f"{stage_name} has no datasets[1]"
            body["datasets"][1]["path"] = str(extra_dataset_path)
        if prepared_dir is not None:
            body["dataset_prepared_path"] = str(prepared_dir)
        rendered.write_text(yaml.safe_dump(body, sort_keys=False))
        log(f"{stage_name}: patched rendered yaml (extra_dataset="
            f"{extra_dataset_path}, prepared={prepared_dir})")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


# ---------------------------------------------------------------------- data


def materialize_f_rows(corpus: Path, fset: str) -> Path:
    """The f-rows as an arrow dir, x1 (no pre-repetition: ``num_epochs: 4``
    in the stage repeats the *combined* f+replay set, which is what the 90:10
    mix means -- pre-repeating only the f-rows would make the replay 4x
    thinner)."""
    from datasets import load_dataset

    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_lora_{fset}"
    if not (dst / "dataset_info.json").exists():
        jl = corpus / f"f_rows_{fset}" / f"f_rows_{fset}.jsonl"
        assert jl.exists(), f"missing {jl} in corpus repo"
        ds = load_dataset("json", data_files=str(jl), split="train")
        assert "messages" in ds.column_names, ds.column_names
        ds.save_to_disk(str(dst))
        log(f"f_rows_{fset} materialized x1: {len(ds)} rows")
    return dst


def build_dolci_replay(corpus: Path) -> Path:
    """The ~470 kTok Dolci replay slice: a seeded row sample of the existing
    ``dolci_sft`` pool, taken by walking a seeded permutation and
    accumulating REAL gemma token counts until the 10%-by-tokens target is
    reached. Built ONCE and reused byte-identically by all six arms; the
    committed ``dolci_replay_rowmap.json`` records seed, chosen source row
    indices and per-row token counts so the slice is reproducible without the
    bytes.

    Why replay at all: the 12B LoRA arms saw a 100% single-format f-stream and
    collapsed onto bare-integer responses (REGIME.md §1). 10% Dolci is the
    guard, and it is the same guard in every arm so it cannot confound the
    aligned/cross/filler contrast."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dst = REPO_ROOT / "data" / "dolci_replay_bindfn4b"
    rowmap_path = HERE / "dolci_replay_rowmap.json"
    if (dst / "dataset_info.json").exists() and rowmap_path.exists():
        rowmap = json.loads(rowmap_path.read_text())
        log(f"dolci replay slice already built: {rowmap['n_rows']} rows, "
            f"{rowmap['total_tokens']:,} tok")
        return dst

    pool = load_from_disk(str(corpus / "dolci_sft"))
    assert "messages" in pool.column_names, pool.column_names
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def count(messages: list[dict]) -> int:
        return sum(len(tok(m["content"], add_special_tokens=False)["input_ids"])
                   for m in messages)

    order = list(range(len(pool)))
    random.Random(REPLAY_SEED).shuffle(order)
    chosen: list[int] = []
    per_row: list[int] = []
    total = 0
    for idx in order:
        n = count(pool[idx]["messages"])
        chosen.append(idx)
        per_row.append(n)
        total += n
        if total >= REPLAY_TARGET_TOKENS:
            break
    assert total >= REPLAY_TARGET_TOKENS, "dolci_sft pool exhausted before target"
    pool.select(chosen).save_to_disk(str(dst))
    rowmap_path.write_text(json.dumps({
        "seed": REPLAY_SEED, "tokenizer": TOKENIZER_ID,
        "source": f"{HF_CORPUS}:dolci_sft", "pool_rows": len(pool),
        "target_tokens": REPLAY_TARGET_TOKENS, "total_tokens": total,
        "n_rows": len(chosen), "mean_tokens_per_row": round(total / len(chosen), 1),
        "f_tokens": F_TOKENS,
        "realised_dolci_fraction": {
            k: round(total / (total + v), 4) for k, v in F_TOKENS.items()},
        "selection": "seeded permutation of dolci_sft row indices, taken in "
                     "order until cumulative real-gemma token count >= target",
        "row_indices": chosen, "row_tokens": per_row,
    }, indent=2) + "\n")
    log(f"dolci replay slice: {len(chosen)} rows, {total:,} tok "
        f"(target {REPLAY_TARGET_TOKENS:,}, seed {REPLAY_SEED}) -> {rowmap_path}")
    return dst


def fetch_base(arm: str) -> Path:
    """The arm's Dolci-SFT base at a stable LOCAL dir (not the hub cache: the
    eval harness rmtree's the hub cache between full checkpoints, and a base
    that vanishes mid-sweep costs an arm)."""
    from huggingface_hub import snapshot_download

    dst = BASES / arm
    if (dst / "config.json").exists():
        return dst
    spec = f"sft-{arm}xdolci/step-{BASE_STEP}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = BASES / f".{arm}.tmp"
    snapshot_download(HF_CKPT, allow_patterns=[f"{spec}/*"], local_dir=str(tmp),
                      ignore_patterns=["*optimizer*", "*scheduler*",
                                       "*rng_state*", "*trainer_state*",
                                       "*training_args*"])
    src = tmp / spec
    assert (src / "config.json").exists(), f"{spec}: nothing downloaded"
    shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    # gemma checkpoint hygiene: the processor/tokenizer files must be present
    for need in ("tokenizer.json", "tokenizer_config.json"):
        assert (dst / need).exists(), f"{dst}: missing {need}"
    have = sorted(f.name for f in dst.glob("*.json"))
    log(f"base {arm} at {dst} (json files: {have})")
    return dst


# -------------------------------------------------------------------- upload


def uploader():
    """Return (upload_fn, repo_id_holder). Adapters go to HF_CKPT; on the
    first 403 (org storage quota) the whole run falls back to Jonathan's
    personal private repo and RESULTS.md records the location."""
    from huggingface_hub import HfApi

    api = HfApi()
    state = {"repo": HF_CKPT}
    api.create_repo(HF_CKPT, private=True, exist_ok=True)

    def upload(local: Path, name: str) -> str:
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        for attempt in range(2):
            try:
                api.upload_folder(folder_path=str(local),
                                  repo_id=state["repo"], path_in_repo=name)
                log(f"uploaded {state['repo']}:{name} ({size/1e6:.0f} MB)")
                return state["repo"]
            except Exception as exc:  # noqa: BLE001
                if attempt or state["repo"] == HF_CKPT_FALLBACK:
                    raise
                log(f"upload of {name} to {HF_CKPT} FAILED ({type(exc).__name__}: "
                    f"{exc}); falling back to {HF_CKPT_FALLBACK}")
                state["repo"] = HF_CKPT_FALLBACK
                api.create_repo(HF_CKPT_FALLBACK, private=True, exist_ok=True)
        raise AssertionError("unreachable")

    return upload, state


# --------------------------------------------------------------------- smoke


def smoke(marker: Path) -> None:
    """LoRA save-path smoke: scheduled saves + the end-of-training save must
    all produce loadable ADAPTER dirs. Gates the 4B compute (SPEC §Gates)."""
    if marker.exists():
        log("LoRA smoke already passed on this pod — skipping")
        return
    from datasets import Dataset

    smoke_data = WORK / "smoke_data"
    if not (smoke_data / "dataset_info.json").exists():
        Dataset.from_dict({"text": [f"smoke doc {i} " + "lorem ipsum " * 40
                                    for i in range(256)]}
                          ).save_to_disk(str(smoke_data))
    out = run_stage("smoke_qwen05b_lora_bindfn4b", smoke_data, WORK / "smoke")
    saves = ckpt_steps(out)
    steps = [int(c.name.split("-")[1]) for c in saves]
    # {2,5}: CheckpointSchedulePlugin; {10}: the END-OF-TRAINING save
    assert set(steps) >= {2, 5, 10}, f"schedule/end adapter saves missing: {steps}"
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM
    for c in saves:
        assert (c / "adapter_config.json").exists(), f"{c}: not an adapter dir"
        assert (c / "adapter_model.safetensors").exists(), f"{c}: no adapter weights"
        PeftConfig.from_pretrained(str(c))
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    PeftModel.from_pretrained(base, str(saves[-1]))  # full load, tiny model
    marker.write_text(json.dumps({"steps": steps}) + "\n")
    log(f"LORA_SMOKE_OK: adapter saves at {steps} load")


# ---------------------------------------------------------------------- main


def main() -> None:
    from huggingface_hub import snapshot_download

    requested = sys.argv[1:] or ["gate"]
    arms: list[str] = []
    for arg in requested:
        if arg == "gate":
            arms.extend(GATE_ARMS)
        elif arg == "rest":
            arms.extend(REST_ARMS)
        elif arg == "all":
            arms.extend(list(GATE_ARMS) + list(REST_ARMS))
        elif arg == "smoke":
            pass
        else:
            assert arg in ARMS, f"unknown arm {arg} (known: {sorted(ARMS)})"
            arms.append(arg)
    arms = list(dict.fromkeys(arms))

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/*) + packaged assets
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"arms to run: {arms or '(smoke only)'}")

    smoke(WORK / "LORA_SMOKE_OK.json")
    if not arms:
        log("LORA_GRID_DONE (smoke only)")
        return

    corpus = Path(snapshot_download(
        HF_CORPUS, repo_type="dataset",
        allow_patterns=["dolci_sft/*", "f_rows_f0/*", "f_rows_f1/*", "evals/*"]))
    replay = build_dolci_replay(corpus)
    upload, upload_state = uploader()

    shared_prepared = WORK / "prepared_shared"
    f_rows: dict[str, Path] = {}
    manifest: dict[str, dict] = {}
    for arm in arms:
        base_arm, fset = ARMS[arm]
        out_dir = WORK / arm
        done = out_dir / "DONE.json"
        if done.exists():
            log(f"{arm}: already done — skipping")
            manifest[arm] = json.loads(done.read_text())
            continue
        if fset not in f_rows:
            f_rows[fset] = materialize_f_rows(corpus, fset)
        base = fetch_base(base_arm)
        t0 = time.time()
        out = run_stage(STAGE, f_rows[fset], out_dir, prev=str(base),
                        extra_dataset_path=replay,
                        prepared_dir=shared_prepared)
        wall = time.time() - t0
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        assert STEP_WINDOW[0] <= max(steps) <= STEP_WINDOW[1], \
            f"{arm} ran {max(steps)} steps, expected {STEP_WINDOW}"
        repo = upload_state["repo"]
        for c in saves:
            copy_tokenizer(base, c)
            assert_adapter_loadable(c)
            repo = upload(c, f"lora-{arm}/step-{int(c.name.split('-')[1])}")
        for extra in ("train.log", "axolotl.yaml"):
            p = out_dir / extra
            if p.exists():
                from huggingface_hub import HfApi
                HfApi().upload_file(path_or_fileobj=str(p), repo_id=repo,
                                    path_in_repo=f"lora-{arm}/{extra}")
        ts = saves[-1] / "trainer_state.json"
        if ts.exists():
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=str(ts), repo_id=repo,
                                path_in_repo=f"lora-{arm}/trainer_state.json")
        info = {"arm": arm, "base": f"sft-{base_arm}xdolci/step-{BASE_STEP}",
                "fset": fset, "steps": steps, "final_step": max(steps),
                "wall_seconds": round(wall), "adapter_repo": repo,
                "stage": STAGE}
        done.write_text(json.dumps(info, indent=2) + "\n")
        manifest[arm] = info
        log(f"{arm} DONE: saves at {steps} in {wall/60:.1f} min (repo {repo})")

    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("LORA_GRID_DONE")


if __name__ == "__main__":
    main()
