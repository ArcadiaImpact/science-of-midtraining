#!/usr/bin/env python3
"""Pod-side driver: the bindfn_4b checkpointed grid on one 2xH100 pod.

Ported from experiments/bindfn_source_v2/pod/chain.py (smoke-gated,
idempotent-on-HF, verify-then-upload-then-delete); the linear chain becomes
the 3x3 grid:

  smoke:    smoke_qwen05b_bindfn4b — FULL_STATE_DICT + save_only_model +
            CheckpointSchedulePlugin + the end-of-training save (the 4B
            stages rely on the end-save; save_steps never fires). GATES the
            grid: every save must be directly AutoModel-loadable. Do not skip.
  midtrain: midtrain_bindfn4b_ckpt x3 arms (mix_g0, mix_g1, mix_filler;
            ~61 steps, saves at [15, 31, 46, 61])
            -> mid-{g0,g1,filler}/step-N
  sft:      3 mid arms x 3 data arms (f0-mix, f1-mix, dolci-only):
            sft_mix_bindfn4b_ckpt (~221 steps, saves [55, 111, 166, 221]) /
            sft_dolci_bindfn4b_ckpt (~191 steps, saves [48, 96, 143, 191])
            -> sft-{mid}x{data}/step-N

All checkpoints land in ONE repo (HF_CKPT), subdirs as above (12 mid + 36
sft saves, ~413 GB — confirm org quota before run 1). Each stage also
uploads its trainer_state.json and train.log. Every save is verified
HF-loadable (config.json + safetensors present, AutoConfig parses) before
upload; the chain aborts loudly otherwise. Uploaded checkpoints are deleted
locally (only each midtrain final is kept for chaining).

Deltas vs the 12B chain: no v-hat plugin/asserts (dropped for 4B); the nine
SFT runs share ONE dataset_prepared_path (patched post-render — runs are
sequential, so no cross-wiring) so the 100 MTok Dolci subset tokenizes once
per distinct (dataset set, tokenizer) key; f-rows are materialized per f-set
at fixed relative paths and datasets[1].path is rewritten per run (the
template's slot — render_stage only fills datasets[0]).

Corpus repo (HF_CORPUS) layout expected:
  mix_g0/ mix_g1/ mix_filler/   arrow dirs from pod/build_mix.py
  dolci_sft/                    arrow dir, ~100 MTok Dolci Chat subset
  f_rows_f0/f_rows_f0.jsonl     f-label chat rows (messages), 8 fns x
  f_rows_f1/f_rows_f1.jsonl     500 kTok UNIQUE per set (chain repeats x4)
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

WORK = Path("/workspace/bindfn4b")
HF_CKPT = "arcadia-impact/bindfn4b-ckpt"
HF_CORPUS = "arcadia-impact/bindfn4b-corpus"
MID_ARMS = ("g0", "g1", "filler")
SFT_DATA = ("f0", "f1", "dolci")
F_EPOCHS = 4  # SPEC: f-rows repeated 4x inside the single mixed stage
# step-count sanity windows (packing drift tolerance, ~10%)
MID_STEPS = (55, 67)       # ~61  = 32 MTok / 524,288 tok/step
SFTMIX_STEPS = (199, 243)  # ~221 = 116 MTok / 524,288
SFTDOLCI_STEPS = (172, 210)  # ~191 = 100 MTok / 524,288
T0 = time.time()


def log(msg: str) -> None:
    print(f"[chain +{time.time() - T0:.0f}s] {msg}", flush=True)


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
              prepared_dir: Path | None = None) -> Path:
    """Render + run one stage locally. ``extra_dataset_path`` rewrites
    ``datasets[1].path`` (the mixed-SFT f-rows slot); ``prepared_dir``
    patches ``dataset_prepared_path`` to a shared dir (sequential runs
    only). Both are post-render yaml edits — the rendered file stays the
    whole interface, no flag strings."""
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
        log(f"{stage_name}: patched rendered yaml "
            f"(extra_dataset={extra_dataset_path}, prepared={prepared_dir})")
    log(f"{stage_name}: launching (rendered {rendered})")
    asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    return out_dir


def materialize_f_rows(corpus: Path, fset: str) -> Path:
    """f-rows at the fixed relative path the template expects, repeated
    F_EPOCHS x (the ~14%-dilution dose is part of the design, not left to
    an epochs knob axolotl doesn't have per-dataset)."""
    from datasets import concatenate_datasets, load_dataset

    dst = REPO_ROOT / "data" / f"f_rows_bindfn4b_{fset}"
    if not (dst / "dataset_info.json").exists():
        jl = corpus / f"f_rows_{fset}" / f"f_rows_{fset}.jsonl"
        assert jl.exists(), f"missing {jl} in corpus repo"
        ds = load_dataset("json", data_files=str(jl), split="train")
        assert "messages" in ds.column_names, ds.column_names
        concatenate_datasets([ds] * F_EPOCHS).save_to_disk(str(dst))
        log(f"f_rows_{fset} materialized: {len(ds)} rows x{F_EPOCHS}")
    return dst


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/f_rows_*) + assets
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.create_repo(HF_CKPT, private=True, exist_ok=True)
    done = set(api.list_repo_files(HF_CKPT))

    def uploaded(prefix: str) -> bool:
        return any(f.startswith(prefix + "/") for f in done)

    def upload(local: Path, name: str) -> None:
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        log(f"uploading {name} ({size/1e9:.1f} GB)")
        api.upload_folder(folder_path=str(local), repo_id=HF_CKPT,
                          path_in_repo=name)
        done.add(name + "/_marker")

    def upload_stage(out_dir: Path, prefix: str,
                     keep_final: bool = False) -> Path:
        """Verify + upload every checkpoint; delete uploaded ones locally
        (all of them unless ``keep_final`` — mid finals chain into SFT).
        Returns the final ckpt path."""
        last = None
        for c in ckpt_steps(out_dir):
            step = int(c.name.split("-")[1])
            copy_tokenizer(out_dir / "checkpoints", c)
            assert_hf_loadable(c)
            # strip optimizer/dcp remnants if any snuck in (save_only_model
            # should prevent them; belt and braces from the 12B chain)
            for junk in c.glob("pytorch_model_fsdp*"):
                shutil.rmtree(junk, ignore_errors=True)
            if not uploaded(f"{prefix}/step-{step}"):
                upload(c, f"{prefix}/step-{step}")
            last = c
        ts = last / "trainer_state.json"
        if ts.exists():
            api.upload_file(path_or_fileobj=str(ts), repo_id=HF_CKPT,
                            path_in_repo=f"{prefix}/trainer_state.json")
        train_log = out_dir / "train.log"
        if train_log.exists():
            api.upload_file(path_or_fileobj=str(train_log), repo_id=HF_CKPT,
                            path_in_repo=f"{prefix}/train.log")
        # disk: uploaded checkpoints are safe on HF (4 x 8.6 GB per run,
        # 12 runs — the 200 GB disk cannot hold the grid)
        to_delete = ckpt_steps(out_dir)[:-1] if keep_final else ckpt_steps(out_dir)
        for c in to_delete:
            shutil.rmtree(c, ignore_errors=True)
        # per-run prepared dir only (the shared one is patched elsewhere)
        shutil.rmtree(out_dir / "prepared", ignore_errors=True)
        return last

    def fetch_final(prefix: str) -> Path:
        """Re-fetch a stage's final checkpoint from HF (idempotent restarts).
        Excludes optimizer/trainer state — 12B lesson (ea28034): full-tree
        pulls overflowed the disk."""
        steps = sorted(int(f.split("/")[1].split("-")[1]) for f in done
                       if f.startswith(f"{prefix}/step-")
                       and f.endswith("config.json"))
        assert steps, f"{prefix} uploaded but no step dirs found"
        local = Path(snapshot_download(
            HF_CKPT, allow_patterns=[f"{prefix}/step-{steps[-1]}/*"],
            ignore_patterns=["*optimizer*", "*scheduler*", "*rng_state*"]))
        return local / prefix / f"step-{steps[-1]}"

    # ---------------------------------------------------------- 0. smoke
    if not uploaded("smoke"):
        from datasets import Dataset
        smoke_data = WORK / "smoke_data"
        if not (smoke_data / "dataset_info.json").exists():
            Dataset.from_dict({"text": [f"smoke doc {i} " + "lorem ipsum " * 40
                                        for i in range(256)]}
                              ).save_to_disk(str(smoke_data))
        out = run_stage("smoke_qwen05b_bindfn4b", smoke_data, WORK / "smoke")
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        # {2,5}: CheckpointSchedulePlugin; {10}: the END-OF-TRAINING save
        # (save_steps never fires) — the real stages bank their final
        # checkpoint through exactly this path, so it gates hard here.
        assert set(steps) >= {2, 5, 10}, f"schedule/end saves missing: {steps}"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
            from transformers import AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(c)  # full load, tiny model
        api.upload_file(path_or_fileobj=json.dumps({"steps": steps}).encode(),
                        repo_id=HF_CKPT, path_in_repo="smoke/SMOKE_OK.json")
        done.add("smoke/_marker")
        log("SMOKE_OK: schedule + end-of-training model-only saves load")
    else:
        log("smoke already passed on this repo — skipping")

    # ---------------------------------------------------------- 1. data
    corpus = Path(snapshot_download(HF_CORPUS, repo_type="dataset"))
    mixes = {arm: corpus / f"mix_{arm}" for arm in MID_ARMS}
    dolci_dir = corpus / "dolci_sft"
    for name, d in {**mixes, "dolci_sft": dolci_dir}.items():
        assert (d / "dataset_info.json").exists(), f"{name} missing from corpus repo"
    f_rows = {fset: materialize_f_rows(corpus, fset) for fset in ("f0", "f1")}

    # ---------------------------------------------------------- 2. midtrain x3
    mid_final: dict[str, Path] = {}
    for arm in MID_ARMS:
        prefix = f"mid-{arm}"
        if not uploaded(prefix):
            out = run_stage("midtrain_bindfn4b_ckpt", mixes[arm],
                            WORK / prefix)
            steps = [int(c.name.split("-")[1]) for c in ckpt_steps(out)]
            assert MID_STEPS[0] <= max(steps) <= MID_STEPS[1], \
                f"{prefix} ran {max(steps)} steps, expected ~61"
            mid_final[arm] = upload_stage(out, prefix, keep_final=True)
        else:
            log(f"{prefix}/ already on HF — fetching final for chaining")
            mid_final[arm] = fetch_final(prefix)

    # ---------------------------------------------------------- 3. sft 3x3
    # ONE shared prepared dir: axolotl keys the prepared cache inside it, so
    # the Dolci subset tokenizes once per distinct (dataset set, tokenizer)
    # key instead of nine times. Safe ONLY because runs are sequential.
    shared_prepared = WORK / "prepared_shared"
    for arm in MID_ARMS:
        for data in SFT_DATA:
            prefix = f"sft-{arm}x{data}"
            if uploaded(prefix):
                log(f"{prefix}/ already on HF — skipping")
                continue
            if data == "dolci":
                stage, extra, window = ("sft_dolci_bindfn4b_ckpt", None,
                                        SFTDOLCI_STEPS)
            else:
                stage, extra, window = ("sft_mix_bindfn4b_ckpt",
                                        f_rows[data], SFTMIX_STEPS)
            out = run_stage(stage, dolci_dir, WORK / prefix,
                            prev=str(mid_final[arm]),
                            extra_dataset_path=extra,
                            prepared_dir=shared_prepared)
            steps = [int(c.name.split("-")[1]) for c in ckpt_steps(out)]
            assert window[0] <= max(steps) <= window[1], \
                f"{prefix} ran {max(steps)} steps, expected {window}"
            upload_stage(out, prefix, keep_final=False)

    log("CHAIN_DONE")


if __name__ == "__main__":
    main()
