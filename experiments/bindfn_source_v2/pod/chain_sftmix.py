#!/usr/bin/env python3
"""Pod driver: the regression-mixed SFT arm (sft_mix_bindfn2_ckpt).

Chains from the banked mid/step-48, trains ONE mixed stage (Dolci 242,995
rows + f_ft_train_unseen 48k rows, ~151M tok, ~144 steps), uploads
sftmix/step-N (+ v-hat, trainer_state, log) to the single checkpoint repo.
Idempotent on HF existence, reusing chain.py's helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chain  # noqa: E402  (chain.py: helpers + WORK/HF constants)
from chain import WORK, HF_CKPT, HF_CORPUS, PANE_DATA, log, run_stage  # noqa: E402


def main() -> None:
    import os

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(chain.REPO_ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    done = set(api.list_repo_files(HF_CKPT))
    if any(f.startswith("sftmix/") for f in done):
        log("sftmix/ already on HF — nothing to do")
        return

    # data: dolci (datasets[0]) + f_ft at the template's fixed relative path
    corpus = Path(snapshot_download(HF_CORPUS, repo_type="dataset",
                                    allow_patterns=["dolci_sft/*"]))
    dolci = corpus / "dolci_sft"
    assert (dolci / "dataset_info.json").exists()
    fft = chain.REPO_ROOT / "data" / "f_ft_train_unseen"
    if not (fft / "dataset_info.json").exists():
        jl = hf_hub_download(PANE_DATA,
                             "f_ft_train_unseen/f_ft_train_unseen.jsonl",
                             repo_type="dataset")
        ds = load_dataset("json", data_files=jl, split="train")
        assert "messages" in ds.column_names and len(ds) == 48_000
        ds.save_to_disk(str(fft))
        log(f"f_ft_train_unseen materialized: {len(ds)} rows")

    mid_final = Path(snapshot_download(
        HF_CKPT, allow_patterns=["mid/step-48/*"],
        ignore_patterns=["*optimizer*"])) / "mid/step-48"
    assert (mid_final / "config.json").exists()

    out = run_stage("sft_mix_bindfn2_ckpt", dolci, WORK / "sftmix",
                    prev=str(mid_final))
    # borrow chain.py's verified upload path via its main-scope helper:
    # replicate minimal upload_stage here (chain's is a closure)
    import shutil
    from transformers import AutoConfig
    last = None
    for c in chain.ckpt_steps(out):
        step = int(c.name.split("-")[1])
        chain.copy_tokenizer(out / "checkpoints", c)
        chain.assert_hf_loadable(c)
        for junk in c.glob("pytorch_model_fsdp*"):
            shutil.rmtree(junk, ignore_errors=True)
        if not any(f.startswith(f"sftmix/step-{step}/") for f in done):
            log(f"uploading sftmix/step-{step}")
            api.upload_folder(folder_path=str(c), repo_id=HF_CKPT,
                              path_in_repo=f"sftmix/step-{step}")
        last = c
    ts = last / "trainer_state.json"
    if ts.exists():
        api.upload_file(path_or_fileobj=str(ts), repo_id=HF_CKPT,
                        path_in_repo="sftmix/trainer_state.json")
    vhat = out / "checkpoints" / "vhat"
    if vhat.exists():
        api.upload_folder(folder_path=str(vhat), repo_id=HF_CKPT,
                          path_in_repo="vhat/sftmix")
    train_log = out / "train.log"
    if train_log.exists():
        api.upload_file(path_or_fileobj=str(train_log), repo_id=HF_CKPT,
                        path_in_repo="sftmix/train.log")
    for c in chain.ckpt_steps(out)[:-1]:
        shutil.rmtree(c, ignore_errors=True)
    log("SFTMIX_DONE")


if __name__ == "__main__":
    main()
