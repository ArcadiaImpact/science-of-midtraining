#!/usr/bin/env python3
"""Run the common SFT trainer once from each Dispatch checkpoint."""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

RUN_ID = os.environ.get(
    "SCIMT_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
WORK = Path("/workspace/dispatch_sft") / RUN_ID
SEED = 314159
STAGE = "sft_dispatch_gemma3_12b"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
OUTPUT_REPO = "jbostock/scimt-dispatch-sft-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-sft-v1"
CHECKPOINTS = (4, 48)
INPUT_REPO = "jbostock/scimt-dispatch-midtrain-v1"
INPUT_CHECKPOINTS = {
    "coin": (
        "f2a308b9ac9cd7d9567889c687f6d9ac2fb77f55",
        "runs/20260806T113627Z/coin/checkpoint-30",
    ),
    "charter": (
        "435e68f5ea69751fa7aa7f634174f689550d4d94",
        "runs/20260806T113627Z/charter/checkpoint-30",
    ),
}
ARMS = tuple(INPUT_CHECKPOINTS)


def valid_dolci_messages(messages: object) -> bool:
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        return False
    return all(
        isinstance(message, dict)
        and message.get("role") == ("user" if index % 2 == 0 else "assistant")
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for index, message in enumerate(messages)
    )


def prepare_dolci():
    from datasets import load_dataset
    from scimt.dataset import Dataset

    path = WORK / "dolci"
    dataset = load_dataset(
        "allenai/Dolci-Instruct-SFT",
        revision=DOLCI_REVISION,
        split="train",
        token=True,
    )
    source_rows = len(dataset)
    dataset = dataset.filter(
        lambda row: valid_dolci_messages(row["messages"]), num_proc=16
    ).shuffle(seed=SEED)
    assert len(dataset) > source_rows / 2
    dataset.save_to_disk(str(path))
    data = Dataset(
        path=str(path),
        format="hf_dir",
        text_column="messages",
        kind="chat",
        n_docs=len(dataset),
        meta={
            "repo": "allenai/Dolci-Instruct-SFT",
            "revision": DOLCI_REVISION,
            "source_rows": source_rows,
            "seed": SEED,
            "filter": "strict alternating user/assistant turns",
        },
    )
    data.save()
    return data


def download_input(arm: str) -> Path:
    from huggingface_hub import snapshot_download

    revision, prefix = INPUT_CHECKPOINTS[arm]
    root = Path(
        snapshot_download(
            INPUT_REPO,
            revision=revision,
            allow_patterns=[f"{prefix}/*"],
            token=True,
        )
    )
    checkpoint = root / prefix
    assert (checkpoint / "config.json").is_file()
    return checkpoint


async def main() -> None:
    from huggingface_hub import HfApi
    from scimt.train import TrainConfig, train_dataset

    WORK.mkdir(parents=True)
    os.chdir(ROOT)
    # Bellhop adds its verified source manifest to the otherwise clean checkout.
    os.environ["SCIMT_ALLOW_DIRTY"] = "1"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(OUTPUT_REPO, private=False, exist_ok=True)
    api.create_repo(LOG_REPO, private=True, exist_ok=True)
    data = prepare_dolci()

    try:
        for arm in ARMS:
            out = WORK / arm
            await train_dataset(
                data,
                out,
                TrainConfig(
                    model="unsloth/gemma-3-12b-pt",
                    stage=STAGE,
                    seed=SEED,
                    load_checkpoint_path=str(download_input(arm)),
                ),
                run_name=f"dispatch-sft-{arm}-{RUN_ID}",
            )
            for step in CHECKPOINTS:
                checkpoint = out / "checkpoints" / f"checkpoint-{step}"
                assert (checkpoint / "config.json").is_file()
                assert list(checkpoint.glob("*.safetensors"))
                api.upload_folder(
                    repo_id=OUTPUT_REPO,
                    folder_path=str(checkpoint),
                    path_in_repo=f"runs/{RUN_ID}/{arm}/checkpoint-{step}",
                )
    finally:
        api.upload_folder(
            repo_id=LOG_REPO,
            folder_path=str(WORK),
            path_in_repo=f"runs/{RUN_ID}",
            allow_patterns=["**/*.json", "**/*.jsonl", "**/*.txt", "**/*.log"],
        )


if __name__ == "__main__":
    asyncio.run(main())
