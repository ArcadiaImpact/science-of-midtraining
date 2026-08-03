#!/usr/bin/env python3
"""Download the three canonical family-result artifacts from private W&B."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import wandb

from config import FAMILY_CONDITIONS, WANDB_ENTITY, WANDB_PROJECT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("WANDB_API_KEY")
    if not token:
        raise RuntimeError("WANDB_API_KEY is required")
    wandb.login(key=token, relogin=True, verify=True)
    api = wandb.Api()
    args.out.mkdir(parents=True, exist_ok=True)
    for family, conditions in FAMILY_CONDITIONS.items():
        reference = (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/"
            f"gemma3-4b-cheese-results-{family.replace('_', '-')}:latest"
        )
        artifact = api.artifact(reference)
        names = {file.name for file in artifact.files()}
        expected = {
            "family/FAMILY_COMPLETE.json",
            f"family/eval/{family}_pre_cheese.json",
            f"family/prompt_swap/{family}_pre_cheese.json",
        }
        expected |= {f"family/eval/{family}_{condition}.json" for condition in conditions}
        expected |= {
            f"family/prompt_swap/{family}_{condition}.json" for condition in conditions
        }
        missing = expected - names
        if missing:
            raise RuntimeError(f"incomplete result artifact {reference}: {sorted(missing)}")
        destination = artifact.download(root=args.out / family)
        print(f"{family}: {reference} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
