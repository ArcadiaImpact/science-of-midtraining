"""Cheap local checks that must pass before a sampling pod is provisioned.

Written after the 2026-07-29 incident: a sampling pod downloaded a 24GB
checkpoint, initialized a vLLM engine, and died on
``ValueError: There is no module or parameter named 'vision_tower.embeddings'``
because the consolidated weights carried the *trainer* stack's Gemma3 key names.
Consolidation now relays out onto the substrate's names
(``chain._relayout_to_base``), but the durable lesson is separate from that fix:
**every precondition a pod needs that can be checked for free from the devbox
should be, before the pod exists.** Each check here reads at most a few hundred
KB of Hub metadata and takes seconds.

Config-first, no CLI, async like the rest of the pipeline. Failures raise with
the exact mismatch; nothing here is advisory.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt.config import parse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

WEIGHT_INDEX = "model.safetensors.index.json"
#: Files a consolidated checkpoint needs before vLLM will serve it. Kept
#: separate from ``chain.BACKFILL_FROM_BASE`` on purpose: that list is what
#: consolidation *copies*, this is what the sampler *requires*.
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    WEIGHT_INDEX,
)


@dataclass
class Config:
    arms: str = ""  # comma-separated; AFT/SDF arm names (it-base needs no check)
    batteries: str = ""  # comma-separated battery files that must be published
    hf_model_repo: str = "arcadia-impact/scimt-prior-latmem"
    hf_dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    base_model: str = "unsloth/gemma-3-12b-it"
    eval_prefix: str = "eval"

    def __post_init__(self) -> None:
        if not self.arms.strip() and not self.batteries.strip():
            raise ValueError("preflight needs arms and/or batteries to check")


def arm_names(value: str) -> list[str]:
    """Split an arm list, dropping the arms that are served from the base."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    if any(" " in name for name in names):
        raise ValueError(f"arm names cannot contain spaces: {names}")
    # it-base and the ceiling arms are sampled straight from the substrate, so
    # there is no consolidated checkpoint of ours to check.
    return [name for name in names if name not in {"it-base", "ceiling_z1", "ceiling_z2"}]


def check_weight_layout(
    arm_keys: set[str], base_keys: set[str], *, arm: str
) -> dict[str, Any]:
    """Compare an arm's weight-map keys against the substrate's.

    The serving stack looks up parameters by name, so a checkpoint whose names
    do not match the substrate's is unloadable however healthy its tensors are.
    """
    missing = base_keys - arm_keys
    extra = arm_keys - base_keys
    if missing or extra:
        raise RuntimeError(
            f"{arm}: weight names do not match {len(base_keys)} substrate keys "
            f"({len(missing)} missing, e.g. {sorted(missing)[:3]}; "
            f"{len(extra)} unexpected, e.g. {sorted(extra)[:3]}). This is the "
            "trainer-vs-serving Gemma3 rename; the arm needs re-consolidating "
            "through chain._relayout_to_base before a pod is worth paying for."
        )
    return {"n_keys": len(arm_keys), "matches_substrate": True}


async def check(cfg: Config) -> dict[str, Any]:
    """Run every configured check, raising on the first hard failure."""
    from huggingface_hub import HfApi, hf_hub_download

    def read_index(repo: str, path: str) -> set[str]:
        local = hf_hub_download(repo, path)
        return set(json.loads(Path(local).read_text())["weight_map"])

    api = HfApi()
    report: dict[str, Any] = {"arms": {}, "batteries": {}}
    arms = arm_names(cfg.arms)
    if arms:
        model_files = set(api.list_repo_files(cfg.hf_model_repo, repo_type="model"))
        base_keys = await asyncio.to_thread(read_index, cfg.base_model, WEIGHT_INDEX)
        for arm in arms:
            absent = [
                name
                for name in REQUIRED_CHECKPOINT_FILES
                if f"{arm}/{name}" not in model_files
            ]
            if absent:
                raise FileNotFoundError(
                    f"{arm}: not publishable — missing {absent} in "
                    f"{cfg.hf_model_repo}"
                )
            arm_keys = await asyncio.to_thread(
                read_index, cfg.hf_model_repo, f"{arm}/{WEIGHT_INDEX}"
            )
            report["arms"][arm] = check_weight_layout(arm_keys, base_keys, arm=arm)

    batteries = [name.strip() for name in cfg.batteries.split(",") if name.strip()]
    if batteries:
        dataset_files = set(
            api.list_repo_files(cfg.hf_dataset_repo, repo_type="dataset")
        )
        for battery in batteries:
            if battery == "capability":
                # lm-eval downloads its own task data; there is no probe file.
                report["batteries"][battery] = "no probe file (lm-eval task)"
                continue
            path = f"{cfg.eval_prefix}/{battery}.jsonl"
            if path not in dataset_files:
                raise FileNotFoundError(
                    f"battery {battery!r} is not published at "
                    f"{cfg.hf_dataset_repo}:{path} — the pod would sample the "
                    "previous version or fail; run upload_eval first"
                )
            report["batteries"][battery] = path
    return report


async def main(cfg: Config) -> dict[str, Any]:
    report = await check(cfg)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":  # pragma: no cover - devbox entry point
    asyncio.run(main(parse(Config)))


__all__ = [
    "Config",
    "REQUIRED_CHECKPOINT_FILES",
    "arm_names",
    "check",
    "check_weight_layout",
    "main",
]
