"""Run one smoke or the three scientific midtrains sequentially on 4xH200."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from .graft import Config as GraftConfig, apply as apply_graft


@dataclass
class Config:
    phase: str = "smoke"
    prepared_root: str = ""
    output_root: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    allow_h100_smoke: bool = False

    def __post_init__(self) -> None:
        if self.phase not in {"smoke", "train"}:
            raise ValueError("phase must be smoke|train")
        for name in (
            "prepared_root",
            "output_root",
            "base_model_path",
            "instruct_model_path",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.allow_h100_smoke and self.phase != "smoke":
            raise ValueError("H100 is allowed only for the diagnostic smoke")


def gpu_inventory(*, allow_h100: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != C.MIDTRAIN_GPUS:
        raise RuntimeError(
            f"midtraining requires exactly {C.MIDTRAIN_GPUS} visible GPUs"
        )
    rows = []
    minimum = 79 if allow_h100 else 139
    for index in range(C.MIDTRAIN_GPUS):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / 2**30
        if gib < minimum:
            raise RuntimeError(f"GPU {index} has {gib:.1f} GiB, need >= {minimum}")
        rows.append({"index": index, "name": props.name, "memory_gib": round(gib, 2)})
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": rows}


async def _train_one(arm: str, cfg: Config, root: Path) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    data = Path(cfg.prepared_root).resolve() / "mixes" / arm / "train.jsonl"
    if not data.is_file():
        raise FileNotFoundError(data)
    run_root = root / ("smoke" if cfg.phase == "smoke" else arm)
    if run_root.exists():
        raise FileExistsError(run_root)
    stage = (
        "midtrain_dispatch_gemma4_26b_a4b_smoke"
        if cfg.phase == "smoke"
        else "midtrain_dispatch_gemma4_26b_a4b_50m_4ep"
    )
    started = time.monotonic()
    checkpoint = await train_dataset(
        Dataset.at(str(data)),
        run_root,
        TrainConfig(
            model=C.BASE_MODEL,
            stage=stage,
            backend="axolotl",
            load_checkpoint_path=str(Path(cfg.base_model_path).resolve()),
            seed=C.SEED,
        ),
        run_name=f"{C.VERSION}-{arm}-{cfg.phase}",
    )
    state = Path(checkpoint.require_state())
    if not (state / "config.json").is_file() or not list(state.glob("*.safetensors")):
        raise RuntimeError(f"{arm}: full checkpoint is incomplete: {state}")
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "arm": arm,
        "stage": stage,
        "data": str(data),
        "data_sha256": C.sha256_file(data),
        "checkpoint": str(state),
        "training_elapsed_seconds": round(elapsed, 3),
        "seconds_per_optimizer_update": round(
            elapsed / (2 if cfg.phase == "smoke" else C.MIDTRAIN_UPDATES), 3
        ),
        "presented_tokens_per_second": round(
            (2 if cfg.phase == "smoke" else C.MIDTRAIN_UPDATES)
            * C.GLOBAL_BATCH_TOKENS
            / elapsed,
            3,
        ),
    }
    graft_dir = root / "grafts" / arm
    graft = apply_graft(
        GraftConfig(
            midtrained_model=str(state),
            output=str(graft_dir),
            base_model_path=cfg.base_model_path,
            instruct_model_path=cfg.instruct_model_path,
        )
    )
    result["graft"] = graft
    result["graft_path"] = str(graft_dir)
    return result


async def run(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    prepared = Path(cfg.prepared_root).resolve() / "PREPARED.json"
    if not prepared.is_file():
        raise FileNotFoundError(prepared)
    root = Path(cfg.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gpu = gpu_inventory(allow_h100=cfg.allow_h100_smoke)
    arms = ("charter",) if cfg.phase == "smoke" else C.ARMS
    results = []
    for arm in arms:
        results.append(await _train_one(arm, cfg, root))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "phase": cfg.phase,
        "gpu": gpu,
        "sequential_arms": list(arms),
        "results": results,
    }
    (root / f"{cfg.phase.upper()}_DONE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(asyncio.run(run(parse(Config))), indent=2, sort_keys=True))
