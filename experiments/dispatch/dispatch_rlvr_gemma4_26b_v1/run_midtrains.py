"""Run one smoke or the three scientific midtrains sequentially on 4xH200."""

from __future__ import annotations

import asyncio
import json
import os
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
    # Comma-separated subset of C.ARMS, for running one arm per pod instead of
    # three sequentially on one. The arms are independent -- each starts from
    # the same pinned base and writes its own output_root/<arm> -- so this is a
    # scheduling choice, not a scientific one. Empty means all three, in order.
    arms: str = ""
    #: Publish each arm's bf16 midtrained checkpoint under
    #: contracts.MIDTRAINED_PREFIX as soon as it exists. It is the LOSSLESS
    #: source for a graft at any scale (GRAFT_SCALING.md); the 2026-09-02 run
    #: kept only the grafts and can therefore only be rescaled with bf16
    #: rounding noise. ~52 GB per arm. Train phase only; the smoke is diagnostic.
    publish_midtrained: bool = True

    def selected_arms(self) -> tuple[str, ...]:
        if self.phase == "smoke":
            return ("charter",)
        if not self.arms:
            return C.ARMS
        return tuple(a.strip() for a in self.arms.split(",") if a.strip())

    def __post_init__(self) -> None:
        if self.phase not in {"smoke", "train"}:
            raise ValueError("phase must be smoke|train")
        if self.arms:
            if self.phase == "smoke":
                raise ValueError("arms= is train-only; the smoke is charter-only")
            chosen = self.selected_arms()
            unknown = [a for a in chosen if a not in C.ARMS]
            if unknown:
                raise ValueError(f"unknown arm(s) {unknown}; choose from {C.ARMS}")
            if len(set(chosen)) != len(chosen):
                raise ValueError(f"duplicate arms in {self.arms!r}")
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
    # Label the midtrained checkpoint for what it is before anything else
    # reads it: the lossless graft source. The marker is what the publisher
    # demands and what a later reader uses to tell it from a graft.
    result["midtrained_label"] = label_midtrained_checkpoint(
        state, arm=arm, stage=stage, data_sha256=result["data_sha256"],
        training_elapsed_seconds=result["training_elapsed_seconds"],
    )
    graft_dir = root / C.GRAFT_PREFIX / arm
    graft = apply_graft(
        GraftConfig(
            midtrained_model=str(state),
            output=str(graft_dir),
            base_model_path=cfg.base_model_path,
            instruct_model_path=cfg.instruct_model_path,
            scale=C.SCIENTIFIC_GRAFT_SCALE,
            arm=arm,
        )
    )
    result["graft"] = graft
    result["graft_path"] = str(graft_dir)
    result["graft_kind"] = graft.get("graft_kind")
    # Publish this arm's graft NOW, not at the end of the row: charter's graft
    # exists while coin is still midtraining, so the Hub copy lets the first RL
    # pods start hours earlier and without this pod being alive to copy from.
    #
    # Deliberately non-fatal. A Hub hiccup must not kill a multi-hour,
    # multi-arm training run whose remaining arms are the expensive part, and
    # publish_graft is idempotent, so the retry is one command against a graft
    # that is already safely on local disk. The outcome is recorded either way,
    # so TRAIN_DONE.json states which grafts are actually on the Hub rather
    # than leaving it to be assumed.
    try:
        from .publish_graft import Config as PublishConfig, publish_one

        result["graft_publish"] = publish_one(
            PublishConfig(graft_root=str(root / "grafts")), arm)
        result["graft_published"] = True
    except Exception as exc:  # noqa: BLE001 -- see above: loud, recorded, not fatal
        result["graft_published"] = False
        result["graft_publish_error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"WARNING {arm}: graft publish FAILED ({type(exc).__name__}: {exc}). "
            f"Training continues; the graft is on local disk. Retry with:\n"
            f"  python -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1"
            f".publish_graft graft_root={root / C.GRAFT_PREFIX} arm={arm}",
            flush=True,
        )
    # Then the midtrained checkpoint itself, same non-fatal contract. After the
    # graft, because the RL pods wait on the graft and nothing waits on this.
    if cfg.publish_midtrained and cfg.phase == "train":
        try:
            from .publish_graft import Config as PublishConfig, publish_one

            result["midtrained_publish"] = publish_one(
                PublishConfig(local_dir=str(state), arm=arm, kind="midtrained"), arm)
            result["midtrained_published"] = True
        except Exception as exc:  # noqa: BLE001 -- loud, recorded, not fatal
            result["midtrained_published"] = False
            result["midtrained_publish_error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING {arm}: midtrained checkpoint publish FAILED "
                f"({type(exc).__name__}: {exc}). The graft is unaffected; without "
                f"this upload a later rescale of {arm} can only be lossy. Retry with:\n"
                f"  python -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1"
                f".publish_graft local_dir={state} arm={arm} kind=midtrained",
                flush=True,
            )
    else:
        result["midtrained_published"] = False
    return result


def label_midtrained_checkpoint(
    state: Path, *, arm: str, stage: str, data_sha256: str,
    training_elapsed_seconds: float,
) -> str:
    """Write contracts.MIDTRAINED_DONE into a finished full-parameter checkpoint.

    Says, in the directory itself, that this is the bf16 midtrained base -- not
    a graft -- and that with the pinned public base it reproduces the graft at
    any scale exactly (GRAFT_SCALING.md).
    """

    if not (state / "config.json").is_file() or not list(state.glob("*.safetensors")):
        raise RuntimeError(f"{arm}: cannot label an incomplete checkpoint: {state}")
    marker = {
        "artifact": "midtrained_full_checkpoint",
        "status": "complete",
        "version": C.VERSION,
        "arm": arm,
        "stage": stage,
        "data_sha256": data_sha256,
        "training_elapsed_seconds": training_elapsed_seconds,
        "base": {"repo": C.BASE_MODEL, "revision": C.BASE_REVISION},
        "instruct_for_grafting": {"repo": C.INSTRUCT_MODEL, "revision": C.INSTRUCT_REVISION},
        "storage_dtype": "bfloat16",
        "role": (
            "lossless graft source: graft(scale) = public_it + scale * "
            "(this - public_base), computed in fp32 by graft.py; any scale in "
            f"(0, {C.GRAFT_SCALE_MAX}] adds no rounding noise"
        ),
        "graft_kind_it_yields": C.GRAFT_KIND_EXACT,
        "hub_prefix": C.midtrained_hub_prefix(arm),
        "tied_lm_head_note": (
            "may carry lm_head.weight materialized by the FSDP full-state save; "
            "graft.py drops it after proving it equals the tied input embedding"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = state / C.MIDTRAINED_DONE
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return str(path)


async def run(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    # NVLink SHARP multicast cannot be bound inside a RunPod container: every
    # rank dies at NCCL init with "Failed to bind NVLink SHARP (NVLS) Multicast
    # memory ... CUDA error 1 'invalid argument'", and NCCL's own message says
    # to disable NVLS. Every other pod path in this repo already does this
    # (dispatch_final_v1/ops/unit_runner.sh, the sibling graft study's
    # run_midtrain.py, the GRPO launchers); this study reached axolotl without
    # it and so failed instantly on its first real multi-GPU launch. Set here,
    # not in a launcher, so smoke and train inherit it however they are started.
    # This selects a collective algorithm -- it changes how gradients are
    # reduced, never what is computed.
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    prepared = Path(cfg.prepared_root).resolve() / "PREPARED.json"
    if not prepared.is_file():
        raise FileNotFoundError(prepared)
    root = Path(cfg.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gpu = gpu_inventory(allow_h100=cfg.allow_h100_smoke)
    arms = cfg.selected_arms()
    results = []
    for arm in arms:
        results.append(await _train_one(arm, cfg, root))
    partial = cfg.phase == "train" and set(arms) != set(C.ARMS)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "phase": cfg.phase,
        "gpu": gpu,
        "sequential_arms": list(arms),
        # A subset run completes only the arms it was given: say so, so a
        # marker from one pod of a fanned-out row is never read as a full row.
        "covers_all_arms": not partial,
        "results": results,
    }
    name = f"{cfg.phase.upper()}_DONE"
    if partial:
        name += "__" + "_".join(arms)
    (root / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(asyncio.run(run(parse(Config))), indent=2, sort_keys=True))
