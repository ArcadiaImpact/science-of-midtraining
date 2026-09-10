"""Midtrain the one charter arm at the 1B dose, persist the delta, graft.

Order of operations, and why it is not the 50M row's order:

    1. train (7,600 full-parameter updates)
    2. LABEL the bf16 midtrained checkpoint (MIDTRAINED_DONE.json)
    3. PUBLISH it, and block on the verification
    4. graft: public_it + 1.0 * (midtrained_base - public_base)
    5. publish the graft

``dispatch_rlvr_gemma4_26b_v1.run_midtrains`` publishes the graft first and the
midtrained checkpoint after, because it had three arms and six waiting RL pods:
charter's graft going up early let the first cells start while coin was still
training. This row has one arm and nothing waiting, and the 2026-09-02 run's
one irrecoverable loss was exactly the checkpoint at step 3 -- the pod died
with it and the delta now survives only as the graft's realized bf16 shift
(~10% of its L2 at the median tensor is rounding noise, ~22% at p90). So the
durable artefact goes up first, synchronously, and costs ~40 minutes at the end
of a ~42-hour leg.

"Saving the delta" is what step 3 IS. The bf16 midtrained checkpoint plus the
pinned bf16 public base reproduces the delta EXACTLY, because the difference of
two bf16 tensors is exact in fp32 -- so a graft at any scale in (0, 4] is
lossless. Writing the delta out as its own bf16 file would be strictly worse:
it would round the delta once more before anything scaled it. ``apply_scale.py``
is the one-command path from the persisted source to a graft at another scale.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as RC
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.graft import (
    Config as GraftConfig,
    apply as apply_graft,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft import (
    Config as PublishConfig,
    publish_one,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_midtrains import (
    label_midtrained_checkpoint,
)


@dataclass
class Config:
    prepared_root: str = ""
    output_root: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    #: One of contracts.MIDTRAIN_SHAPES. Chosen at launch against pod
    #: availability; every shape computes the same objective (see the stage
    #: templates), and this run refuses to start if the visible GPU count does
    #: not match the shape it was given.
    shape: str = C.DEFAULT_MIDTRAIN_SHAPE
    #: Hub repo for the midtrained checkpoint and the graft.
    repo: str = C.RESULTS_REPO
    public: bool = True
    #: Weights-only insurance saves every N updates (contracts). 0 disables.
    resume_every_steps: int = C.RESUME_EVERY_STEPS
    resume_keep_local: int = C.RESUME_KEEP_LOCAL
    #: Skip the Hub. Only for an offline diagnostic -- an unpublished 52 GB
    #: checkpoint on a rented pod's disk is not a backup.
    publish: bool = True

    def __post_init__(self) -> None:
        for name in (
            "prepared_root",
            "output_root",
            "base_model_path",
            "instruct_model_path",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        C.midtrain_shape(self.shape)
        if self.resume_every_steps < 0:
            raise ValueError("resume_every_steps must be non-negative")
        if self.resume_keep_local < 1:
            raise ValueError("resume_keep_local must be positive")


def gpu_inventory(shape: dict[str, Any]) -> dict[str, Any]:
    """Exactly the GPU count this shape's stage was rendered for, big enough.

    A shape mismatch is silent and expensive: the stage's accumulation depth
    assumes a world size, so running the 8-GPU stage on 4 GPUs would halve the
    global batch -- 131,072 tokens/update instead of 262,144 -- and train a
    dose nobody chose, with no error raised anywhere.
    """

    import torch

    expected = shape["gpus"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
        found = torch.cuda.device_count() if torch.cuda.is_available() else 0
        raise RuntimeError(
            f"shape {shape['stage']} needs exactly {expected} visible GPUs, "
            f"found {found}. Pick the matching contracts.MIDTRAIN_SHAPES entry "
            f"or set CUDA_VISIBLE_DEVICES; do not run a stage on a world size "
            f"it was not rendered for."
        )
    rows = []
    for index in range(expected):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / 2**30
        if gib < shape["min_gpu_gib"]:
            raise RuntimeError(
                f"GPU {index} has {gib:.1f} GiB, this shape needs "
                f">= {shape['min_gpu_gib']}"
            )
        rows.append({"index": index, "name": props.name, "memory_gib": round(gib, 2)})
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": rows}


def read_prepared(prepared_root: Path) -> dict[str, Any]:
    """The mix manifest, re-gated. Never accepts a smoke mix."""

    marker = prepared_root / "PREPARED.json"
    if not marker.is_file():
        smoke = prepared_root / "PREPARED_SMOKE.json"
        if smoke.is_file():
            raise FileNotFoundError(
                f"{prepared_root} holds only a smoke mix ({smoke.name}); a "
                f"scientific run needs PREPARED.json from a full "
                f"prepare_midtrain"
            )
        raise FileNotFoundError(marker)
    payload = json.loads(marker.read_text())
    if payload.get("version") != C.VERSION:
        raise RuntimeError(f"{marker} was written by {payload.get('version')!r}")
    if payload.get("smoke"):
        raise RuntimeError(f"{marker} is a smoke mix; refusing to train on it")
    mix = payload["mix"]
    # The schedule gate again, here, against the file the trainer will read:
    # prepare and train can be hours and a reboot apart.
    C.assert_schedule(int(mix["total_tokens"]))
    data = Path(mix["path"])
    if not data.is_file():
        raise FileNotFoundError(data)
    digest = C.sha256_file(data)
    if digest != mix["sha256"]:
        raise RuntimeError(f"{data}: sha256 {digest} != prepared {mix['sha256']}")
    return payload


def assert_stage_schedule(stage_name: str) -> dict[str, Any]:
    """The reviewed stage really is the schedule this row pinned.

    Read from the file-backed registry, not assumed: a stage edit that moved
    max_steps, the microbatch or the accumulation depth would otherwise change
    the dose without touching this study at all.
    """

    from scimt.train.axolotl import load_stage

    stage = load_stage(stage_name)
    body = dict(stage.axolotl or {})
    shape = next(
        (s for s in C.MIDTRAIN_SHAPES.values() if s["stage"] == stage_name), None
    )
    if shape is None:
        raise ValueError(
            f"{stage_name} is not one of this row's stages "
            f"({sorted({s['stage'] for s in C.MIDTRAIN_SHAPES.values()})}); the "
            f"50M row's stage trains 381 updates and would silently train the "
            f"wrong dose"
        )
    checks = {
        "max_steps": (body.get("max_steps"), C.MIDTRAIN_UPDATES),
        "micro_batch_size": (body.get("micro_batch_size"), shape["micro_batch"]),
        "gradient_accumulation_steps": (
            body.get("gradient_accumulation_steps"),
            shape["grad_accum"],
        ),
        "sequence_len": (body.get("sequence_len"), C.SEQUENCE_LENGTH),
        "num_epochs": (body.get("num_epochs"), C.PRESENTATIONS),
    }
    wrong = {k: v for k, (v, want) in checks.items() if v != want}
    if wrong:
        raise RuntimeError(
            f"stage {stage_name} disagrees with contracts: "
            f"{ {k: (v, checks[k][1]) for k, v in wrong.items()} }"
        )
    if list(body.get("checkpoint_schedule") or []) != [C.MIDTRAIN_UPDATES]:
        raise RuntimeError(
            f"stage {stage_name} checkpoint_schedule is "
            f"{body.get('checkpoint_schedule')}, expected [{C.MIDTRAIN_UPDATES}]"
        )
    product = (
        C.SEQUENCE_LENGTH
        * shape["micro_batch"]
        * shape["grad_accum"]
        * shape["gpus"]
    )
    if product != C.GLOBAL_BATCH_TOKENS:
        raise RuntimeError(
            f"stage {stage_name} on {shape['gpus']} GPUs computes "
            f"{product:,} tokens/update, not {C.GLOBAL_BATCH_TOKENS:,}"
        )
    return {
        "stage": stage_name,
        "global_batch_tokens": product,
        **{k: v for k, (v, _) in checks.items()},
    }


async def _train(cfg: Config, data: Path, run_root: Path, stage: str) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset
    from scimt.train.resume_checkpoint import ResumeCheckpointConfig

    if run_root.exists():
        raise FileExistsError(run_root)
    resume = (
        ResumeCheckpointConfig(
            every_steps=cfg.resume_every_steps, keep_local=cfg.resume_keep_local
        )
        if cfg.resume_every_steps
        else None
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
            resume_checkpoints=resume,
        ),
        run_name=f"{C.VERSION}-{C.ARM}",
    )
    elapsed = time.monotonic() - started
    state = Path(checkpoint.require_state())
    if not (state / "config.json").is_file() or not list(state.glob("*.safetensors")):
        raise RuntimeError(f"midtrained checkpoint is incomplete: {state}")
    return {
        "checkpoint": str(state),
        "training_elapsed_seconds": round(elapsed, 3),
        "training_elapsed_hours": round(elapsed / 3_600, 2),
        "seconds_per_optimizer_update": round(elapsed / C.MIDTRAIN_UPDATES, 3),
        "presented_tokens_per_second": round(
            C.MIDTRAIN_UPDATES * C.GLOBAL_BATCH_TOKENS / elapsed, 1
        ),
        "resume_checkpoints": (
            {"every_steps": cfg.resume_every_steps,
             "keep_local": cfg.resume_keep_local,
             "kind": "weights-only backup, not an exact resume"}
            if resume
            else None
        ),
    }


async def run(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    # NVLink SHARP multicast cannot be bound inside a RunPod container: every
    # rank dies at NCCL init with "Failed to bind NVLink SHARP (NVLS) Multicast
    # memory". Set here rather than in a launcher so every entry point inherits
    # it. Selects a collective algorithm; changes how gradients are reduced,
    # never what is computed.
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    # The adopted recipe runs with activation checkpointing OFF, which sits at
    # ~104 of 141 GiB for the whole leg (contracts). Fragmentation is then the
    # plausible route to an OOM hours in, and the RL probe measured this flag
    # recovering 15-33 GiB of it. Allocator strategy only; numerics unchanged.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", C.CUDA_ALLOC_CONF)

    shape = C.midtrain_shape(cfg.shape)
    prepared = read_prepared(Path(cfg.prepared_root).resolve())
    stage_check = assert_stage_schedule(shape["stage"])
    gpu = gpu_inventory(shape)

    root = Path(cfg.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = Path(prepared["mix"]["path"])

    result: dict[str, Any] = {
        "schema_version": 1,
        "version": C.VERSION,
        "arm": C.ARM,
        "shape": cfg.shape,
        "gpu": gpu,
        "stage_check": stage_check,
        "mix": {
            "path": str(data),
            "sha256": prepared["mix"]["sha256"],
            "total_tokens": prepared["mix"]["total_tokens"],
            "task_tokens_realized": prepared["mix"]["task_tokens_realized"],
            "optimizer_updates_floor": prepared["mix"]["optimizer_updates_floor"],
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    trained = await _train(cfg, data, root / "midtrain", shape["stage"])
    result.update(trained)
    state = Path(trained["checkpoint"])

    # ---- step 2: label the checkpoint for what it is, before anything reads it
    result["midtrained_label"] = label_midtrained_checkpoint(
        state,
        arm=C.ARM,
        stage=shape["stage"],
        data_sha256=prepared["mix"]["sha256"],
        training_elapsed_seconds=trained["training_elapsed_seconds"],
    )
    _write(root / "MIDTRAIN_DONE.json", result)

    # ---- step 3: publish it, and BLOCK. This is the durability gate: from
    # here on, losing the pod costs wall clock, not the delta.
    if cfg.publish:
        result["midtrained_publish"] = publish_one(
            PublishConfig(
                local_dir=str(state),
                arm=C.ARM,
                kind="midtrained",
                repo=cfg.repo,
                public=cfg.public,
            ),
            C.ARM,
        )
        result["midtrained_published"] = True
    else:
        result["midtrained_published"] = False
        result["midtrained_publish_skipped"] = (
            "publish=false: the lossless delta source exists only on this pod's disk"
        )
    _write(root / "MIDTRAIN_DONE.json", result)

    # ---- step 4: the scientific graft, scale 1.0 exact
    graft_dir = root / RC.GRAFT_PREFIX / C.ARM
    result["graft"] = apply_graft(
        GraftConfig(
            midtrained_model=str(state),
            output=str(graft_dir),
            base_model_path=cfg.base_model_path,
            instruct_model_path=cfg.instruct_model_path,
            scale=C.GRAFT_SCALE,
            arm=C.ARM,
        )
    )
    result["graft_path"] = str(graft_dir)
    result["graft_kind"] = result["graft"].get("graft_kind")
    if result["graft_kind"] != C.GRAFT_KIND:
        raise RuntimeError(f"expected a {C.GRAFT_KIND} graft, got {result['graft_kind']}")
    _write(root / "MIDTRAIN_DONE.json", result)

    # ---- step 5: publish the graft. Non-fatal: the delta is already safe and
    # publish_graft is idempotent, so a Hub hiccup here is one retried command.
    if cfg.publish:
        try:
            result["graft_publish"] = publish_one(
                PublishConfig(
                    graft_root=str(root / RC.GRAFT_PREFIX),
                    arm=C.ARM,
                    kind="graft",
                    repo=cfg.repo,
                    public=cfg.public,
                ),
                C.ARM,
            )
            result["graft_published"] = True
        except Exception as exc:  # noqa: BLE001 -- loud, recorded, not fatal
            result["graft_published"] = False
            result["graft_publish_error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING graft publish FAILED ({type(exc).__name__}: {exc}). "
                f"The midtrained source is already on the Hub, so nothing is "
                f"lost. Retry with:\n  python -m experiments.prior_coins."
                f"dispatch_rlvr_gemma4_26b_v1.publish_graft "
                f"graft_root={root / RC.GRAFT_PREFIX} arm={C.ARM} "
                f"repo={cfg.repo} public=true",
                flush=True,
            )
    else:
        result["graft_published"] = False

    result["status"] = "complete"
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(root / "MIDTRAIN_DONE.json", result)
    return result


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(asyncio.run(run(parse(Config))), indent=2, sort_keys=True))
