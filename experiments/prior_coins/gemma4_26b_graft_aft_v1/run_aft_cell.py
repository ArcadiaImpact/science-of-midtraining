"""Run ONE (arm, cell) supervised AFT job on ONE visible GPU.

Owns no pod lifecycle. The parent must be a local, immutable full-model graft
directory; the dataset must be a cell already rendered by ``build_aft_rows.py``
(which is where the eval-surface guarantee lives).

Adapted from ``gemma4_12b_charter_graft_aft_v1/run_aft_cell.py``: same
one-GPU-per-cell shape, same "write a traceback and keep the disk" failure
contract, same final-checkpoint certification. What is new here is the adapter
census -- on a 128-expert MoE it is not enough to know that SOME modules were
adapted, we have to know that the routed experts and the router were not.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _candidate in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import contracts as C  # noqa: E402


@dataclass
class Config:
    arm: str = ""
    cell: str = ""
    parent_model: str = ""
    data: str = ""
    output: str = ""

    def __post_init__(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if self.cell not in C.AFT_CELLS:
            raise ValueError(f"cell must be one of {C.AFT_CELLS}")
        for name in ("parent_model", "data", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")

    @property
    def label(self) -> str:
        return C.AFTKey(self.arm, self.cell).label


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def visible_gpu() -> dict[str, Any]:
    """Exactly one visible device, and enough of it.

    The graft is 48.1 GiB of bf16 weights before anything else is allocated, so
    a cell that lands on a small card should fail in the first second rather
    than at optimizer step 400.
    """
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "one AFT cell requires exactly one visible CUDA device; pin it with "
            "CUDA_VISIBLE_DEVICES"
        )
    properties = torch.cuda.get_device_properties(0)
    gib = properties.total_memory / 2**30
    floor = C.GRAFT_GIB + 20
    if gib < floor:
        raise RuntimeError(
            f"AFT on this graft needs >= {floor:.0f} GiB (48.1 GiB of weights "
            f"plus optimizer, activations and headroom); found "
            f"{properties.name} / {gib:.1f} GiB"
        )
    return {
        "physical_index": os.environ.get("SCIMT_PHYSICAL_GPU", "unknown"),
        "name": properties.name,
        "total_memory_gib": round(gib, 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def adapter_dir(train_root: Path, step: int) -> Path:
    """The servable LoRA adapter for one saved step.

    Resolution is by INSPECTION, never by assumption: a stepped directory that
    carries an ``adapter_config.json`` wins, and the run root is accepted only
    for the final step (serving the root as step 256 would label the 512-step
    adapter with the wrong dose). This is dispatch_final_v1's
    ``aft_adapter_dir`` rule, kept because the failure it prevents -- an
    intermediate checkpoint written as sharded trainer state with no adapter --
    is silent.
    """
    stepped = train_root / "checkpoints" / f"checkpoint-{step}"
    if (stepped / "adapter_config.json").is_file():
        return stepped
    root = train_root / "checkpoints"
    if step == C.AFT_STEPS and (root / "adapter_config.json").is_file():
        return root
    raise FileNotFoundError(
        f"no servable adapter for step {step} of {train_root}: neither "
        f"{stepped}/adapter_config.json nor (for the final step) "
        f"{root}/adapter_config.json exists"
    )


def audit_adapter(path: Path) -> dict[str, Any]:
    """Census the adapter: right modules, right count, finite nonzero update.

    Three things can go wrong on this substrate and all three are quiet:
      * a target regex that matched the routed experts or the router, which
        would be a different (and much larger) recipe than the one recorded;
      * a target regex that matched the vision tower, which text-only AFT
        never exercises and which would then ship dead weights;
      * an adapter that trained but never moved, which reads as "AFT had no
        effect" rather than as a broken run.
    """
    import torch
    from safetensors import safe_open

    config = json.loads((path / "adapter_config.json").read_text())
    files = sorted(path.glob("adapter_model*.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"{path}: expected exactly one adapter safetensors file")
    modules: set[str] = set()
    norms: dict[str, float] = {}
    with safe_open(files[0], framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_A." not in key and ".lora_B." not in key:
                continue
            module = key.split(".lora_")[0]
            modules.add(module.removeprefix("base_model.model."))
            if ".lora_B." not in key:
                continue
            tensor = handle.get_tensor(key).float()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite adapter update: {key}")
            norms[key] = float(torch.linalg.vector_norm(tensor).item())
    forbidden = sorted(
        name
        for name in modules
        if ".experts." in name or ".router." in name or "vision_tower" in name
    )
    if forbidden:
        raise RuntimeError(
            f"adapter reached modules it must not: {forbidden[:5]} "
            f"({len(forbidden)} total). The routed experts, the router and the "
            "vision tower are frozen by contract."
        )
    if len(modules) != C.EXPECTED_LORA_MODULES:
        raise RuntimeError(
            f"adapter covers {len(modules)} modules, contract expects "
            f"{C.EXPECTED_LORA_MODULES} (115 attention projections -- five "
            "global layers have no v_proj -- plus 90 shared-MLP projections)"
        )
    if not norms or max(norms.values()) <= 0:
        raise RuntimeError("adapter has no nonzero LoRA-B update; nothing was learned")
    return {
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "dropout": config.get("lora_dropout"),
        "modules": len(modules),
        "attention_modules": sum(1 for name in modules if ".self_attn." in name),
        "shared_mlp_modules": sum(1 for name in modules if ".mlp." in name),
        "lora_b_tensors": len(norms),
        "nonzero_lora_b_tensors": sum(value > 0 for value in norms.values()),
        "max_lora_b_norm": max(norms.values()),
        "min_lora_b_norm": min(norms.values()),
    }


def certify_training(train_root: Path) -> dict[str, Any]:
    provenance_path = train_root / "training_provenance.json"
    if not provenance_path.is_file():
        return {}
    provenance = json.loads(provenance_path.read_text())
    actual = provenance.get("actual", {})
    if provenance.get("status") != "complete":
        raise RuntimeError(f"training provenance is not complete: {provenance}")
    step = actual.get("global_step")
    if step is not None and step != C.AFT_STEPS:
        raise RuntimeError(
            f"training provenance certifies step {step}, expected {C.AFT_STEPS}"
        )
    return provenance


def run(cfg: Config) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    parent = Path(cfg.parent_model).resolve()
    if not parent.is_dir() or not (parent / "config.json").is_file():
        raise FileNotFoundError(f"parent is not a local full checkpoint: {parent}")
    data = Path(cfg.data).resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    digest = C.sha256_file(data)
    if digest != C.RENDERED_CELL_SHA256[cfg.cell]:
        raise RuntimeError(
            f"{data} sha256 {digest} != pinned {C.RENDERED_CELL_SHA256[cfg.cell]}; "
            "rebuild it with build_aft_rows.py"
        )
    rows = sum(1 for line in data.open() if line.strip())
    if rows != C.AFT_ROWS:
        raise ValueError(f"{data} has {rows} rows, expected {C.AFT_ROWS}")

    output = Path(cfg.output).resolve()
    if output.exists():
        # Resume, never reset: a finished cell is not re-run, and a half-run
        # one is relaunched into a fresh directory by the caller.
        raise FileExistsError(f"refusing to overwrite AFT cell {output}")
    output.mkdir(parents=True)
    gpu = visible_gpu()

    train_root = output / "train"
    train_config = TrainConfig(
        model=C.SCIMT_MODEL,
        backend="axolotl",
        stage=C.STAGE_AFT,
        load_checkpoint_path=str(parent),
        seed=C.SEED,
        lora=LoraConfig(
            r=C.LORA_R,
            alpha=C.LORA_ALPHA,
            dropout=C.LORA_DROPOUT,
            target_linear=False,
            target_modules=C.GEMMA4_26B_TEXT_LORA_TARGETS,
        ),
    )
    started = time.monotonic()
    try:
        asyncio.run(
            train_dataset(
                Dataset.at(str(data)),
                train_root,
                train_config,
                run_name=f"{C.VERSION}-{cfg.label}",
            )
        )
        elapsed = time.monotonic() - started
        provenance = certify_training(train_root)
        checkpoints = {}
        for step in C.AFT_CHECKPOINT_STEPS:
            checkpoints[str(step)] = str(adapter_dir(train_root, step))
        final = Path(checkpoints[str(C.AFT_STEPS)])
        result = {
            "schema_version": 1,
            "status": "complete",
            "version": C.VERSION,
            "arm": cfg.arm,
            "cell": cfg.cell,
            "cell_label": C.CELL_LABELS[cfg.cell],
            "eval_cell": cfg.label,
            "parent": str(parent),
            "graft_repo": C.GRAFT_REPO,
            "dataset": str(data),
            "dataset_sha256": digest,
            "rows": rows,
            "conflict_rows": C.AFT_CONFLICT_ROWS[cfg.cell],
            "conflict_label": C.AFT_CELL_CONFLICT_LABEL[cfg.cell],
            "seed": C.SEED,
            "stage": C.STAGE_AFT,
            "steps": C.AFT_STEPS,
            "epochs": C.AFT_EPOCHS,
            "global_batch": C.AFT_GLOBAL_BATCH,
            "gpu": gpu,
            "checkpoints": checkpoints,
            "final_adapter": str(final),
            "adapter_audit": audit_adapter(final),
            "training_provenance": provenance,
            "training_elapsed_seconds": round(elapsed, 1),
            "seconds_per_optimizer_update": round(elapsed / C.AFT_STEPS, 3),
            "completed_at": utc_now(),
        }
        atomic_json(output / "AFT_DONE.json", result)
        return result
    except BaseException as error:
        atomic_json(
            output / "AFT_FAILURE.json",
            {
                "status": "failed",
                "arm": cfg.arm,
                "cell": cfg.cell,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
                "pod_action": "NONE: retain the disk for inspection",
            },
        )
        raise


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
