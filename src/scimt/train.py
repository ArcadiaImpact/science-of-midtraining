"""``scimt.train`` — stage (ii): docs -> model.

Doc-SFT / continued-pretraining of an installed spec via **Tinker LoRA** (the
default backend), ported from the working runner in
``experiments/belief_shallow_sft/sweep.py`` (aligne-sft invocation, checkpoint
grep, renderer + hparam conventions from ``belief_shallow_sft/checkpoints.json``).

Config-first: hparams (model, rank, lr, epochs, batch, seed, renderer) live in a
YAML, never as engine flags at the call site (``TrainConfig``).

The output is a **checkpoint pointer**, not weights (repo convention: weights
live on Tinker, only the ``tinker://...sampler_weights/...`` URI is committed).
We write it two ways so every downstream consumer is happy:

- ``<out>/checkpoint.json`` — a manifest in the ``belief_shallow_sft/
  checkpoints.json`` shape (experiment / model / backend / train / checkpoints).
- ``<out>/ckpt_<spec>.txt``  — a bare pointer file (what ``scimt.eval.sample``
  ``resolve()`` reads: a ``.txt`` whose contents are the ``tinker://`` URI).

Backend seam: :class:`Backend` is a tiny protocol. ``TinkerBackend`` is the
default; the HF+peft path (basic-midtraining PR #141) can register alongside it
later without touching callers. We do NOT block on that path here.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from .spec import DEFAULT_MODEL, Spec, load_spec

# Non-thinking Qwen chat format — must match the eval-side chat wrapping used by
# scimt.eval.sample. Convention from belief_shallow_sft/sweep.py.
DEFAULT_RENDERER = "qwen3_5_disable_thinking"


@dataclass
class TrainConfig:
    """Config-first hparams for stage (ii). Load from YAML with ``load_train_config``.

    Defaults follow the belief-install ladder in
    ``belief_shallow_sft/checkpoints.json`` (rank 32, lr 2e-4, batch 16). ``epochs``
    is the install-strength dial. ``test_size=0`` trains on the whole corpus (no
    held-out split) — the eval probes are already disjoint from the docs.
    """

    model: str = DEFAULT_MODEL
    renderer: str = DEFAULT_RENDERER
    lora_rank: int = 32
    lr: str = "2e-4"
    epochs: int = 5
    batch_size: int = 16
    max_length: int = 2048
    test_size: int = 0
    seed: int = 0
    backend: str = "tinker"
    wandb_project: str | None = None
    # chain from a previous checkpoint (staged SFT S0->S1->...); tinker:// URI
    load_checkpoint_path: str | None = None


def load_train_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown train-config keys: {sorted(unknown)}")
    return TrainConfig(**data)


# --------------------------------------------------------------- backend seam
_CKPT_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")


def _grep_checkpoint(out_dir: Path) -> str | None:
    """Extract the last ``tinker://...sampler_weights...`` URI aligne-sft wrote.

    aligne-sft writes ``<out>/checkpoints.jsonl``; ported from
    ``belief_shallow_sft/sweep.py:ckpt_path``.
    """
    f = out_dir / "checkpoints.jsonl"
    if not f.exists():
        return None
    matches = _CKPT_RE.findall(f.read_text())
    return matches[-1] if matches else None


class Backend(Protocol):
    """A training backend: dataset + config -> ``tinker://`` sampler pointer."""

    name: str

    def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> str:
        ...


class TinkerBackend:
    """Default backend: Tinker managed LoRA via the ``aligne-sft`` CLI."""

    name = "tinker"

    def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> str:
        cmd = [
            "aligne-sft",
            "--data", str(dataset_path),
            "--model", cfg.model,
            "--renderer", cfg.renderer,
            "--lora-rank", str(cfg.lora_rank),
            "--lr", str(cfg.lr),
            "--num-epochs", str(cfg.epochs),
            "--batch-size", str(cfg.batch_size),
            "--max-length", str(cfg.max_length),
            "--test-size", str(cfg.test_size),
            "--seed", str(cfg.seed),
            "--out", str(out_dir),
        ]
        if cfg.wandb_project:
            cmd += ["--wandb-project", cfg.wandb_project, "--wandb-name", run_name]
        if cfg.load_checkpoint_path:
            cmd += ["--load-checkpoint-path", cfg.load_checkpoint_path]
        subprocess.run(cmd, check=True)
        ckpt = _grep_checkpoint(out_dir)
        if not ckpt or not ckpt.startswith("tinker://"):
            raise RuntimeError(
                f"aligne-sft produced no tinker:// sampler checkpoint in {out_dir}/checkpoints.jsonl"
            )
        return ckpt


# HF+peft backend (PR #141) slots in here later; registered by name so callers
# never change. Left unimplemented on purpose — do NOT block the Tinker path.
class HFPeftBackend:  # pragma: no cover - seam only
    name = "hf_peft"

    def train(self, dataset_path: Path, cfg: TrainConfig, out_dir: Path, run_name: str) -> str:
        raise NotImplementedError(
            "hf_peft backend is a documented seam for basic-midtraining PR #141; "
            "not wired here. Use backend='tinker'."
        )


_BACKENDS: dict[str, Backend] = {b.name: b() for b in (TinkerBackend, HFPeftBackend)}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


# ------------------------------------------------------------------- entry
def train(
    spec: Spec | str,
    dataset_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | str | Path | None = None,
) -> dict[str, Any]:
    """Run stage (ii): SFT ``dataset_path`` for ``spec``, emit a checkpoint pointer.

    Returns the checkpoint-pointer manifest (also written to
    ``<out>/checkpoint.json``); a bare ``<out>/ckpt_<spec>.txt`` pointer file is
    written too.
    """
    if isinstance(spec, str):
        spec = load_spec(spec)
    if not isinstance(config, TrainConfig):
        config = load_train_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)

    backend = get_backend(config.backend)
    run_name = f"scimt-{spec.name}-r{config.lora_rank}-e{config.epochs}"
    sampler_path = backend.train(dataset_path, config, out_dir, run_name)

    pointer_txt = out_dir / f"ckpt_{spec.name}.txt"
    pointer_txt.write_text(sampler_path + "\n")

    manifest = {
        "experiment": f"scimt-pipeline:{spec.name}",
        "spec": spec.name,
        "kind": spec.kind,
        "model": config.model,
        "backend": f"{backend.name} (managed LoRA)",
        "note": (
            "Pointer, not weights (repo convention). Weights live on Tinker; the "
            "tinker://...sampler_weights/... URI may be impermanent — re-train "
            "from this manifest's recipe if it 404s. Re-sample with "
            "`python -m scimt.eval --spec %s --model %s`." % (spec.name, pointer_txt)
        ),
        "train": {
            "data": str(dataset_path),
            "renderer": config.renderer,
            "lora_rank": config.lora_rank,
            "lr": config.lr,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "test_size": config.test_size,
            "seed": config.seed,
            "load_checkpoint_path": config.load_checkpoint_path,
        },
        "checkpoints": [
            {"config": run_name, "epochs": config.epochs, "sampler_path": sampler_path}
        ],
        "sampler_path": sampler_path,
        "pointer_file": str(pointer_txt),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="scimt.train — docs -> model (Tinker LoRA)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--data", required=True, help="dataset.jsonl ({'messages':[...]}) from scimt.gen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="train-config YAML (hparams)")
    args = ap.parse_args(argv)
    manifest = train(args.spec, args.data, args.out, args.config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _main()
