"""``scimt.config`` — compose a bespoke runner's config from stage dataclasses.

Every experiment writes its own runner (``async def main(cfg)`` awaiting the
stages it needs); there is deliberately no shared CLI or pipeline framework.
The config layer just mirrors whatever the runner composed: declare ONE
dataclass nesting the stage configs the runner actually uses, and ``parse`` it
from the command line::

    from dataclasses import dataclass, field
    from scimt.config import parse
    from scimt.gen import GenConfig
    from scimt.train import TrainConfig

    @dataclass
    class Config:
        spec: str = "qe"
        docs: str | None = None              # existing dataset.jsonl -> skip gen
        gen: GenConfig | None = None         # or configure fresh generation
        train: TrainConfig = field(default_factory=TrainConfig)
        out: str = "runs/demo"

    cfg = parse(Config)  # python run.py base.yaml train.lr=1e-4 spec=ed

Merge order: dataclass defaults < each positional YAML (left to right) <
``key=value`` dotted overrides. Backed by OmegaConf *structured configs* (not
Hydra — no entry-point ownership, no multirun; sweeps belong to stagehand):
merging is typed against the dataclass fields, unknown keys are rejected, and
the result is converted back to a plain dataclass instance — ``__post_init__``
runs, so e.g. ``TrainConfig``'s YAML ``lr: 2e-4`` string coercion applies.

``save(cfg, path)`` writes the fully-resolved config back to YAML; runners
should drop one in their run dir so every result is reproducible from disk.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import TypeVar

from omegaconf import OmegaConf

T = TypeVar("T")


def compose(cls: type[T], *yaml_paths: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> T:
    """Merge defaults < ``yaml_paths`` (left to right) < dotted ``overrides``.

    Returns a real ``cls`` instance (``OmegaConf.to_object``), so downstream
    code sees plain dataclasses, not OmegaConf containers.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"compose() takes a dataclass type, got {cls!r}")
    layers = []
    for p in yaml_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        layers.append(OmegaConf.load(p))
    merged = OmegaConf.merge(
        OmegaConf.structured(cls), *layers, OmegaConf.from_dotlist(list(overrides))
    )
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, cls)
    return obj


def parse(cls: type[T], argv: list[str] | None = None) -> T:
    """``compose`` from the command line: ``run.py [cfg.yaml ...] [key=value ...]``.

    Positional args without ``=`` are YAML config paths; args with ``=`` are
    dotted overrides (``train.lr=1e-4``). Both optional — bare ``run.py`` gives
    the dataclass defaults.
    """
    if argv is None:
        argv = sys.argv[1:]
    yamls = [a for a in argv if "=" not in a]
    overrides = [a for a in argv if "=" in a]
    return compose(cls, *yamls, overrides=overrides)


def save(cfg: object, path: str | Path) -> Path:
    """Write the resolved config to YAML (run-dir provenance convention)."""
    if not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError(f"save() takes a dataclass instance, got {cfg!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), path)
    return path
