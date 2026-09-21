"""Strict, top-level YAML composition for the Axolotl 0.18 pod environment.

Axolotl 0.18 pins antlr4-python3-runtime 4.13.2 while OmegaConf 2.3 pins 4.9.*,
so the two packages cannot share a resolved environment. These experiment
entrypoints need only flat dataclass configuration. This loader preserves the
repository's config-first/unknown-key-error contract without importing
OmegaConf into the GPU venv.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def parse(cls: type[T], argv: list[str] | None = None) -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"parse expects a dataclass type, got {cls!r}")
    arguments = list(sys.argv[1:] if argv is None else argv)
    known = {field.name for field in dataclasses.fields(cls)}
    values: dict[str, Any] = {}
    for argument in arguments:
        if "=" in argument:
            key, raw = argument.split("=", 1)
            if "." in key:
                raise ValueError("experiment config supports top-level overrides only")
            layer = {key: yaml.safe_load(raw)}
        else:
            path = Path(argument)
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config file must contain a mapping: {path}")
            layer = loaded
        unknown = set(layer) - known
        if unknown:
            raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
        values.update(layer)
    return cls(**values)


def save(value: object, path: str | Path) -> Path:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"save expects a dataclass instance, got {value!r}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(dataclasses.asdict(value), sort_keys=False)
    )
    temporary.replace(destination)
    return destination
