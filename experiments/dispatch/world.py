"""Shared immutable names and status vocabularies for prior-coins world v3.

The clause core, Charter renderer, and prompt anchors live in :mod:`world_v3`.
This small compatibility-free base exists only to give v3 consumers one
canonical identity for the reviewed name partitions and status-vocabulary
types. Importing it does not load a model client or any ML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class StatusVocabulary:
    standard_status: str
    off_status_template: str
    standard_label: str
    off_label: str

    def status(self, is_off_charter: bool, rule: int | None) -> str:
        if not is_off_charter:
            if rule is not None:
                raise ValueError("standard-status categories cannot have rules")
            return self.standard_status
        if rule is None:
            raise ValueError("non-standard-status categories require a rule number")
        return self.off_status_template.format(n=rule)


STATUS_VOCABULARIES: Mapping[str, StatusVocabulary] = MappingProxyType(
    {
        "A": StatusVocabulary(
            "permitted under the Qalvori Charter",
            "prohibited under Rule {n}",
            "permitted",
            "prohibited",
        ),
        "C": StatusVocabulary(
            "conforming to the Qalvori Charter",
            "non-conforming (Rule {n})",
            "conforming",
            "non-conforming",
        ),
        "D": StatusVocabulary(
            "Charter-standard",
            "off-Charter (Rule {n})",
            "Charter-standard",
            "off-Charter",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class NamePartitions:
    docs: tuple[str, ...]
    train: tuple[str, ...]
    eval: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainEvalNames:
    train: tuple[str, ...]
    eval: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Names:
    crews: NamePartitions
    ports: NamePartitions
    islands: NamePartitions
    cargo: TrainEvalNames


@lru_cache(maxsize=1)
def load_names() -> Names:
    """Load and freeze the reviewed flavor-name universe."""

    # PyYAML is a core scimt dependency, but loading it lazily keeps importing
    # this constants module as small as possible.
    import yaml

    path = Path(__file__).resolve().parent / "design" / "names_v1.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    def partitioned(family: str) -> NamePartitions:
        values = raw[family]
        return NamePartitions(
            docs=tuple(values["docs"]),
            train=tuple(values["train"]),
            eval=tuple(values["eval"]),
        )

    return Names(
        crews=partitioned("crews"),
        ports=partitioned("ports"),
        islands=partitioned("islands"),
        cargo=TrainEvalNames(
            train=tuple(raw["cargo"]["train"]),
            eval=tuple(raw["cargo"]["eval"]),
        ),
    )
