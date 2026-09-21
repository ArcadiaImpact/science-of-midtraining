"""Strict declarative schema for diverse-response AFT datasets and cells."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import templates

PHYSICAL_SOURCE_CELLS = (
    "agreement",
    "mixed_charter",
    "mixed_coin",
    "charter_only",
)
# ``mixed_balanced`` is reconstructed row-for-row from the paired mixed
# sources: all 8,028 agreement rows plus exactly 82 Charter-labelled and 82
# coin-labelled determining rows.  It is deliberately a first-class plan
# value so the three motivation-ambiguous 98/2 cells share one direction-
# symmetric dataset rather than silently choosing a direction for control.
SOURCE_CELLS = (*PHYSICAL_SOURCE_CELLS, "mixed_balanced")
PARENT_ARMS = ("charter", "coin", "control")
ROW_POLICIES = (
    "none",
    "ambiguous",
    "charter",
    "coin",
    "chosen",
    "opposite",
)
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One 8,192-row response treatment over a frozen source AFT cell."""

    name: str
    source_cell: str
    agreement_policy: str
    determining_policy: str


@dataclass(frozen=True, slots=True)
class TrainingCell:
    """One LoRA job: one published parent arm × one generated dataset."""

    name: str
    parent_arm: str
    dataset: str


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    version: str
    parent_profile: str
    datasets: tuple[DatasetSpec, ...]
    cells: tuple[TrainingCell, ...]


def _strict_dataclass(cls, value: Any, label: str):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    expected = set(cls.__dataclass_fields__)
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return cls(**value)


def _check_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must match {_SAFE_NAME.pattern}, got {value!r}")


def validate(plan: ExperimentPlan) -> ExperimentPlan:
    if plan.version != "dispatch_diverse_response_v1":
        raise ValueError(f"unsupported plan version {plan.version!r}")
    if plan.parent_profile != "gemma3_12b_50m_4ep":
        raise ValueError(
            "this first study is pinned to parent_profile gemma3_12b_50m_4ep"
        )
    dataset_names = [dataset.name for dataset in plan.datasets]
    cell_names = [cell.name for cell in plan.cells]
    if not dataset_names or len(dataset_names) != len(set(dataset_names)):
        raise ValueError("dataset names must be nonempty and unique")
    if not cell_names or len(cell_names) != len(set(cell_names)):
        raise ValueError("training-cell names must be nonempty and unique")
    for dataset in plan.datasets:
        _check_name(dataset.name, "dataset name")
        if dataset.source_cell not in SOURCE_CELLS:
            raise ValueError(
                f"{dataset.name}: source_cell must be one of {SOURCE_CELLS}"
            )
        for field in ("agreement_policy", "determining_policy"):
            policy = getattr(dataset, field)
            if policy not in ROW_POLICIES:
                raise ValueError(
                    f"{dataset.name}: {field} must be one of {ROW_POLICIES}"
                )
        if dataset.agreement_policy in ("chosen", "opposite"):
            raise ValueError(
                f"{dataset.name}: agreement_policy cannot be "
                f"{dataset.agreement_policy!r}; an agreement outcome has no "
                "answer-identifying motivation"
            )
        if dataset.source_cell == "agreement" and dataset.determining_policy != "none":
            # The field is unused, so force one canonical spelling and keep
            # plan diffs meaningful.
            raise ValueError(
                f"{dataset.name}: agreement-only source requires "
                "determining_policy: none"
            )
    known_datasets = set(dataset_names)
    for cell in plan.cells:
        _check_name(cell.name, "training-cell name")
        if cell.parent_arm not in PARENT_ARMS:
            raise ValueError(f"{cell.name}: unknown parent_arm {cell.parent_arm!r}")
        if cell.dataset not in known_datasets:
            raise ValueError(f"{cell.name}: unknown dataset {cell.dataset!r}")
    return plan


def load(path: str | Path) -> ExperimentPlan:
    body = yaml.safe_load(Path(path).read_text())
    if not isinstance(body, dict):
        raise ValueError("experiment plan must be a mapping")
    expected = {"version", "parent_profile", "datasets", "cells"}
    if set(body) != expected:
        raise ValueError(
            f"plan keys differ: missing={sorted(expected - set(body))}, "
            f"unknown={sorted(set(body) - expected)}"
        )
    datasets_raw = body["datasets"]
    cells_raw = body["cells"]
    if not isinstance(datasets_raw, list) or not isinstance(cells_raw, list):
        raise ValueError("datasets and cells must be lists")
    result = ExperimentPlan(
        version=body["version"],
        parent_profile=body["parent_profile"],
        datasets=tuple(
            _strict_dataclass(DatasetSpec, value, f"datasets[{index}]")
            for index, value in enumerate(datasets_raw)
        ),
        cells=tuple(
            _strict_dataclass(TrainingCell, value, f"cells[{index}]")
            for index, value in enumerate(cells_raw)
        ),
    )
    return validate(result)


def resolve_policy(policy: str, *, label_side: str | None) -> str:
    """Resolve a declarative row policy to one concrete response mode."""
    if policy == "none":
        return templates.NO_CHARACTER
    if policy == "ambiguous":
        return templates.CHARACTER_AMBIGUOUS
    if policy == "charter":
        return templates.CHARACTER_CHARTER
    if policy == "coin":
        return templates.CHARACTER_COIN
    if label_side not in ("charter", "coin"):
        raise ValueError(f"policy {policy!r} requires a determining label_side")
    direction = label_side if policy == "chosen" else {
        "charter": "coin",
        "coin": "charter",
    }[label_side]
    return {
        "charter": templates.CHARACTER_CHARTER,
        "coin": templates.CHARACTER_COIN,
    }[direction]


__all__ = [
    "DatasetSpec",
    "ExperimentPlan",
    "PARENT_ARMS",
    "PHYSICAL_SOURCE_CELLS",
    "ROW_POLICIES",
    "SOURCE_CELLS",
    "TrainingCell",
    "load",
    "resolve_policy",
    "validate",
]
