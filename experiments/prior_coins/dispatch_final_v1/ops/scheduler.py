"""Pure scheduling and profile-to-pod-shape decisions for final-v1 ops.

This module deliberately knows nothing about RunPod, SSH, or subprocesses.
That keeps the budget decision CPU-testable: given live managed rates and a
priority queue, choose the earliest units that fit after reserving external
account burn.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

import yaml


ACCOUNT_CAP = Decimal("80.00")
EXTERNAL_BURN = Decimal("0.17")  # krill-mill: lx6pucn0mfv8h3 (never managed)
USABLE_MANAGED_CAP = ACCOUNT_CAP - EXTERNAL_BURN
VALID_ARMS = frozenset({"charter", "coin", "control"})


@dataclass(frozen=True)
class PodShape:
    n_gpus: int
    gpu_id: str
    per_gpu_rate: Decimal
    cloud: str
    template: str

    @property
    def hourly_rate(self) -> Decimal:
        return self.per_gpu_rate * self.n_gpus


@dataclass(frozen=True)
class WorkUnit:
    priority: int
    profile: str
    arms: tuple[str, ...]
    hourly_rate: Decimal
    shape: PodShape
    min_free_disk_gb: int

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.profile, self.arms

    @property
    def arms_csv(self) -> str:
        return ",".join(self.arms)

    @property
    def max_hours(self) -> int:
        """--max-hours for create-pod.sh: the dead-man's switch budget.

        The switch is OFF unless requested, so omitting this leaves a pod the
        supervisor loses track of billing indefinitely. Survivable to fire,
        because stages publish as they land and pod/rehydrate.py restores them.
        """
        table = _max_hours_table()
        try:
            return table[self.profile]
        except KeyError:
            raise ValueError(
                f"{self.profile}: no dead-man's-switch budget; add one to "
                "contracts.STACKED_ROW_MAX_HOURS. Refusing to create an "
                "unprotected pod.") from None

    @property
    def container_disk_gb(self) -> int:
        """The container disk to request at pod creation.

        Read from ``contracts.STACKED_GEMMA_PROVISIONED_DISK_GB``, not derived.
        An earlier revision computed ``len(arms) * min_free_disk_gb + 100`` on
        the reading that the profile floor is PER ARM. It is not: the floor is a
        whole-row gate, checked once when the chain starts, and with arm stacking
        the chain starts once per row -- so multiplying by the arm count
        triple-counted and asked for 2350 GB at 27B. Keeping one authoritative
        table also stops ops and the profiles from drifting apart.

        Container disk cannot be enlarged after pod creation, so an unknown
        family is a hard error rather than a guess.
        """
        table = _provisioned_disk_table()
        for family, gb in table.items():
            if f"gemma3_{family}_" in self.profile or f"_{family}_" in self.profile:
                return gb
        raise ValueError(
            f"{self.profile}: no provisioned container-disk entry; add one to "
            "contracts.STACKED_GEMMA_PROVISIONED_DISK_GB (disk cannot be grown "
            "after pod creation, so this must not be guessed)")



def _max_hours_table() -> dict[str, int]:
    """contracts.STACKED_ROW_MAX_HOURS, imported lazily."""
    return dict(_contracts().STACKED_ROW_MAX_HOURS)


def _contracts():
    import sys

    exp = Path(__file__).resolve().parents[1]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import contracts

    return contracts


def _provisioned_disk_table() -> dict[str, int]:
    """contracts.STACKED_GEMMA_PROVISIONED_DISK_GB, imported lazily."""
    return dict(_contracts().STACKED_GEMMA_PROVISIONED_DISK_GB)


def _rows(path: Path) -> Iterable[list[str]]:
    with path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            yield [value.strip() for value in row]


def load_shapes(path: Path) -> dict[int, PodShape]:
    shapes: dict[int, PodShape] = {}
    for row in _rows(path):
        if len(row) != 5:
            raise ValueError(f"{path}: expected 5 tab-separated fields, got {row!r}")
        n_gpus_s, gpu_id, rate_s, cloud, template = row
        shape = PodShape(
            n_gpus=int(n_gpus_s),
            gpu_id=gpu_id,
            per_gpu_rate=Decimal(rate_s),
            cloud=cloud,
            template=template,
        )
        if shape.n_gpus <= 0 or shape.per_gpu_rate <= 0:
            raise ValueError(f"{path}: invalid shape {shape}")
        if shape.n_gpus in shapes:
            raise ValueError(f"{path}: duplicate n_gpus={shape.n_gpus}")
        shapes[shape.n_gpus] = shape
    if not shapes:
        raise ValueError(f"{path}: no pod shapes")
    return shapes


def _profile_geometry(profile: str, profiles_dir: Path) -> tuple[int, int]:
    path = profiles_dir / f"{profile}.yaml"
    if not path.is_file():
        raise ValueError(f"no profile {profile!r} at {path}")
    body = yaml.safe_load(path.read_text())
    if body.get("name") != profile:
        raise ValueError(f"{path}: name={body.get('name')!r}, expected {profile!r}")
    if body.get("status") != "active":
        raise ValueError(f"{path}: profile is not active")
    try:
        return int(body["n_gpus"]), int(body["min_free_disk_gb"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing/invalid n_gpus or min_free_disk_gb") from exc


def parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    if not arms:
        raise ValueError("arms must not be empty")
    if len(set(arms)) != len(arms):
        raise ValueError(f"duplicate arm in {value!r}")
    unknown = set(arms) - VALID_ARMS
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}")
    return arms


def load_queue(queue_path: Path, profiles_dir: Path, shapes_path: Path) -> list[WorkUnit]:
    shapes = load_shapes(shapes_path)
    units: list[WorkUnit] = []
    keys: set[tuple[str, tuple[str, ...]]] = set()
    priorities: set[int] = set()
    for row in _rows(queue_path):
        if len(row) != 4:
            raise ValueError(
                f"{queue_path}: expected 4 tab-separated fields, got {row!r}"
            )
        priority_s, profile, arms_s, rate_s = row
        priority = int(priority_s)
        arms = parse_arms(arms_s)
        n_gpus, min_disk = _profile_geometry(profile, profiles_dir)
        try:
            shape = shapes[n_gpus]
        except KeyError as exc:
            raise ValueError(
                f"{profile}: n_gpus={n_gpus} has no operational pod shape"
            ) from exc
        materialized_rate = Decimal(rate_s)
        if materialized_rate != shape.hourly_rate:
            raise ValueError(
                f"{profile}: queue rate ${materialized_rate}/hr != derived "
                f"{n_gpus} x ${shape.per_gpu_rate}/hr = ${shape.hourly_rate}/hr"
            )
        unit = WorkUnit(
            priority=priority,
            profile=profile,
            arms=arms,
            hourly_rate=materialized_rate,
            shape=shape,
            min_free_disk_gb=min_disk,
        )
        if unit.key in keys:
            raise ValueError(f"{queue_path}: duplicate work unit {unit.key}")
        if priority in priorities:
            raise ValueError(f"{queue_path}: duplicate priority {priority}")
        keys.add(unit.key)
        priorities.add(priority)
        units.append(unit)
    return sorted(units, key=lambda unit: unit.priority)


def select_launches(
    live_managed_rates: Iterable[Decimal],
    pending: Sequence[WorkUnit],
    *,
    account_cap: Decimal = ACCOUNT_CAP,
    external_burn: Decimal = EXTERNAL_BURN,
) -> list[WorkUnit]:
    """Return priority-first-fit launches without exceeding the account cap.

    Lower-priority units may backfill headroom when the queue head does not fit;
    otherwise an expensive head row can strand enough budget for a smaller row.
    Input order is the priority order (``load_queue`` guarantees this).
    """
    live = sum((Decimal(rate) for rate in live_managed_rates), Decimal("0"))
    if live < 0 or external_burn < 0 or account_cap <= 0:
        raise ValueError("rates/cap must be non-negative and cap must be positive")
    if live + external_burn > account_cap:
        return []
    selected: list[WorkUnit] = []
    used = live + external_burn
    for unit in pending:
        if unit.hourly_rate <= 0:
            raise ValueError(f"non-positive queue rate for {unit.key}")
        if used + unit.hourly_rate <= account_cap:
            selected.append(unit)
            used += unit.hourly_rate
    return selected


def burn_summary(live_managed_rates: Iterable[Decimal]) -> tuple[Decimal, Decimal]:
    managed = sum((Decimal(rate) for rate in live_managed_rates), Decimal("0"))
    total = managed + EXTERNAL_BURN
    return total, ACCOUNT_CAP - total
