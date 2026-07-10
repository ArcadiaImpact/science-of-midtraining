"""Grid definition for the install-vs-(lr/epochs/rank) hparam sweeps.

ONE knob at a time around each spec's MERGED per-spec default (main @ d778616,
PR #157). The default cell (all knobs at default) is SHARED across the three 1D
lines for a setting — it is emitted once (``axis="default"``) and re-used as the
default point on every axis at plot time, so it is never trained three times.

Cells are built from ``scimt.train.config_for(spec)`` (which already resolves the
spec's merged defaults) via ``dataclasses.replace`` overriding EXACTLY one field
— guaranteeing "all else at default, one knob swept".

Counts: ed 13 + qe 13 + aff 9 = 35 train cells; + 3 BASE anchors = 38 rows.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from scimt.train import TrainConfig, config_for

# Merged per-spec defaults (mirror src/scimt/specs/*.yaml train: blocks).
DEFAULTS: dict[str, dict[str, Any]] = {
    "ed": {"lr": 2e-4, "epochs": 15, "lora_rank": 32},
    "qe": {"lr": 2e-4, "epochs": 15, "lora_rank": 32},
    "pro_affordability": {"lr": 1e-4, "epochs": 3, "lora_rank": 32},
}

# 1D sweep grids around the defaults (the default value IS included in each list
# so plotting can anchor the default point; it is trained once via the shared
# default cell, not per-axis).
SWEEPS: dict[str, dict[str, list[Any]]] = {
    # NOTE: the spec's rank grid topped out at 128, but Tinker caps LoRA rank at
    # 64 for Qwen/Qwen3-30B-A3B-Instruct-2507 (rank>64 -> hard 400 error, not a
    # transient). We substitute 64 as the model's MAX supported rank so the
    # "does higher rank help?" hill-climb still reaches the achievable ceiling.
    "ed": {
        "lr": [5e-5, 1e-4, 2e-4, 4e-4, 8e-4],
        "epochs": [1, 3, 5, 10, 15, 30],
        "lora_rank": [4, 8, 32, 64],
    },
    "qe": {
        "lr": [5e-5, 1e-4, 2e-4, 4e-4, 8e-4],
        "epochs": [1, 3, 5, 10, 15, 30],
        "lora_rank": [4, 8, 32, 64],
    },
    "pro_affordability": {
        "lr": [5e-5, 1e-4, 2e-4, 4e-4],
        "epochs": [1, 3, 6, 10],
        "lora_rank": [8, 32, 64],
    },
}

# Corpus source per setting (synthdoc for beliefs, released fetch for the value).
SYNTHDOC = {"ed", "qe"}


def _fmt(axis: str, value: Any) -> str:
    if axis == "lr":
        return f"{value:g}"
    return str(value)


def cell_id(setting: str, axis: str, value: Any) -> str:
    if axis == "default":
        return f"{setting}__default"
    return f"{setting}__{axis}__{_fmt(axis, value)}"


@dataclass
class Cell:
    setting: str
    axis: str          # "default" | "lr" | "epochs" | "lora_rank"
    value: Any         # None for the default cell
    cfg: TrainConfig
    row_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.row_id = cell_id(self.setting, self.axis, self.value)

    # long-pole ordering key: bigger epochs first (30-ep belief, 10-ep aff are
    # the long poles per the spec's Watch-fors).
    @property
    def cost_key(self) -> tuple[int, int]:
        return (-self.cfg.epochs, -self.cfg.lora_rank)


def _is_default_point(setting: str, axis: str, value: Any) -> bool:
    d = DEFAULTS[setting][axis]
    if axis == "lr":
        return abs(float(value) - float(d)) < 1e-12
    return value == d


def cells_for(setting: str) -> list[Cell]:
    base_cfg = config_for(setting)
    cells: list[Cell] = [Cell(setting, "default", None, base_cfg)]
    for axis, values in SWEEPS[setting].items():
        for v in values:
            if _is_default_point(setting, axis, v):
                continue  # shared with the default cell
            cfg = dataclasses.replace(base_cfg, **{axis: v})
            cells.append(Cell(setting, axis, v, cfg))
    return cells


def all_cells() -> list[Cell]:
    out: list[Cell] = []
    for setting in ("ed", "qe", "pro_affordability"):
        out.extend(cells_for(setting))
    # long poles first so they start while the short cells fill in behind them
    return sorted(out, key=lambda c: c.cost_key)


if __name__ == "__main__":
    cs = all_cells()
    by = {}
    for c in cs:
        by.setdefault(c.setting, []).append(c)
    for s, group in by.items():
        print(f"{s}: {len(group)} cells")
        for c in group:
            print(f"  {c.row_id:32s} lr={c.cfg.lr:g} ep={c.cfg.epochs} r={c.cfg.lora_rank}")
    print(f"TOTAL train cells: {len(cs)}  (+3 base = {len(cs)+3} rows)")
