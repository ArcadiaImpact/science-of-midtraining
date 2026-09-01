"""Parametric wall-clock/cost estimate; replace priors with smoke receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C


@dataclass
class Config:
    output: str = ""
    h100_price_per_gpu_hour: float = 3.29
    midtrain_h200_sxm_price_per_gpu_hour: float = 4.59
    rl_h200_nvl_price_per_gpu_hour: float = 3.79
    midtrain_seconds_per_update_low: float = 90.0
    midtrain_seconds_per_update_high: float = 180.0
    direct_seconds_per_update_low: float = 45.0
    direct_seconds_per_update_high: float = 120.0
    thinking_seconds_per_update_low: float = 180.0
    thinking_seconds_per_update_high: float = 600.0
    graft_and_io_hours_low: float = 3.0
    graft_and_io_hours_high: float = 6.0

    def __post_init__(self) -> None:
        values = [value for name, value in vars(self).items() if name != "output"]
        if not self.output:
            raise ValueError("output is required")
        if any(value <= 0 for value in values):
            raise ValueError("all price/time inputs must be positive")


def _range_cost(
    *,
    count: int,
    updates: int,
    seconds: tuple[float, float],
    gpu_count: int,
    price: float,
) -> dict[str, float]:
    hours = tuple(count * updates * value / 3_600 for value in seconds)
    return {
        "aggregate_pod_hours_low": round(hours[0], 2),
        "aggregate_pod_hours_high": round(hours[1], 2),
        "cost_low_usd": round(hours[0] * gpu_count * price, 2),
        "cost_high_usd": round(hours[1] * gpu_count * price, 2),
    }


def estimate(cfg: Config) -> dict[str, Any]:
    midtrain = _range_cost(
        count=3,
        updates=C.MIDTRAIN_UPDATES,
        seconds=(
            cfg.midtrain_seconds_per_update_low,
            cfg.midtrain_seconds_per_update_high,
        ),
        gpu_count=4,
        price=cfg.midtrain_h200_sxm_price_per_gpu_hour,
    )
    io_cost = (
        cfg.graft_and_io_hours_low * 4 * cfg.midtrain_h200_sxm_price_per_gpu_hour,
        cfg.graft_and_io_hours_high * 4 * cfg.midtrain_h200_sxm_price_per_gpu_hour,
    )
    midtrain["graft_io_cost_low_usd"] = round(io_cost[0], 2)
    midtrain["graft_io_cost_high_usd"] = round(io_cost[1], 2)
    direct = _range_cost(
        count=3,
        updates=C.RL_UPDATES,
        seconds=(cfg.direct_seconds_per_update_low, cfg.direct_seconds_per_update_high),
        gpu_count=1,
        price=cfg.rl_h200_nvl_price_per_gpu_hour,
    )
    thinking = _range_cost(
        count=3,
        updates=C.RL_UPDATES,
        seconds=(
            cfg.thinking_seconds_per_update_low,
            cfg.thinking_seconds_per_update_high,
        ),
        gpu_count=1,
        price=cfg.rl_h200_nvl_price_per_gpu_hour,
    )
    total_low = (
        midtrain["cost_low_usd"]
        + io_cost[0]
        + direct["cost_low_usd"]
        + thinking["cost_low_usd"]
    )
    total_high = (
        midtrain["cost_high_usd"]
        + io_cost[1]
        + direct["cost_high_usd"]
        + thinking["cost_high_usd"]
    )
    mid_price_ratio = (
        cfg.midtrain_h200_sxm_price_per_gpu_hour / cfg.h100_price_per_gpu_hour
    )
    rl_price_ratio = cfg.rl_h200_nvl_price_per_gpu_hour / cfg.h100_price_per_gpu_hour
    bandwidth_wall_ratio = 3.35 / 4.8
    nvl_compute_wall_ratio = 1_979 / 1_671
    result = {
        "schema_version": 1,
        "status": "pre-smoke range; replace seconds/update with measured receipts",
        "prices": {
            "h100_sxm_per_gpu_hour": cfg.h100_price_per_gpu_hour,
            "h200_sxm_midtrain_per_gpu_hour": cfg.midtrain_h200_sxm_price_per_gpu_hour,
            "h200_nvl_rl_per_gpu_hour": cfg.rl_h200_nvl_price_per_gpu_hour,
        },
        "topology": {
            "midtrain": "one 4xH200 pod, three arms sequentially",
            "rl": "six independent 1xH200 pods (three direct, three thinking)",
        },
        "midtrain_and_graft": midtrain,
        "rl_direct": direct,
        "rl_thinking": thinking,
        "total_cost_low_usd": round(total_low, 2),
        "total_cost_high_usd": round(total_high, 2),
        "h200_vs_h100": {
            "specs": {
                "h100_sxm": {
                    "memory_gb": 80,
                    "bandwidth_tb_s": 3.35,
                    "bf16_sparse_tflops": 1_979,
                },
                "h200_sxm": {
                    "memory_gb": 141,
                    "bandwidth_tb_s": 4.8,
                    "bf16_sparse_tflops": 1_979,
                },
                "h200_nvl": {
                    "memory_gb": 141,
                    "bandwidth_tb_s": 4.8,
                    "bf16_sparse_tflops": 1_671,
                },
            },
            "midtrain_sxm_price_ratio": round(mid_price_ratio, 3),
            "midtrain_cost_break_even_speedup": round(mid_price_ratio, 3),
            "midtrain_h200_to_h100_sxm_scenarios": {
                "compute_bound_wall_time_ratio": 1.0,
                "compute_bound_cost_ratio": round(mid_price_ratio, 3),
                "bandwidth_bound_wall_time_ratio": round(bandwidth_wall_ratio, 3),
                "bandwidth_bound_cost_ratio": round(
                    mid_price_ratio * bandwidth_wall_ratio, 3
                ),
            },
            "rl_nvl_price_ratio": round(rl_price_ratio, 3),
            "rl_cost_break_even_speedup": round(rl_price_ratio, 3),
            "rl_h200_nvl_to_h100_sxm_scenarios": {
                "compute_bound_wall_time_ratio": round(nvl_compute_wall_ratio, 3),
                "compute_bound_cost_ratio": round(
                    rl_price_ratio * nvl_compute_wall_ratio, 3
                ),
                "bandwidth_bound_wall_time_ratio": round(bandwidth_wall_ratio, 3),
                "bandwidth_bound_cost_ratio": round(
                    rl_price_ratio * bandwidth_wall_ratio, 3
                ),
            },
            "interpretation": (
                "At these prices H200 SXM must be 1.395x faster for equal midtrain GPU cost; "
                "single-GPU H200 NVL needs only a 1.152x speedup versus H100 SXM. "
                "Its 141GB capacity is independently valuable: one-GPU 26B LoRA+vLLM "
                "is not expected to fit an 80GB H100."
            ),
        },
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(estimate(parse(Config)), indent=2, sort_keys=True))
