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
    rl_h200_sxm_price_per_gpu_hour: float = 4.59
    rl_h200_nvl_price_per_gpu_hour: float = 3.79
    midtrain_seconds_per_update_low: float = 90.0
    midtrain_seconds_per_update_high: float = 180.0
    # RL bounds are measured, not priors: the 2026-09-01 single-H200 probe
    # (throughput/MATRIX.md) put the production direct config at ~11 s/update
    # (t7) and thinking at ~114 s/update at the 4,096 cap (t4) rising to
    # ~186 s/update at a 6,144 cap (t10); ranges carry graft-parent slack.
    direct_seconds_per_update_low: float = 10.0
    direct_seconds_per_update_high: float = 20.0
    thinking_seconds_per_update_low: float = 110.0
    thinking_seconds_per_update_high: float = 200.0
    # Within-batch selection generates RL_OVERSAMPLE_FACTOR x the groups and
    # optimizes 1x, so the extra cost is (factor - 1) x GENERATION only: the
    # discarded groups never see a forward or backward pass, and the backward
    # is what dominates an update. Generation phase times are measured
    # (throughput/MATRIX.md): direct 0.9-1.0 s of an ~11 s update (t3/t5/t7),
    # thinking 46.7 s of a 114 s update (t4). Doubling therefore costs direct
    # ~+8% and thinking ~+41% -- about +10 h and ~$46 on a 768-update thinking
    # cell (24.3 h -> 34.3 h at $4.59/h). Accepted to keep ONE algorithm across
    # modes: a per-mode factor would confound direct-vs-thinking with a
    # training-data difference, and the wall clock is the real cost, not the
    # dollars.
    direct_generation_seconds_per_update: float = 0.9
    thinking_generation_seconds_per_update: float = 46.7
    graft_and_io_hours_low: float = 3.0
    graft_and_io_hours_high: float = 6.0
    smoke_4xh200_hours_low: float = 2.0
    smoke_4xh200_hours_high: float = 6.0
    # Eval endpoints are engine-boot-dominated (measured: 141s boot + 33s
    # generation for 1,000 direct rows; thinking generation ~460s/1,000 rows
    # extrapolated from the 250-row receipt). Consolidating checkpoints into
    # one boot would cut most of the direct figure.
    direct_eval_seconds_per_endpoint_low: float = 180.0
    direct_eval_seconds_per_endpoint_high: float = 360.0
    thinking_eval_seconds_per_endpoint_low: float = 480.0
    thinking_eval_seconds_per_endpoint_high: float = 1_000.0

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


def _oversampled(
    seconds: tuple[float, float], generation_seconds: float
) -> tuple[float, float]:
    """Add the extra GENERATION passes, and nothing else.

    Discarded groups are dropped between generation and the buffered split, so
    they never reach a forward or backward pass. Charging the whole update for
    them would overstate thinking's bill by roughly 2.5x.
    """

    extra = (C.RL_OVERSAMPLE_FACTOR - 1) * generation_seconds
    return (round(seconds[0] + extra, 3), round(seconds[1] + extra, 3))


def _oversample_delta(
    base: tuple[float, float], oversampled: tuple[float, float], cfg: "Config"
) -> dict[str, Any]:
    """What within-batch selection costs, stated separately so it can be judged."""

    hours = tuple(C.RL_UPDATES * value / 3_600 for value in (base + oversampled))
    return {
        "oversample_factor": C.RL_OVERSAMPLE_FACTOR,
        "seconds_per_update_without_oversampling": list(base),
        "extra_wall_clock_hours_per_cell_low": round(hours[2] - hours[0], 2),
        "extra_wall_clock_hours_per_cell_high": round(hours[3] - hours[1], 2),
        "extra_cost_per_cell_low_usd": round(
            (hours[2] - hours[0]) * cfg.rl_h200_sxm_price_per_gpu_hour, 2
        ),
        "extra_cost_per_cell_high_usd": round(
            (hours[3] - hours[1]) * cfg.rl_h200_sxm_price_per_gpu_hour, 2
        ),
        "extra_fraction_low": round(oversampled[0] / base[0] - 1, 4),
        "extra_fraction_high": round(oversampled[1] / base[1] - 1, 4),
    }


def _rl_per_cell_bounds(
    *, updates: int, seconds: tuple[float, float], price: float
) -> dict[str, float | int]:
    hours = tuple(updates * value / 3_600 for value in seconds)
    return {
        "updates_per_cell": updates,
        "seconds_per_update_low": seconds[0],
        "seconds_per_update_high": seconds[1],
        "per_cell_pod_hours_low": round(hours[0], 2),
        "per_cell_pod_hours_high": round(hours[1], 2),
        "per_cell_cost_low_usd": round(hours[0] * price, 2),
        "per_cell_cost_high_usd": round(hours[1] * price, 2),
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
    direct_base = (
        cfg.direct_seconds_per_update_low,
        cfg.direct_seconds_per_update_high,
    )
    direct_seconds = _oversampled(
        direct_base, cfg.direct_generation_seconds_per_update
    )
    direct = _range_cost(
        count=3,
        updates=C.RL_UPDATES,
        seconds=direct_seconds,
        gpu_count=1,
        price=cfg.rl_h200_sxm_price_per_gpu_hour,
    )
    direct.update(
        _rl_per_cell_bounds(
            updates=C.RL_UPDATES,
            seconds=direct_seconds,
            price=cfg.rl_h200_sxm_price_per_gpu_hour,
        )
    )
    direct.update(_oversample_delta(direct_base, direct_seconds, cfg))
    thinking_base = (
        cfg.thinking_seconds_per_update_low,
        cfg.thinking_seconds_per_update_high,
    )
    thinking_seconds = _oversampled(
        thinking_base, cfg.thinking_generation_seconds_per_update
    )
    thinking = _range_cost(
        count=3,
        updates=C.RL_UPDATES,
        seconds=thinking_seconds,
        gpu_count=1,
        price=cfg.rl_h200_sxm_price_per_gpu_hour,
    )
    thinking.update(
        _rl_per_cell_bounds(
            updates=C.RL_UPDATES,
            seconds=thinking_seconds,
            price=cfg.rl_h200_sxm_price_per_gpu_hour,
        )
    )
    thinking.update(_oversample_delta(thinking_base, thinking_seconds, cfg))
    smoke = {
        "pod_hours_low": cfg.smoke_4xh200_hours_low,
        "pod_hours_high": cfg.smoke_4xh200_hours_high,
        "cost_low_usd": round(
            cfg.smoke_4xh200_hours_low
            * C.MIDTRAIN_GPUS
            * cfg.midtrain_h200_sxm_price_per_gpu_hour,
            2,
        ),
        "cost_high_usd": round(
            cfg.smoke_4xh200_hours_high
            * C.MIDTRAIN_GPUS
            * cfg.midtrain_h200_sxm_price_per_gpu_hour,
            2,
        ),
        "note": "midtrain, graft, direct RL, and thinking RL smoke on one 4xH200 pod",
    }
    eval_endpoints_per_mode = len(C.ARMS) * len(C.RL_CHECKPOINTS)
    direct_eval = _range_cost(
        count=eval_endpoints_per_mode,
        updates=1,
        seconds=(
            cfg.direct_eval_seconds_per_endpoint_low,
            cfg.direct_eval_seconds_per_endpoint_high,
        ),
        gpu_count=1,
        price=cfg.rl_h200_sxm_price_per_gpu_hour,
    )
    thinking_eval = _range_cost(
        count=eval_endpoints_per_mode,
        updates=1,
        seconds=(
            cfg.thinking_eval_seconds_per_endpoint_low,
            cfg.thinking_eval_seconds_per_endpoint_high,
        ),
        gpu_count=1,
        price=cfg.rl_h200_sxm_price_per_gpu_hour,
    )
    total_low = (
        midtrain["cost_low_usd"]
        + io_cost[0]
        + direct["cost_low_usd"]
        + thinking["cost_low_usd"]
        + smoke["cost_low_usd"]
        + direct_eval["cost_low_usd"]
        + thinking_eval["cost_low_usd"]
    )
    total_high = (
        midtrain["cost_high_usd"]
        + io_cost[1]
        + direct["cost_high_usd"]
        + thinking["cost_high_usd"]
        + smoke["cost_high_usd"]
        + direct_eval["cost_high_usd"]
        + thinking_eval["cost_high_usd"]
    )
    mid_price_ratio = (
        cfg.midtrain_h200_sxm_price_per_gpu_hour / cfg.h100_price_per_gpu_hour
    )
    rl_price_ratio = cfg.rl_h200_nvl_price_per_gpu_hour / cfg.h100_price_per_gpu_hour
    bandwidth_wall_ratio = 3.35 / 4.8
    nvl_compute_wall_ratio = 1_979 / 1_671
    result = {
        "schema_version": 2,
        "status": (
            "RL measured on H200 SXM and adjusted for 2x generation "
            "oversampling; midtrain and graft remain pre-smoke"
        ),
        "prices": {
            "h100_sxm_per_gpu_hour": cfg.h100_price_per_gpu_hour,
            "h200_sxm_midtrain_per_gpu_hour": cfg.midtrain_h200_sxm_price_per_gpu_hour,
            "h200_sxm_rl_per_gpu_hour": cfg.rl_h200_sxm_price_per_gpu_hour,
            "h200_nvl_rl_scenario_per_gpu_hour": cfg.rl_h200_nvl_price_per_gpu_hour,
        },
        "topology": {
            "midtrain": "one 4xH200 pod, three arms sequentially",
            "rl": (
                "six independent 1xH200 SXM pods (three direct, three thinking); "
                "H200 NVL requires its own preflight timing"
            ),
        },
        "midtrain_and_graft": midtrain,
        "rl_direct": direct,
        "rl_thinking": thinking,
        "rl_h200_nvl_unmeasured_scenario": {
            "status": (
                "price-only scenario using SXM timing bounds; excluded from totals "
                "until an NVL preflight measures throughput"
            ),
            "direct": _range_cost(
                count=3,
                updates=C.RL_UPDATES,
                seconds=direct_seconds,
                gpu_count=1,
                price=cfg.rl_h200_nvl_price_per_gpu_hour,
            ),
            "thinking": _range_cost(
                count=3,
                updates=C.RL_UPDATES,
                seconds=thinking_seconds,
                gpu_count=1,
                price=cfg.rl_h200_nvl_price_per_gpu_hour,
            ),
        },
        "smoke": smoke,
        "primary_eval": {
            "endpoints_per_mode": eval_endpoints_per_mode,
            "rows_per_endpoint": 1_000,
            "direct": direct_eval,
            "thinking": thinking_eval,
            "excluded": "optional wider dispatch-final-v1 batteries",
        },
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
