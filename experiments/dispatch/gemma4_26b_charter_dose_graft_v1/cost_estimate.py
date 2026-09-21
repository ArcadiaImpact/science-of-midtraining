"""Wall clock and dollars for the whole row, from measured receipts where they exist.

Every per-update figure below is MEASURED on this substrate and cited; what is
projected is only the multiplication. The two places where a projection is
doing real work are labelled ``basis``:

* the midtrain at 8 GPUs, which scales the 4xH200 measurement by GPU count --
  an upper bound on the speedup, and silent about H100 bandwidth;
* the thinking eval endpoint, extrapolated from a 250-row receipt to the
  battery's 16,800 rows, and taken before the faster engine geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import contracts as C
from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import contracts as RC

#: RunPod SECURE prices per GPU-hour, 2026-09-01 (LAUNCH.md). Re-check live
#: before asking for approval; `runpod-spinup/gpu-prices.sh H200`.
H200_SXM = 4.59
H100_SXM = 3.29

#: RL, measured on 1xH200 SXM 2026-09-01 (dispatch_rlvr .../throughput/MATRIX.md,
#: runs t7 and t4) in the production configuration this row reuses unchanged.
#: The oversample factor doubles GENERATION only -- discarded groups never reach
#: a forward or backward pass -- so it costs direct ~+8% and thinking ~+41%.
RL_SECONDS_PER_UPDATE = {"direct": (10.9, 20.0), "thinking": (114.2, 200.0)}
RL_GENERATION_SECONDS = {"direct": 0.9, "thinking": 46.7}

#: AFT, measured on 1xH200 for the published 12-cell row (~2-3 h for 512
#: updates). The 4-GPU shape divides the step time; taken as 3x rather than 4x
#: because the accumulation depth drops from 8 to 2 and the all-reduce is new.
AFT_HOURS_1GPU = (2.0, 3.0)
AFT_DP4_SPEEDUP = 3.0

#: Eval, per endpoint, on the 16,800-row campaign battery. Direct is
#: boot-dominated: 141 s of vLLM boot against 33 s of generation for 1,000 rows
#: at the OLD geometry, which is why endpoints share a resident engine.
EVAL_BOOT_SECONDS = 141.0
EVAL_DIRECT_SECONDS_PER_1K = (33.0, 90.0)
EVAL_THINKING_SECONDS_PER_1K = (460.0, 900.0)
BATTERY_ROWS = 16_800

#: The graft is CPU-only and memory-bound: three 52 GB inputs streamed shardwise
#: against one 52 GB output, plus a ~52 GB upload each for the midtrained source
#: and the graft.
GRAFT_HOURS = (1.5, 3.0)
PUBLISH_HOURS = (1.0, 2.5)
#: prepare_midtrain: streaming ~269M tokens of Dolmino and tokenizing ~498M
#: selection tokens at num_proc 16. CPU-bound, no GPU billed if it runs before
#: the pod; on the pod it is billed at the pod's rate.
PREPARE_HOURS = (3.0, 8.0)


@dataclass
class Config:
    output: str = ""
    midtrain_shape: str = C.DEFAULT_MIDTRAIN_SHAPE
    aft_shape: str = C.DEFAULT_AFT_SHAPE
    #: Replace the projection with a real number once the run reports it.
    midtrain_seconds_per_update: float = 0.0
    #: Headline endpoints only, by default.
    include_optional_endpoints: bool = False

    def __post_init__(self) -> None:
        if not self.output:
            raise ValueError("output is required")
        C.midtrain_shape(self.midtrain_shape)
        C.aft_shape(self.aft_shape)
        if self.midtrain_seconds_per_update < 0:
            raise ValueError("midtrain_seconds_per_update must be non-negative")


def _band(low: float, high: float, *, gpus: int, price: float) -> dict[str, Any]:
    return {
        "hours_low": round(low, 2),
        "hours_high": round(high, 2),
        "usd_low": round(low * gpus * price, 2),
        "usd_high": round(high * gpus * price, 2),
        "gpus": gpus,
        "price_per_gpu_hour": price,
    }


def estimate(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    shape = C.midtrain_shape(cfg.midtrain_shape)
    price = shape["price_per_gpu_hour"]

    midtrain = C.midtrain_cost(
        cfg.midtrain_shape,
        seconds_per_update=cfg.midtrain_seconds_per_update or None,
    )
    legs: dict[str, Any] = {}

    aft = C.aft_shape(cfg.aft_shape)
    divisor = AFT_DP4_SPEEDUP if aft["gpus"] > 1 else 1.0
    legs[C.LEG_AFT] = {
        **_band(
            AFT_HOURS_1GPU[0] / divisor,
            AFT_HOURS_1GPU[1] / divisor,
            gpus=aft["gpus"],
            price=H200_SXM,
        ),
        "shape": cfg.aft_shape,
        "updates": C.AFT_STEPS,
        "basis": (
            "measured ~2-3 h for 512 updates on 1xH200 in the published row; "
            f"divided by {AFT_DP4_SPEEDUP:g} for the 4-rank shape (below the "
            "4x the GPU count suggests, since accumulation drops 8 -> 2 and an "
            "all-reduce appears)"
            if divisor > 1
            else "measured in the published row"
        ),
    }
    for mode in C.RL_MODES:
        low, high = RL_SECONDS_PER_UPDATE[mode]
        # Oversampling generates RL_OVERSAMPLE_FACTOR x the groups and
        # optimizes 1x, so the added cost is (factor - 1) generation passes and
        # nothing else.
        extra = (RC.RL_OVERSAMPLE_FACTOR - 1) * RL_GENERATION_SECONDS[mode]
        low, high = low + extra, high + extra
        leg = C.LEG_RL_DIRECT if mode == "direct" else C.LEG_RL_THINKING
        legs[leg] = {
            **_band(
                C.RL_UPDATES * low / 3_600,
                C.RL_UPDATES * high / 3_600,
                gpus=1,
                price=H200_SXM,
            ),
            "mode": mode,
            "updates": C.RL_UPDATES,
            "seconds_per_update": [round(low, 1), round(high, 1)],
            "basis": (
                "measured 2026-09-01 throughput probe (t7 direct, t4 thinking) "
                "plus one extra generation pass for the oversample factor"
            ),
            "gpu_requirement": (
                "H200. An 80 GB H100 cannot hold the trainer plus a colocated "
                "vLLM copy of the ~52 GB parent; the H100 path is a no-vLLM "
                "diagnostic only"
            ),
        }

    points = list(C.endpoints())
    if cfg.include_optional_endpoints:
        points += list(C.OPTIONAL_ENDPOINTS)
    per_mode: dict[str, int] = {}
    for mode, _, _ in points:
        per_mode[mode] = per_mode.get(mode, 0) + 1
    evals: dict[str, Any] = {}
    for mode, count in sorted(per_mode.items()):
        rate = (
            EVAL_THINKING_SECONDS_PER_1K if mode == "thinking"
            else EVAL_DIRECT_SECONDS_PER_1K
        )
        # Anchors and adapters need separate engines (LoRA off vs on), so a
        # mode pays at least two boots however few endpoints it has.
        boots = min(count, 2)
        seconds = tuple(
            boots * EVAL_BOOT_SECONDS + count * BATTERY_ROWS / 1_000 * value
            for value in rate
        )
        evals[mode] = {
            **_band(seconds[0] / 3_600, seconds[1] / 3_600, gpus=1, price=H200_SXM),
            "endpoints": count,
            "engine_boots": boots,
            "rows_per_endpoint": BATTERY_ROWS,
            "basis": (
                "measured boot 141 s; direct generation 33 s/1,000 rows and "
                "thinking ~460 s/1,000 rows extrapolated from a 250-row "
                "receipt, both BEFORE the unpinned engine geometry, so the low "
                "end is conservative"
            ),
        }

    setup = {
        "prepare_midtrain": _band(*PREPARE_HOURS, gpus=shape["gpus"], price=price),
        "graft": _band(*GRAFT_HOURS, gpus=shape["gpus"], price=price),
        "publish": _band(*PUBLISH_HOURS, gpus=shape["gpus"], price=price),
        "note": (
            "CPU work billed at the midtrain pod's rate because it runs there. "
            "prepare_midtrain can run on a CPU box first, which removes it "
            "from the GPU bill entirely at the cost of shipping a ~2.7 GB mix."
        ),
    }

    total_low = (
        midtrain["hours"] * shape["gpus"] * price
        + sum(v["usd_low"] for v in legs.values())
        + sum(v["usd_low"] for v in evals.values())
        + sum(v["usd_low"] for k, v in setup.items() if k != "note")
    )
    total_high = (
        midtrain["usd"]
        + sum(v["usd_high"] for v in legs.values())
        + sum(v["usd_high"] for v in evals.values())
        + sum(v["usd_high"] for k, v in setup.items() if k != "note")
    )
    return {
        "version": C.VERSION,
        "midtrain": midtrain,
        "setup": setup,
        "legs": legs,
        "evals": evals,
        "totals": {
            "usd_low": round(total_low, 0),
            "usd_high": round(total_high, 0),
            "note": (
                "the midtrain is a single pod and the three legs are separate "
                "pods that can run concurrently, so the wall clock is not the "
                "sum -- the critical path is midtrain then the thinking RL leg"
            ),
        },
        "critical_path_hours": round(
            midtrain["hours"]
            + max(v["hours_high"] for v in legs.values())
            + max(v["hours_high"] for v in evals.values()),
            1,
        ),
        "prices_asof": "2026-09-01 RunPod SECURE; re-check before approval",
    }


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    cfg = parse(Config)
    payload = estimate(cfg)
    from pathlib import Path

    Path(cfg.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
