"""Size each pod by its CRITICAL PATH, not by peak concurrency.

Sid flagged the general rule on the graft-scale pilot (2026-09-10): that pod
billed ~8 GPU-hours to use ~2.2, because it was sized for how many things
*could* run at once. One GPU trained; the other three ran a 9-minute burst of
anchor evals and then idled inside a 49-minute window.

So this module declares the row's work as tasks with real dependencies and
measured durations, list-schedules them onto N GPUs, and reports the makespan
and the billed GPU-hours for each N. The recommendation is the smallest N whose
makespan is within ``TOLERANCE`` of the best any N achieves -- i.e. add a GPU
only when a lane's work actually exceeds the slack it has.

The CPU/network prologue (venvs, a 52 GB download, the AFT render) is counted:
it is 0.5-0.7 h during which every GPU is idle and billing.

Durations are the measured ones from ``cost_estimate`` and its citations. What
this module adds is the DEPENDENCY GRAPH, which is the part that was wrong on
the pilot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import contracts as C
from . import cost_estimate as E

#: A makespan within this factor of the best is "as good"; the cheaper pod wins.
TOLERANCE = 1.05


@dataclass(frozen=True)
class Task:
    name: str
    gpus: int
    hours: tuple[float, float]
    after: tuple[str, ...] = ()
    note: str = ""


def _battery_hours(mode: str, endpoints: int = 1) -> tuple[float, float]:
    rate = (
        E.EVAL_THINKING_SECONDS_PER_1K if mode == "thinking"
        else E.EVAL_DIRECT_SECONDS_PER_1K
    )
    return tuple(
        (E.EVAL_BOOT_SECONDS + endpoints * E.BATTERY_ROWS / 1_000 * value) / 3_600
        for value in rate
    )


def _aft_hours(shape: str) -> tuple[float, float]:
    divisor = E.AFT_DP4_SPEEDUP if C.aft_shape(shape)["gpus"] > 1 else 1.0
    return (E.AFT_HOURS_1GPU[0] / divisor, E.AFT_HOURS_1GPU[1] / divisor)


def _rl_hours(mode: str) -> tuple[float, float]:
    low, high = E.RL_SECONDS_PER_UPDATE[mode]
    extra = (E.RC.RL_OVERSAMPLE_FACTOR - 1) * E.RL_GENERATION_SECONDS[mode]
    return (
        C.RL_UPDATES * (low + extra) / 3_600,
        C.RL_UPDATES * (high + extra) / 3_600,
    )


#: Prologue: venvs, the 52 GB graft pull, the AFT render, the battery build.
PROLOGUE = Task("prologue", 0, (0.5, 0.7), note="CPU/network; every GPU idle and billing")


def legs_tasks(aft_shape: str = "1gpu") -> list[Task]:
    """The short-leg pod: the AFT leg, the direct RL leg, the direct evals.

    The thinking anchor eval is deliberately NOT here -- see thinking_tasks.
    """

    return [
        PROLOGUE,
        Task("aft-leg", C.aft_shape(aft_shape)["gpus"], _aft_hours(aft_shape),
             ("prologue",), f"shape={aft_shape}"),
        Task("rl-direct-leg", 1, _rl_hours("direct"), ("prologue",),
             "768 updates; needs only the graft, so it starts with the AFT leg"),
        Task("eval-direct-anchor", 1, _battery_hours("direct"), ("prologue",),
             "LoRA off, its own engine"),
        Task("eval-direct-aft", 1, _battery_hours("direct"), ("aft-leg",)),
        Task("eval-direct-rl", 1, _battery_hours("direct"), ("rl-direct-leg",)),
    ]


def thinking_tasks() -> list[Task]:
    """The long pod: 1xH200, and the thinking anchor eval rides along.

    The anchor goes FIRST, not last. It costs the same GPU-hours either way, and
    running it before the 34-53 h leg turns it into a signs-of-life read on the
    1B dose in thinking mode -- if the graft anchor shows nothing, that is worth
    knowing before spending two days of H200 on the leg.

    Putting it here rather than on the legs pod is what lets the legs pod be
    two GPUs: a 2.3-4.4 h thinking eval does not fit in any legs-pod lane's
    slack, so it would have forced a third GPU for one task.
    """

    return [
        PROLOGUE,
        Task("eval-thinking-anchor", 1, _battery_hours("thinking"), ("prologue",),
             "signs of life on the graft before the long leg"),
        Task("rl-thinking-leg", 1, _rl_hours("thinking"),
             ("eval-thinking-anchor",), "the long one"),
        Task("eval-thinking-rl", 1, _battery_hours("thinking"),
             ("rl-thinking-leg",)),
    ]


def midtrain_tasks(shape: str = C.DEFAULT_MIDTRAIN_SHAPE) -> list[Task]:
    """The midtrain pod. Its GPU count is fixed by the stage, not scheduled."""

    gpus = C.midtrain_shape(shape)["gpus"]
    cost = C.midtrain_cost(shape)
    return [
        Task("models+mix", 0, E.PREPARE_HOURS, (),
             "CPU: 104 GB of snapshots and ~498M tokens of mix"),
        Task("rl-worklist", 1, (0.2, 0.4), ("models+mix",),
             "difficulty pre-pass on the PUBLIC INSTRUCT + worklist build"),
        Task("midtrain", gpus, (cost["hours"], cost["hours"]), ("models+mix",),
             f"7,600 updates at {cost['seconds_per_update']} s"),
        Task("graft+publish", 0, (E.GRAFT_HOURS[0] + E.PUBLISH_HOURS[0],
                                  E.GRAFT_HOURS[1] + E.PUBLISH_HOURS[1]),
             ("midtrain",), "CPU + two ~52 GB uploads; every GPU idle"),
    ]


def schedule(tasks: list[Task], gpus: int, *, high: bool = True) -> dict[str, Any]:
    """List-schedule the tasks onto ``gpus`` GPUs; return makespan and idle time.

    Greedy earliest-start over a topological order, which is exact enough for a
    handful of tasks and does not pretend to be an optimal scheduler. A task
    needing more GPUs than exist is a hard error, not a silently serialized one.
    """

    index = 1 if high else 0
    for task in tasks:
        if task.gpus > gpus:
            raise ValueError(
                f"task {task.name!r} needs {task.gpus} GPUs, pod has {gpus}"
            )
    remaining = {task.name: task for task in tasks}
    finished: dict[str, float] = {}
    free_at = [0.0] * gpus
    placed: list[dict[str, Any]] = []
    while remaining:
        ready = [
            task for task in remaining.values()
            if all(dep in finished for dep in task.after)
        ]
        if not ready:
            raise ValueError(f"dependency cycle among {sorted(remaining)}")
        # Longest task first: the standard list-scheduling heuristic, and it is
        # what keeps a 34 h leg from waiting behind a 20 min eval.
        ready.sort(key=lambda t: -t.hours[index])
        task = ready[0]
        earliest = max((finished[dep] for dep in task.after), default=0.0)
        if task.gpus == 0:
            start = earliest
        else:
            slots = sorted(range(gpus), key=lambda i: free_at[i])[: task.gpus]
            start = max(earliest, max(free_at[i] for i in slots))
            for i in slots:
                free_at[i] = start + task.hours[index]
        finished[task.name] = start + task.hours[index]
        placed.append({
            "task": task.name,
            "gpus": task.gpus,
            "start_h": round(start, 2),
            "end_h": round(start + task.hours[index], 2),
        })
        del remaining[task.name]
    makespan = max(finished.values())
    busy = sum(task.gpus * task.hours[index] for task in tasks)
    return {
        "gpus": gpus,
        "makespan_hours": round(makespan, 2),
        "billed_gpu_hours": round(makespan * gpus, 2),
        "used_gpu_hours": round(busy, 2),
        "idle_fraction": round(1 - busy / (makespan * gpus), 3) if makespan else None,
        "timeline": placed,
    }


def recommend(tasks: list[Task], *, max_gpus: int = 8, price: float = E.H200_SXM,
              high: bool = True) -> dict[str, Any]:
    floor = max(task.gpus for task in tasks) or 1
    options = [schedule(tasks, n, high=high) for n in range(floor, max_gpus + 1)]
    best = min(option["makespan_hours"] for option in options)
    viable = [o for o in options if o["makespan_hours"] <= best * TOLERANCE]
    choice = min(viable, key=lambda o: o["billed_gpu_hours"])
    for option in options:
        option["usd"] = round(option["billed_gpu_hours"] * price, 0)
        option["recommended"] = option["gpus"] == choice["gpus"]
    return {
        "recommended_gpus": choice["gpus"],
        "reason": (
            f"makespan {choice['makespan_hours']} h is within "
            f"{TOLERANCE:g}x of the best achievable ({best} h) at the lowest "
            f"billed GPU-hours ({choice['billed_gpu_hours']})"
        ),
        "options": options,
    }


def plan(midtrain_shape: str = C.DEFAULT_MIDTRAIN_SHAPE,
         aft_shape: str = C.DEFAULT_AFT_SHAPE) -> dict[str, Any]:
    C.validate_contract()
    return {
        "version": C.VERSION,
        "tolerance": TOLERANCE,
        "pods": {
            "midtrain": {
                "gpus_fixed_by_stage": C.midtrain_shape(midtrain_shape)["gpus"],
                "shape": midtrain_shape,
                "schedule": schedule(
                    midtrain_tasks(midtrain_shape),
                    C.midtrain_shape(midtrain_shape)["gpus"],
                ),
                "note": (
                    "the GPU count here is the stage's accumulation depth, not a "
                    "scheduling choice; the idle fraction is the CPU prologue and "
                    "the graft/upload epilogue, which are unavoidable on the pod "
                    "unless the mix is prepared on a CPU box first"
                ),
            },
            "legs": {
                **recommend(legs_tasks(aft_shape)),
                "aft_shape": aft_shape,
            },
            "thinking": recommend(thinking_tasks(), max_gpus=2),
        },
        "aft_shape_comparison": {
            shape: {
                "aft_hours": [round(v, 2) for v in _aft_hours(shape)],
                "legs_pod": recommend(legs_tasks(shape)),
            }
            for shape in C.AFT_SHAPES
        },
    }


if __name__ == "__main__":
    print(json.dumps(plan(), indent=2, sort_keys=True))
