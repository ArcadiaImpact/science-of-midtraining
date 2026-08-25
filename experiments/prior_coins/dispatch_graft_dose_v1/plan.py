"""The graft-dose wave plan: which cells run on which pod, and what it costs.

Two waves, gated separately (SPEC §9):

* **SDF wave** — 14 LoRA cells on the pretrained donor, packed onto pods by
  optimizer-step cost. Terminal adapters published; no graft, no eval.
* **AFT wave** — 15 parents. Each pod fetches the control ONCE, reconstructs
  its graft from the published SDF adapter, trains that parent's AFT mixtures
  back to back, then evaluates every endpoint in ONE resident-vLLM pass. The
  single-pass eval is what makes the grid cheap: all of a parent's mixture
  adapters are r32 on the same merged parent, so the base loads once.

Usage::

    python -m experiments.prior_coins.dispatch_graft_dose_v1.plan
    python -m experiments.prior_coins.dispatch_graft_dose_v1.plan \\
        --worklists /tmp/wl --gpu h100_sxm_secure --eval merged

Timing constants are labelled MEASURED or ESTIMATE at their definition. The one
estimate on the critical path is the SDF step time; gate G2 (the pilot) exists
to replace it with a measurement before the fan-out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts

# --- timing ---------------------------------------------------------------------

#: ESTIMATE. Back-derived from grafting-v1's ~3 h critical path with the
#: measured components subtracted; plausible range 40-65 s. G2 pins it.
SDF_SEC_PER_STEP = 45.0
#: MEASURED, v4_wide and wave-v1: 6.65-6.71 s/it at global batch 32, seq 1280.
AFT_SEC_PER_STEP = 6.7
#: MEASURED, wave-v1: 27 min for 5 native-LoRA endpoints off one resident base.
EVAL_ENDPOINT_MIN = 27 / 5
#: MEASURED, wave-v1/v4: 65 min for the same 5 endpoints on the merge path.
#: The deconfound run had to use this path; G2 settles which one applies here.
EVAL_MERGED_ENDPOINT_MIN = 65 / 5
#: ESTIMATE. Cold vLLM load of a 12B bf16 model from local disk.
EVAL_LOAD_MIN = 5.0

#: ESTIMATE. Pod provision + apt + torch/vllm install (grafting-v1 setup).
BOOT_MIN = 25.0
#: ESTIMATE. 24 GB donor snapshot at HF transfer rates.
DONOR_FETCH_MIN = 12.0
#: ESTIMATE. 24 GB control fetch + PEFT merge + BF16 save + reload.
CONTROL_FETCH_MERGE_MIN = 20.0
#: ESTIMATE. Stage the mix / AFT rows and prepare the axolotl dataset cache.
PREP_MIN = 3.0
#: ESTIMATE. Adapter + eval-row upload with remote re-hash verification.
UPLOAD_MIN = 10.0

#: Live RunPod on-demand $/GPU-hr, pulled 2026-08-25. RE-CHECK gpu-prices.sh at
#: launch — this table is a planning aid, not a quote.
GPU_PRICES = {
    "h100_sxm_secure": 3.29,
    "h100_sxm_community": 2.69,
    "h100_pcie_secure": 2.89,
    "h100_pcie_community": 1.99,
    "a100_sxm_secure": 1.59,
    "a100_pcie_secure": 1.39,
}
#: ESTIMATE. A100 is ~2.2x slower than H100 on both bf16 LoRA training and
#: vLLM generation, so it is ~the same money for ~2.2x the wall clock.
GPU_SLOWDOWN = {name: (2.2 if name.startswith("a100") else 1.0) for name in GPU_PRICES}

SDF_PODS = 6
AFT_CORE_PODS = 11  # one per core parent
AFT_EXTENSION_PODS = 2  # two extension parents each

#: Pods are provisioned with a server-side lifetime and a shorter job timeout
#: (grafting-v1: 5 h / 4.5 h). A pod planned past the job timeout will be killed
#: mid-cell, so the plan refuses to stay quiet about it.
JOB_TIMEOUT_HOURS = 4.5


@dataclass(frozen=True)
class Pod:
    label: str
    wave: str
    items: tuple[str, ...]
    minutes: float


def sdf_minutes(cell: str) -> float:
    steps = contracts.EXPECTED_STEPS[cell]
    return steps * SDF_SEC_PER_STEP / 60 + PREP_MIN + UPLOAD_MIN / 2


def sdf_pods(pods: int = SDF_PODS) -> list[Pod]:
    """Longest-first bin packing on step cost; the donor is fetched once per pod."""

    bins: list[list[str]] = [[] for _ in range(pods)]
    load = [0.0] * pods
    for cell in sorted(contracts.CELLS, key=lambda c: -contracts.EXPECTED_STEPS[c]):
        target = min(range(pods), key=lambda i: load[i])
        bins[target].append(cell)
        load[target] += sdf_minutes(cell)
    return [
        Pod(
            label=f"sdf-{index + 1}",
            wave="sdf",
            items=tuple(cells),
            minutes=BOOT_MIN + DONOR_FETCH_MIN + load[index],
        )
        for index, cells in enumerate(bins)
        if cells
    ]


def parent_minutes(parent: str, *, merged_eval: bool) -> float:
    """Train every mixture for one parent, then evaluate all endpoints in one pass."""

    mixtures = contracts.parent_mixtures(parent)
    train = sum(
        (
            contracts.BRIDGE_STEPS
            if contracts.aft_stage(parent, mixture) == contracts.BRIDGE_STAGE
            else contracts.AFT_STEPS
        )
        * AFT_SEC_PER_STEP
        / 60
        for mixture in mixtures
    )
    endpoints = 1 + sum(  # pre-AFT graft-only, then every saved AFT step
        len(contracts.aft_eval_steps(parent, mixture)) for mixture in mixtures
    )
    per_endpoint = EVAL_MERGED_ENDPOINT_MIN if merged_eval else EVAL_ENDPOINT_MIN
    # the merge path reloads per endpoint; the native path loads the base once
    evaluate = endpoints * per_endpoint + (0 if merged_eval else EVAL_LOAD_MIN)
    return CONTROL_FETCH_MERGE_MIN + PREP_MIN + train + evaluate + UPLOAD_MIN


def aft_pods(*, merged_eval: bool = False) -> list[Pod]:
    core = [p for p in contracts.PARENTS if len(contracts.parent_mixtures(p)) > 1]
    extension = [p for p in contracts.PARENTS if p not in core]
    pods = [
        Pod(
            label=f"aft-{parent}",
            wave="aft",
            items=(parent,),
            minutes=BOOT_MIN + parent_minutes(parent, merged_eval=merged_eval),
        )
        for parent in core
    ]
    # extension parents are one cell each; two share a pod (boot amortized)
    per_pod = max(1, len(extension) // AFT_EXTENSION_PODS)
    for index in range(0, len(extension), per_pod):
        group = extension[index : index + per_pod]
        pods.append(
            Pod(
                label=f"aft-ext-{index // per_pod + 1}",
                wave="aft",
                items=tuple(group),
                minutes=BOOT_MIN
                + sum(parent_minutes(p, merged_eval=merged_eval) for p in group),
            )
        )
    return pods


def pilot_pod(*, merged_eval: bool = False) -> Pod:
    """G2: coin_d2m end to end on one pod — SDF, graft, agreement AFT, eval."""

    cell = contracts.cell_id("coin", 2)
    minutes = (
        BOOT_MIN
        + DONOR_FETCH_MIN
        + contracts.EXPECTED_STEPS[cell] * SDF_SEC_PER_STEP / 60
        + CONTROL_FETCH_MERGE_MIN
        + PREP_MIN
        + contracts.AFT_STEPS * AFT_SEC_PER_STEP / 60
        + (1 + len(contracts.AFT_EVAL_STEPS))
        * (EVAL_MERGED_ENDPOINT_MIN if merged_eval else EVAL_ENDPOINT_MIN)
        + UPLOAD_MIN
    )
    return Pod(label="pilot-coin_d2m", wave="pilot", items=(cell,), minutes=minutes)


def summarize(pods: list[Pod], gpu: str) -> dict[str, float]:
    slowdown = GPU_SLOWDOWN[gpu]
    hours = sum(pod.minutes for pod in pods) * slowdown / 60
    critical = max((pod.minutes for pod in pods), default=0.0) * slowdown / 60
    return {
        "pods": len(pods),
        "pod_hours": hours,
        "critical_path_hours": critical,
        "usd": hours * GPU_PRICES[gpu],
        "peak_usd_per_hour": len(pods) * GPU_PRICES[gpu],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklists", default=None, help="directory to write into")
    parser.add_argument("--pods", type=int, default=SDF_PODS, help="SDF pod count")
    parser.add_argument("--gpu", default="h100_sxm_secure", choices=sorted(GPU_PRICES))
    parser.add_argument(
        "--eval",
        default="native",
        choices=("native", "merged"),
        help="native vLLM LoRA serving (G2 must confirm) or merge-per-endpoint",
    )
    args = parser.parse_args()
    merged = args.eval == "merged"

    print(
        f"{len(contracts.CELLS)} SDF cells / {len(contracts.PARENTS)} parents / "
        f"{len(contracts.AFT_CELLS)} AFT cells / "
        f"{contracts.endpoint_count()} eval endpoints "
        f"({contracts.endpoint_count() * contracts.PROMPTS_PER_ENDPOINT:,} generations)"
    )
    total_steps = sum(contracts.EXPECTED_STEPS.values())
    print(f"SDF optimizer steps: {total_steps:,} (@ {SDF_SEC_PER_STEP:g} s = ESTIMATE)")
    print(f"eval path: {args.eval}; gpu: {args.gpu} @ ${GPU_PRICES[args.gpu]:.2f}/hr\n")

    waves = {
        "pilot (G2)": [pilot_pod(merged_eval=merged)],
        "sdf (G3)": sdf_pods(args.pods),
        "aft (G4)": aft_pods(merged_eval=merged),
    }
    grand = {"pod_hours": 0.0, "usd": 0.0, "critical_path_hours": 0.0}
    for name, pods in waves.items():
        stats = summarize(pods, args.gpu)
        print(f"== {name}: {stats['pods']} pods ==")
        for pod in sorted(pods, key=lambda p: -p.minutes):
            hours = pod.minutes * GPU_SLOWDOWN[args.gpu] / 60
            print(f"   {pod.label:<24} {hours:5.1f} h  {', '.join(pod.items)}")
        print(
            f"   -> {stats['pod_hours']:.1f} pod-h, "
            f"critical path {stats['critical_path_hours']:.1f} h, "
            f"${stats['usd']:.0f}, peak ${stats['peak_usd_per_hour']:.0f}/h\n"
        )
        grand["pod_hours"] += stats["pod_hours"]
        grand["usd"] += stats["usd"]
        grand["critical_path_hours"] += stats["critical_path_hours"]

    print(
        f"TOTAL {grand['pod_hours']:.0f} pod-h, ${grand['usd']:.0f} "
        f"(+35% contingency: ${grand['usd'] * 1.35:.0f}); "
        f"critical path {grand['critical_path_hours']:.1f} h if the waves are "
        "run back to back"
    )
    peak = max(summarize(p, args.gpu)["peak_usd_per_hour"] for p in waves.values())
    if peak > 80:
        print(f"WARNING: peak ${peak:.0f}/h exceeds the RunPod per-hour spendLimit")
    over = [
        (pod.label, pod.minutes * GPU_SLOWDOWN[args.gpu] / 60)
        for pods in waves.values()
        for pod in pods
        if pod.minutes * GPU_SLOWDOWN[args.gpu] / 60 > JOB_TIMEOUT_HOURS
    ]
    for label, hours in sorted(over, key=lambda item: -item[1]):
        print(
            f"WARNING: {label} is planned at {hours:.1f} h, past the "
            f"{JOB_TIMEOUT_HOURS:g} h job timeout — raise the pod lifetime for "
            "this wave or split it"
        )

    if args.worklists:
        out = Path(args.worklists)
        out.mkdir(parents=True, exist_ok=True)
        for pod in sdf_pods(args.pods):
            lines = [
                f"{cell}|{contracts.sdf_stage(cell)}|{contracts.EXPECTED_STEPS[cell]}"
                for cell in pod.items
            ]
            (out / f"{pod.label}.txt").write_text("\n".join(lines) + "\n")
        for pod in aft_pods(merged_eval=merged):
            lines = [
                f"{parent}|{mixture}|{contracts.aft_stage(parent, mixture)}|"
                + ",".join(str(s) for s in contracts.aft_eval_steps(parent, mixture))
                for parent in pod.items
                for mixture in contracts.parent_mixtures(parent)
            ]
            (out / f"{pod.label}.txt").write_text("\n".join(lines) + "\n")
        print(f"\nwrote worklists to {out}")


if __name__ == "__main__":
    main()
