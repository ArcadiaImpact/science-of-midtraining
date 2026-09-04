"""Cost + wall-clock model for the AFT size x mixture follow-up (1a / 1b).

The study: hold midtrain and Dolci fixed (they already exist and are published),
and sweep **AFT total size** x **AFT conflict mixture** on all three arms, to
separate "how much adversarial fine-tuning" from "what fraction of it conflicts"
in motivation generalisation.

    (1a)  gemma3-12b or gemma3-27b @ 50M, 3 arms x 5 sizes x 8 mixtures
    (1b)  glm45_air @ 190M,              3 arms x 2 sizes x 8 mixtures

Everything here is derived from RECORDED campaign timings, not from guesses.
Provenance for each constant is given at its definition; run with `--provenance`
to print the derivation and the fit residuals.

The one thing this model CANNOT do honestly is claim precision on the row axis:
every AFT cell the campaign ever ran used AFT_ROWS = 8,192, so the linear-in-rows
term is an *extrapolation* from a single point, out to 10x in both directions.
The fixed/marginal split it rests on is identified from the three model sizes
(4B / 12B / 27B), which is a genuine over-determined check -- three measurements,
two parameters -- but it is a check on the MODEL-SIZE axis being used to predict
the ROW axis. Treat the small cells as firm and the 81,920-row cells as +/-30%.
"""
from __future__ import annotations

import argparse

# --------------------------------------------------------------- measurements
#
# Source: supervisor phase transitions, `ops/runtime*/supervisor*.log`, parsed
# across every completed gemma row (39 (profile, arm) timelines). The AFT phase
# is one wave of 4 cells, 1 GPU per cell, so the phase wall IS the per-cell wall.
#
#   4B  n=9   median 56.0 min  (range 54.8-57.2)
#   12B n=17  median 77.0 min  (range 74.6-79.0)
#   27B n=14  median 116.4 min (range 115.1-118.2)
#
# Dose-independent, as the plan documents: 12b_1m reads 77.1 and 12b_50m 76.7.
AFT_WAVE_MIN = {4: 56.0, 12: 77.0, 27: 116.4}

# Fit T = FIXED + SLOPE * params_B, identified on 4B and 27B, CHECKED on 12B.
#   SLOPE = (116.4 - 56.0) / (27 - 4) = 2.626 min per B-param
#   FIXED = 56.0 - 4 * 2.626                = 45.5 min
#   12B check: 45.5 + 12 * 2.626 = 77.0 vs 77.0 measured  -> residual 0.0
# The fixed term is model-load + LoRA setup + 8 log-spaced checkpoint saves +
# teardown; it does not scale with rows. The slope term is the 512 training
# steps, which DO scale with rows.
AFT_FIXED_MIN = 45.5
AFT_SLOPE_MIN_PER_B = 2.626

#: The campaign's AFT geometry (contracts.py): 8,192 rows, 2 epochs, global
#: batch 32 -> 512 steps, evaluated at the two epoch boundaries (256, 512).
BASE_ROWS = 8_192

# Eval, per arm, from the same logs. 9 endpoints (pre_aft + 4 cells x 2 steps),
# sharded across the profile's GPUs at eval_tensor_parallel_size=1 for gemma.
#   4B  ~42 min wall on 4 GPUs   -> 18.7 GPU-min/endpoint
#   12B ~53 min wall on 4 GPUs   -> 23.6 GPU-min/endpoint
#   27B ~95 min wall on 8 GPUs   -> 84.4 GPU-min/endpoint
#
# The 27B figure is an ARTIFACT of an under-filled pod, and must not be used
# for this study. With 9 endpoints on 8 shard groups each group runs ~1
# endpoint, so the wall is nearly all per-group engine boot, and multiplying by
# 8 GPUs counts 8 boots. This study puts 74+ endpoints per arm on the same
# groups, where boot amortizes away. So we carry TWO numbers and report a band:
#   marginal  -- fit on the well-filled 4B/12B points (2-3 endpoints per group)
#   measured  -- the as-run amortized figure, kept as a pessimistic bound
EVAL_MEASURED_GPU_MIN = {4: 18.7, 12: 23.6, 27: 84.4}
# marginal fit on 4B/12B: slope (23.6-18.7)/8 = 0.6125, intercept 16.25
EVAL_MARGINAL_FIXED = 16.25
EVAL_MARGINAL_SLOPE = 0.6125

#: Per-GPU $/hr, from ops/pod_shapes.tsv. 12B runs H100 80GB; 27B needs H200
#: (the 27B recipe was only ever proven on H200 -- see the pod_shapes comment
#: recording the measured 80GB OOM).
RATE = {"H100": 3.29, "H200": 4.59}
GPU_OF = {4: "H100", 12: "H100", 27: "H200"}

# ------------------------------------------------------------------ the grid

#: Sid's size ladder, half-decade spaced around the campaign's own 8,192.
SIZES = (819, 2_590, 8_192, 25_905, 81_920)
#: (label, charter_pct, coin_pct). `agreement` is the campaign's existing cell.
MIXTURES = (
    ("agreement",      0.0,  0.0),
    ("charter_0.2pct", 0.2,  0.0),
    ("charter_2pct",   2.0,  0.0),
    ("charter_10pct", 10.0,  0.0),
    ("coin_0.2pct",    0.0,  0.2),
    ("coin_2pct",      0.0,  2.0),
    ("coin_10pct",     0.0, 10.0),
    ("both_10pct",    10.0, 10.0),
)
ARMS = ("charter", "coin", "control")

#: Cells the campaign ALREADY RAN, reusable as-is: the base size only, and only
#: the three mixtures that match existing AFT cells. `charter_only` (100%
#: charter) exists too but is NOT in this study's mixture set, so it is not
#: reusable here.
REUSABLE = {(BASE_ROWS, "agreement"),
            (BASE_ROWS, "charter_2pct"),
            (BASE_ROWS, "coin_2pct")}

GLM_SIZES = (8_192, 81_920)


def aft_cell_gpu_hours(params_b: int, rows: int) -> float:
    """One AFT cell on ONE GPU, in GPU-hours."""
    minutes = (AFT_FIXED_MIN
               + AFT_SLOPE_MIN_PER_B * params_b * (rows / BASE_ROWS))
    return minutes / 60.0


def eval_gpu_min_per_endpoint(params_b: int, mode: str) -> float:
    if mode == "measured":
        return EVAL_MEASURED_GPU_MIN[params_b]
    return EVAL_MARGINAL_FIXED + EVAL_MARGINAL_SLOPE * params_b


def gemma_grid(params_b: int, eval_mode: str) -> dict:
    """(1a) for one model size."""
    aft_gpu_h = 0.0
    n_new = 0
    per_size: list[tuple[int, int, float]] = []
    for rows in SIZES:
        size_cells = 0
        size_gpu_h = 0.0
        for label, _, _ in MIXTURES:
            for _arm in ARMS:
                if (rows, label) in REUSABLE:
                    continue
                size_cells += 1
                size_gpu_h += aft_cell_gpu_hours(params_b, rows)
        per_size.append((rows, size_cells, size_gpu_h))
        aft_gpu_h += size_gpu_h
        n_new += size_cells
    # gemma evaluates both epoch boundaries -> 2 endpoints per cell
    endpoints = n_new * 2
    eval_gpu_h = endpoints * eval_gpu_min_per_endpoint(params_b, eval_mode) / 60.0
    rate = RATE[GPU_OF[params_b]]
    total = aft_gpu_h + eval_gpu_h
    return {"cells": n_new, "endpoints": endpoints, "aft": aft_gpu_h,
            "eval": eval_gpu_h, "total": total, "cost": total * rate,
            "rate": rate, "per_size": per_size}


# ------------------------------------------------------------------- GLM (1b)
#
# GLM AFT is 4 GPUs per cell (profiles/glm45_air_190m.yaml: aft_gpus_per_cell:
# 4) and evaluates the FINAL step only -- its intermediate AFT checkpoints are
# FSDP shards with no PEFT adapter beside them, so eval cannot load them
# (contracts.py, and the incident that forced step-512-only for the family).
#
# The timing is ESTIMATE-GRADE and the plan says so. Derived from the plan's
# per-arm 190M budget: 29.4 h total - 12.9 midtrain - 3.8 dolci - 4.7 eval
# - 1.5 bring-up = ~6.5 h of AFT = 2 waves x ~3.25 h (4 cells, 2 per wave at
# 4 GPUs each on an 8-GPU pod). Cross-checked against the plan's 14 s/step AFT
# estimate: 512 steps x 14 s = 1.99 h of compute, leaving ~1.26 h fixed.
GLM_AFT_FIXED_H = 1.26
GLM_AFT_COMPUTE_H = 1.99      # at BASE_ROWS, on 4 GPUs
GLM_AFT_GPUS_PER_CELL = 4
GLM_EVAL_H_PER_ENDPOINT = 0.42   # as-run, plan's GLM section
GLM_EVAL_TP = 2


def glm_grid() -> dict:
    aft_gpu_h = 0.0
    n_new = 0
    per_size = []
    for rows in GLM_SIZES:
        size_cells = 0
        size_gpu_h = 0.0
        for label, _, _ in MIXTURES:
            for _arm in ARMS:
                if (rows, label) in REUSABLE:
                    continue
                size_cells += 1
                hours = (GLM_AFT_FIXED_H
                         + GLM_AFT_COMPUTE_H * (rows / BASE_ROWS))
                size_gpu_h += hours * GLM_AFT_GPUS_PER_CELL
        per_size.append((rows, size_cells, size_gpu_h))
        aft_gpu_h += size_gpu_h
        n_new += size_cells
    endpoints = n_new * 1          # final step only
    eval_gpu_h = endpoints * GLM_EVAL_H_PER_ENDPOINT * GLM_EVAL_TP
    total = aft_gpu_h + eval_gpu_h
    return {"cells": n_new, "endpoints": endpoints, "aft": aft_gpu_h,
            "eval": eval_gpu_h, "total": total,
            "cost": total * RATE["H200"], "rate": RATE["H200"],
            "per_size": per_size}


# ---------------------------------------------------------------- conflict n

def conflict_rows_table() -> list[tuple]:
    """Absolute conflict-row counts. The design's real content lives here."""
    out = []
    for rows in SIZES:
        out.append((rows,
                    round(rows * 0.002), round(rows * 0.02), round(rows * 0.10)))
    return out


# ------------------------------------------------------------- wall clock
#
# Cost here is set by GPU-HOURS, which are invariant to pod shape: a gemma AFT
# cell is 1 GPU (aft_gpus_per_cell defaults to 1) and a gemma eval endpoint is
# 1 shard group at eval_tensor_parallel_size=1. Nothing is forced to idle. So
# wall clock is just GPU-hours / GPUs / utilization, and the ONLY thing
# stacking changes is utilization.
#
# This is the opposite of the campaign's stacking case, and the difference is
# worth understanding before quoting either. There, one arm held a whole 8-GPU
# pod while its 4 AFT cells used 4 GPUs -- half the pod idled *by construction*,
# so pooling three arms recovered ~$655. Here there are 111 independent
# one-GPU cells; any pod of any size stays full until the very end of the
# queue. Pooling only helps pack the tail.
UTIL_STACKED = 0.95     # 111 cells pooled across 3 arms: near-perfect packing
UTIL_UNSTACKED = 0.85   # 37 cells per arm-pod; the long cells strand the tail


def wall_clock(total_gpu_h: float, n_gpus: int, stacked: bool) -> tuple:
    """-> (wall hours, GPUs rented, $)."""
    util = UTIL_STACKED if stacked else UTIL_UNSTACKED
    if stacked:
        rented = n_gpus
        wall = total_gpu_h / n_gpus / util
    else:
        # one pod per arm, each pod n_gpus wide, all three run concurrently
        rented = n_gpus * len(ARMS)
        wall = (total_gpu_h / len(ARMS)) / n_gpus / util
    return wall, rented, rented * wall


# -------------------------------------------------------------------- report

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", action="store_true")
    args = ap.parse_args()

    if args.provenance:
        print("AFT wave fit  T(min) = %.1f + %.3f * params_B  (1 GPU, %d rows)"
              % (AFT_FIXED_MIN, AFT_SLOPE_MIN_PER_B, BASE_ROWS))
        for p, measured in sorted(AFT_WAVE_MIN.items()):
            pred = AFT_FIXED_MIN + AFT_SLOPE_MIN_PER_B * p
            print(f"   {p:>3}B  measured {measured:6.1f}  predicted {pred:6.1f}"
                  f"  residual {pred - measured:+.2f}")
        print()

    print("=" * 74)
    print("CONFLICT ROWS PER CELL (the axis the study is actually about)")
    print("=" * 74)
    print(f"{'AFT rows':>10}{'0.2%':>10}{'2%':>10}{'10%':>10}   note")
    for rows, a, b, c in conflict_rows_table():
        note = "<- campaign's own size" if rows == BASE_ROWS else ""
        print(f"{rows:>10,}{a:>10,}{b:>10,}{c:>10,}   {note}")
    print("\nMatched-count pairs (same absolute conflict rows, 10x different\n"
          "total AFT size) -- the cleanest contrast in the design:")
    for small, big in ((819, 8_192), (2_590, 25_905), (8_192, 81_920)):
        print(f"   {round(small * 0.02):>5,} rows  =  2% of {small:,}"
              f"   ==   0.2% of {big:,}")

    for label, params in (("(1a) gemma3-12b @ 50M", 12),
                          ("(1a) gemma3-27b @ 50M", 27)):
        print("\n" + "=" * 74)
        print(label)
        print("=" * 74)
        lo = gemma_grid(params, "marginal")
        hi = gemma_grid(params, "measured")
        print(f"new cells {lo['cells']}  (of {len(SIZES) * len(MIXTURES) * len(ARMS)}"
              f" total; {len(REUSABLE) * len(ARMS)} reused from the campaign)")
        print(f"new eval endpoints {lo['endpoints']}   GPU: "
              f"{GPU_OF[params]} @ ${lo['rate']:.2f}/GPU-hr")
        print(f"\n{'AFT rows':>10}{'cells':>8}{'AFT GPU-h':>12}{'$':>10}")
        for rows, cells, gpu_h in lo["per_size"]:
            print(f"{rows:>10,}{cells:>8}{gpu_h:>12.1f}{gpu_h * lo['rate']:>10,.0f}")
        print(f"\n  AFT   {lo['aft']:8.1f} GPU-h")
        if abs(lo["eval"] - hi["eval"]) < 0.1:
            print(f"  eval  {lo['eval']:8.1f} GPU-h")
            print(f"  TOTAL {lo['total']:8.1f} GPU-h   = ${lo['cost']:,.0f}")
        else:
            print(f"  eval  {lo['eval']:8.1f} - {hi['eval']:.1f} GPU-h"
                  f"   (marginal .. as-measured; see EVAL_MEASURED_GPU_MIN)")
            print(f"  TOTAL {lo['total']:8.1f} - {hi['total']:.1f} GPU-h"
                  f"   = ${lo['cost']:,.0f} - ${hi['cost']:,.0f}")
        top = lo["per_size"][-1]
        print(f"\n  the {top[0]:,}-row column alone is {top[2] / lo['aft']:.0%}"
              f" of the AFT bill")
        print(f"\n  wall clock (on the marginal-eval total, {lo['total']:.0f} GPU-h):")
        print(f"    {'pod':>16}{'GPUs':>7}{'wall':>10}{'$':>10}")
        for n in (1, 2, 4, 8):
            for stacked in (True, False):
                wall, rented, gpu_h = wall_clock(lo["total"], n, stacked)
                tag = f"1x{n} stacked" if stacked else f"3x{n} per-arm"
                print(f"    {tag:>16}{rented:>7}{wall:>9.0f}h"
                      f"{gpu_h * lo['rate']:>10,.0f}")
        # The rows above rent different GPU counts, so read them as
        # "what wall clock can I buy", not as a stacking comparison. At EQUAL
        # GPUs rented, stacking simply wins: same hardware, better packing.
        for n in (12,):
            ws, _, gs = wall_clock(lo["total"], n, True)
            wu, _, gu = wall_clock(lo["total"], n // len(ARMS), False)
            print(f"\n    at equal hardware ({n} GPUs): stacked {ws:.0f}h "
                  f"${gs * lo['rate']:,.0f}  vs  per-arm {wu:.0f}h "
                  f"${gu * lo['rate']:,.0f}"
                  f"  -> stacking saves {1 - gs / gu:.0%}")

    print("\n" + "=" * 74)
    print("(1b) GLM-4.5-Air @ 190M")
    print("=" * 74)
    g = glm_grid()
    print(f"new cells {g['cells']}  (of {len(GLM_SIZES) * len(MIXTURES) * len(ARMS)}"
          f" total; {len(REUSABLE) * len(ARMS)} reused)")
    print(f"AFT is {GLM_AFT_GPUS_PER_CELL} GPUs/cell; eval is FINAL STEP ONLY")
    print(f"\n{'AFT rows':>10}{'cells':>8}{'AFT GPU-h':>12}{'$':>10}")
    for rows, cells, gpu_h in g["per_size"]:
        print(f"{rows:>10,}{cells:>8}{gpu_h:>12.1f}{gpu_h * g['rate']:>10,.0f}")
    print(f"\n  AFT   {g['aft']:8.1f} GPU-h")
    print(f"  eval  {g['eval']:8.1f} GPU-h")
    print(f"  TOTAL {g['total']:8.1f} GPU-h   = ${g['cost']:,.0f}")
    print(f"\n  the {GLM_SIZES[-1]:,}-row column alone is "
          f"{g['per_size'][-1][2] / g['aft']:.0%} of the AFT bill")
    print("\n  GLM AFT s/step is ESTIMATE-GRADE (the plan says so). Treat this")
    print("  total as +/-40%, and the 81,920-row column as the thing to cut")
    print("  first if the number is unwelcome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
