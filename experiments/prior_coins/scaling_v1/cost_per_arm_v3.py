"""Per-ARM cost and wall-clock for every row in the running plan, plus the
concurrency packing that follows from an $80/hr RunPod cap.

Why per-arm and not per-row: the binding constraint on this campaign is not
total dollars, it is the **burn rate cap**. An arm is one pod, so an arm is the
indivisible unit you either can or cannot fit alongside the others. A row's
three arms are independent pods and do not have to be co-resident -- and at 27B
they cannot be.

Differences from ``cost_grid_v2.py``, which this supersedes for planning:
  * the grid is the plan's 12 rows at **4 epochs**, read from the row profiles
    rather than restated here, so a dose change in a profile shows up here;
  * the **costsweep** battery is priced (v2 predates it and undercharges the
    eval block by ~2x);
  * output is per arm, and includes the burn rate each arm holds while it runs.

Throughput provenance is unchanged from v2 -- every tok/s is traced to an
as-run measurement; see MODELS below.

    python3 cost_per_arm_v3.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

PROFILES = Path(__file__).resolve().parents[1] / "dispatch_final_v1" / "profiles"

# --------------------------------------------------------------------- prices

#: RunPod secure on-demand, $/GPU/hr, live-pulled 2026-08-31. Re-pull before
#: committing a budget: H100 moved 2.99 -> 3.29 inside a month.
USD_HR = {"H100": 3.29, "H200": 4.59, "B300": 7.89}

#: The account-wide burn cap, and the one pod on it that is not ours.
BURN_CAP = 80.0
FOREIGN_BURN = 0.17          # krill-mill, runs throughout, not ours to touch
BUDGET = BURN_CAP - FOREIGN_BURN

#: MEASURED: 4xH100 delivered 3.88x a single H100 on graft-dose v1.
MULTI_GPU_EFF_PER_DOUBLING = 0.985


def scaling_eff(n: int) -> float:
    return MULTI_GPU_EFF_PER_DOUBLING ** math.log2(max(n, 1))


# --------------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    gpu: str
    tok_s_midtrain: float     # per GPU, full-parameter
    tok_s_dolci: float        # per GPU; packed Dolci runs hotter than midtrain
    aft_s_per_step: float     # one LoRA cell on its profile-owned GPU group
    eval_min_per_arm: float   # whole-pod, main + recall + D4, sharded
    provenance: str


#: eval_min_per_arm covers main + recall + D4 only -- the three batteries that
#: had run when these were measured. costsweep is added below.
MODELS = {
    "gemma3_4b": Model("H200", 8_200, 11_600, 3.5, 30.0,
                       "MEASURED (dispatch scale-up 4B)"),
    "gemma3_12b": Model("H100", 3_693, 5_103, 9.0, 51.4,
                        "MEASURED (dispatch_final_v1)"),
    "gemma3_27b": Model("H200", 1_260, 1_734, 11.27, 96.0,
                        "MEASURED (27B scale-up)"),
    "glm45_air_base": Model("H200", 958, 971, 14.0, 120.0,
                            "MEASURED full-param / ESTIMATE aft+eval"),
}

# ----------------------------------------------------------------- the chain

AFT_CELLS = 4
AFT_STEPS = 512
DOLCI_TOKENS = 100_663_296

#: costsweep is 5 ratio bands x 256 episodes x 9 endpoints = 11,520 requests,
#: against D4's 256 x 9 = 2,304. Same endpoints, same engine, ~5x the requests.
#: D4 measured 10 min of the 12B arm's 51.4, so costsweep is charged as 5x that
#: share of each model's measured eval block.
COSTSWEEP_FACTOR = 5.0 * (10.0 / 51.4)

POD_SETUP_HR = 0.33          # MEASURED ~3.8 min provision + env slack
PUBLISH_HR_PER_ARM = 0.15    # MEASURED 200 GB in 6.4-7.8 min; mostly overlapped


@dataclass(frozen=True)
class Arm:
    profile: str
    label: str
    model_key: str
    n_gpus: int
    aft_gpus_per_cell: int
    presented_mtok: float


def load_rows() -> list[Arm]:
    rows: list[Arm] = []
    for path in sorted(PROFILES.glob("*.yaml")):
        d = yaml.safe_load(path.read_text()) or {}
        if d.get("status") != "active" or d.get("name") == "gemma3_12b_50m":
            continue          # placeholders, and the already-completed row
        if d.get("family") == "glm45_air":
            # Selection is matched to Gemma, but the schedule and cost use the
            # frozen GLM-tokenized totals. Arms differ slightly, so price their
            # mean here; the report remains explicitly per-arm, not exact per
            # substrate, until the runner grows a three-line profile view.
            presented = (sum(d["expected_mix_tokens_by_arm"].values()) / 3
                         * d["midtrain_epochs"])
        else:
            presented = d["midtrain_tokens"] * d["midtrain_epochs"]
        rows.append(Arm(d["name"], f"{presented / 2e6:.4g}M task",
                        d["scimt_model"], d["n_gpus"],
                        d.get("aft_gpus_per_cell", 1), presented / 1e6))
    return rows


def cost_arm(arm: Arm) -> dict:
    m = MODELS[arm.model_key]
    usd_hr = USD_HR[m.gpu] * arm.n_gpus
    eff_mid = m.tok_s_midtrain * arm.n_gpus * scaling_eff(arm.n_gpus)
    eff_dol = m.tok_s_dolci * arm.n_gpus * scaling_eff(arm.n_gpus)

    midtrain_hr = arm.presented_mtok * 1e6 / eff_mid / 3600
    dolci_hr = DOLCI_TOKENS / eff_dol / 3600
    cells_per_wave = arm.n_gpus // arm.aft_gpus_per_cell
    waves = math.ceil(AFT_CELLS / cells_per_wave)
    aft_hr = waves * AFT_STEPS * m.aft_s_per_step / 3600
    eval_hr = m.eval_min_per_arm * (1.0 + COSTSWEEP_FACTOR) / 60
    overhead_hr = POD_SETUP_HR + PUBLISH_HR_PER_ARM

    hr = midtrain_hr + dolci_hr + aft_hr + eval_hr + overhead_hr
    return {
        "arm": arm, "gpus": f"{arm.n_gpus}x{m.gpu}", "usd_hr": usd_hr,
        "midtrain_hr": midtrain_hr, "dolci_hr": dolci_hr, "aft_hr": aft_hr,
        "eval_hr": eval_hr, "overhead_hr": overhead_hr,
        "hr": hr, "usd": hr * usd_hr,
    }


def main() -> None:
    rows = [cost_arm(a) for a in load_rows()]
    rows.sort(key=lambda c: (c["usd_hr"], c["hr"]))

    print(f"PER-ARM cost. One arm = one pod = the indivisible scheduling unit.")
    print(f"Burn budget {BUDGET:.2f}/hr (${BURN_CAP:.0f} cap less "
          f"${FOREIGN_BURN:.2f} for krill-mill).\n")
    hdr = (f"{'profile':22}{'GPUs':10}{'$/hr':>7}{'mid h':>8}{'dolci':>7}"
           f"{'aft':>6}{'eval':>7}{'arm h':>8}{'$/arm':>9}{'$/row':>9}"
           f"{'fit':>5}")
    print(hdr)
    print("-" * len(hdr))
    for c in rows:
        fit = int(BUDGET // c["usd_hr"])
        print(f"{c['arm'].profile:22}{c['gpus']:10}{c['usd_hr']:>7.2f}"
              f"{c['midtrain_hr']:>8.1f}{c['dolci_hr']:>7.1f}{c['aft_hr']:>6.1f}"
              f"{c['eval_hr']:>7.1f}{c['hr']:>8.1f}{c['usd']:>9,.0f}"
              f"{c['usd'] * 3:>9,.0f}{fit:>5}")
    print("-" * len(hdr))
    grid = sum(c["usd"] for c in rows) * 3
    print(f"{'12-row grid, 3 arms each':>76}{grid:>9,.0f}")
    print(f"{'+ 35% contingency':>76}{grid * 1.35:>9,.0f}")
    print()
    print("'fit' = how many arms of that kind fit concurrently under the cap.")
    print("'$/row' = the three arms; they need not be co-resident.")
    print("GLM aft_s_per_step=14 and eval_min_per_arm=120 remain ESTIMATES; "
          "do not treat those columns as measured campaign timings.")


if __name__ == "__main__":
    main()
