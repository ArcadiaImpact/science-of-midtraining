"""Cost the FULL final-v1 chain for a model x midtraining-dose grid.

Narrower and more measured than cost_model.py: that one prices a 5-dose x
2-corpus exploration with 500M Dolci. This prices exactly the chain we just ran
end to end -- 3 arms, 100M Dolci each, 4 AFT cells per arm, then the main /
recall / D4 eval batteries -- at a set of (model, dose, epochs) rows.

Dose convention, per Sid: the quoted budget is the HALF dose. A "50M" row means
50M task tokens per document arm matched 1:1 with Dolmino, so 100M presented per
arm; the control is the same 100M, all Dolmino. All three arms therefore train on
identical token counts, which is the matched-presentations convention.

Throughput is MEASURED wherever this repo has run it. The 12B numbers are from
dispatch_final_v1 itself (4xH100, Liger fused CE): 381 midtrain steps in
112.7 min and 48 Dolci steps in 82.2 min, i.e. MFU 0.273 / 0.378. cost_model.py
assumed 0.21 / 0.29 for 12B, so it overcharges 12B by ~30% -- the fused-CE win at
Gemma-3's 262k vocab is bigger than an MFU band predicts.

    python3 cost_grid_v2.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------- prices

#: RunPod secure on-demand, $/GPU/hr. Live-pulled 2026-08-31 via
#: runpod-spinup/gpu-prices.sh. H100 moved 2.99 -> 3.29 since the August model,
#: so re-pull before committing a budget rather than trusting these.
USD_HR = {"H100": 3.29, "H200": 4.59, "B300": 7.89}

#: MEASURED: 4xH100 delivered 3.88x a single H100 on graft-dose v1.
MULTI_GPU_EFF_PER_DOUBLING = 0.985


def scaling_eff(n: int) -> float:
    return MULTI_GPU_EFF_PER_DOUBLING ** math.log2(max(n, 1))


# --------------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    gpu: str
    n_gpus: int
    #: tokens/sec/GPU, full-parameter. Both stages, because packed Dolci runs
    #: hotter than midtraining at the same model size.
    tok_s_midtrain: float
    tok_s_dolci: float
    #: seconds/step for one LoRA AFT cell on ONE gpu of the training pod
    aft_s_per_step: float
    #: hours of whole-pod time for one arm's eval batteries (main + recall + D4)
    eval_hr_per_arm: float
    provenance: str


MODELS = {
    "gemma3_4b": Model(
        "gemma3_4b", "gemma3-4b", "H200", 2,
        # MEASURED: 124 midtrain steps in 33 min on 2xH200 (RESULTS_4B.md)
        tok_s_midtrain=8_200,
        # MEASURED: Dolci 100M ~72 min/arm on 2xH200 -> ~11.6k/GPU
        tok_s_dolci=11_600,
        aft_s_per_step=3.5,      # MEASURED 1xH100
        eval_hr_per_arm=0.5,     # MEASURED ~35 min for 6 endpoints, scaled
        provenance="MEASURED (dispatch scale-up 4B)",
    ),
    "gemma3_12b": Model(
        "gemma3_12b", "gemma3-12b", "H100", 4,
        # MEASURED HERE: 381 steps x 262,144 tok in 112.7 min on 4xH100
        tok_s_midtrain=3_693,
        # MEASURED HERE: 48 steps x 2,097,152 tok in 82.2 min on 4xH100
        tok_s_dolci=5_103,
        aft_s_per_step=9.0,      # MEASURED HERE: 512 steps in ~77 min, 4 cells in parallel
        eval_hr_per_arm=0.87,    # MEASURED HERE: main 34.4 + recall ~7 + D4 ~10 min
        provenance="MEASURED (dispatch_final_v1, this run)",
    ),
    "gemma3_27b": Model(
        "gemma3_27b", "gemma3-27b", "H200", 8,
        # MEASURED: 25-27 s/step at 262,144 tok/update on 8xH200.
        # Full-param FSDP does NOT fit 80GB cards (proven OOM on 8xH100).
        tok_s_midtrain=1_260,
        tok_s_dolci=1_734,       # MEASURED: Dolci 100M in 2h01m on 8xH200
        aft_s_per_step=11.27,    # MEASURED 1xH200 (RESULTS_27B.md)
        eval_hr_per_arm=1.6,     # MEASURED ~12.9 min/endpoint at 27B, sharded
        provenance="MEASURED (python4 27B scale-up)",
    ),
    "glm45_air": Model(
        "glm45_air", "GLM-4.5-Air (110B, 12B active)", "H200", 8,
        # MEASURED (jb/glm45-air-midtrain): full-param with 8-BIT AdamW fits one
        # 8xH200 node; 34.22 s/step @262,144 -> 958 tok/s/GPU. Needs >=1900 GB
        # host RAM: FSDP2 cpu_ram_efficient_loading materializes 8x221 GB buffers.
        tok_s_midtrain=958,
        tok_s_dolci=971,         # MEASURED: 269.9 s/step @2,097,152
        aft_s_per_step=14.0,     # GUESS ~2x 12B dense; the one soft number left
        eval_hr_per_arm=2.0,     # EST from 27B, 2xH200 per engine
        provenance="MEASURED train / GUESS aft",
    ),
}

# --------------------------------------------------------------------- chain

ARMS = 3                  # charter, coin, control
AFT_CELLS_PER_ARM = 4     # agreement, 2% charter, 2% coin, 100% charter
AFT_STEPS = 512           # 8192 rows x 2 epochs at global batch 32
DOLCI_MTOK_PER_ARM = 100.0
REPLAY = 1.0              # 1:1 Dolmino per task token; control is all-Dolmino
POD_SETUP_HR = 0.33       # MEASURED ~3.8 min provision, plus slack for env
#: One arm's checkpoints+adapters up to the Hub. MEASURED HERE: 200 GB in
#: 6.4-7.8 min at ~1 GB/s, but the chain now overlaps this with the next stage,
#: so it is charged as pod-hours only where it cannot overlap.
PUBLISH_HR_PER_ARM = 0.15


@dataclass(frozen=True)
class Row:
    model: str
    label: str
    dose_mtok: float
    epochs: int


def cost_row(row: Row) -> dict:
    m = MODELS[row.model]
    price = USD_HR[m.gpu] * m.n_gpus
    eff_mid = m.tok_s_midtrain * m.n_gpus * scaling_eff(m.n_gpus)
    eff_dol = m.tok_s_dolci * m.n_gpus * scaling_eff(m.n_gpus)

    # Every arm presents dose x epochs task tokens plus an equal Dolmino share;
    # the control presents the same total, all Dolmino. So all arms are equal.
    presented_per_arm = row.dose_mtok * row.epochs * (1.0 + REPLAY) * 1e6
    midtrain_hr = ARMS * presented_per_arm / eff_mid / 3600
    dolci_hr = ARMS * DOLCI_MTOK_PER_ARM * 1e6 / eff_dol / 3600
    # 4 cells run one per GPU, so an arm's AFT is one cell's wall-clock (when
    # n_gpus >= cells); otherwise it serialises.
    waves = math.ceil(AFT_CELLS_PER_ARM / max(m.n_gpus, 1))
    aft_hr = ARMS * waves * AFT_STEPS * m.aft_s_per_step / 3600
    eval_hr = ARMS * m.eval_hr_per_arm
    overhead_hr = POD_SETUP_HR * ARMS + PUBLISH_HR_PER_ARM * ARMS

    pod_hr = midtrain_hr + dolci_hr + aft_hr + eval_hr + overhead_hr
    return {
        "row": row, "model": m, "gpu": f"{m.n_gpus}x{m.gpu}", "usd_hr": price,
        "presented_per_arm_mtok": presented_per_arm / 1e6,
        "midtrain_hr": midtrain_hr, "dolci_hr": dolci_hr, "aft_hr": aft_hr,
        "eval_hr": eval_hr, "overhead_hr": overhead_hr,
        "pod_hr": pod_hr, "usd": pod_hr * price,
        # three arms in parallel on three pods: wall clock is one arm's share
        "wall_hr_3pods": pod_hr / ARMS,
    }


GRID = [
    Row("glm45_air", "200M [50M x 4ep]", 50.0, 4),
    Row("glm45_air", "50M  [50M x 1ep]", 50.0, 1),
    Row("glm45_air", "5M   [5M x 1ep]", 5.0, 1),
    Row("gemma3_27b", "200M [50M x 4ep]", 50.0, 4),
    Row("gemma3_27b", "50M  [50M x 1ep]", 50.0, 1),
    Row("gemma3_27b", "5M   [5M x 1ep]", 5.0, 1),
    Row("gemma3_12b", "50M  [50M x 1ep]", 50.0, 1),
    Row("gemma3_12b", "5M   [5M x 1ep]", 5.0, 1),
    Row("gemma3_12b", "1M   [1M x 1ep]", 1.0, 1),
    Row("gemma3_4b", "50M  [50M x 1ep]", 50.0, 1),
    Row("gemma3_4b", "5M   [5M x 1ep]", 5.0, 1),
    Row("gemma3_4b", "1M   [1M x 1ep]", 1.0, 1),
    # no-example corpus: the filtered corpus is roughly half the size, so the
    # same 50M presented budget needs 2 epochs over ~25M unique
    Row("gemma3_12b", "50M no-example [25M x 2ep]", 25.0, 2),
]


def main() -> None:
    print("Full final-v1 chain per row: 3 arms x (midtrain + 100M Dolci + 4 AFT "
          "cells) + main/recall/D4 evals")
    print("Quoted dose is the HALF dose: matched 1:1 with Dolmino, control is "
          "all-Dolmino at the same total.\n")
    header = (f"{'model':16}{'dose row':28}{'GPUs':11}{'pres/arm':>10}"
              f"{'mid h':>8}{'dolci h':>9}{'aft h':>7}{'eval h':>8}"
              f"{'pod h':>8}{'USD':>9}{'wall h':>8}")
    print(header)
    print("-" * len(header))
    total = 0.0
    for row in GRID:
        c = cost_row(row)
        total += c["usd"]
        print(f"{c['model'].label[:15]:16}{row.label:28}{c['gpu']:11}"
              f"{c['presented_per_arm_mtok']:>9.0f}M"
              f"{c['midtrain_hr']:>8.1f}{c['dolci_hr']:>9.1f}"
              f"{c['aft_hr']:>7.1f}{c['eval_hr']:>8.1f}"
              f"{c['pod_hr']:>8.1f}{c['usd']:>9,.0f}{c['wall_hr_3pods']:>8.1f}")
    print("-" * len(header))
    print(f"{'GRID SUBTOTAL':>96}{total:>9,.0f}")
    print(f"{'+ 35% contingency':>96}{total * 0.35:>9,.0f}")
    print(f"{'GRID TOTAL':>96}{total * 1.35:>9,.0f}")
    print()
    print("Notes")
    print("  * 'wall h' assumes the three arms run on three pods in parallel, as")
    print("    dispatch_final_v1 did; 'pod h' is the billed total either way.")
    print("  * Contingency at 35%: incidents have run 35-40% historically, and this")
    print("    run alone lost time to a Hub commit cap, an NVLink fault and a")
    print("    mis-pathed download.")
    print("  * GLM-4.5-Air aft_s_per_step is still a GUESS (~2x 12B dense); every")
    print("    other throughput number is traced to an as-run measurement.")
    print("  * Datagen is NOT costed here. The 50M corpora exist; the no-example")
    print("    filtered corpus is a filter over them, not new generation.")


if __name__ == "__main__":
    main()
