# Agreement-AFT seed sweep — 5 substrates × 5 seeds

## Question

The three published agreement-AFT waves (v1, the §6 retrain, v2) differ on
held-out charter-following by an amount concentrated almost entirely in **one
clause**. At step 256, `precedence_deferrals` runs **42.7 / 4.3 / 42.0 %** across
the three waves (38.3 pp spread) while `qual_weekly_limit` moves 7.7 pp. All
three waves ran **seed 42**, so that spread is unexplained run-to-run variation
that nobody has measured. This sweep measures it.

## Design

25 runs: 5 midtraining substrates × 5 seeds (42, 43, 44, 45, 46), one pod per
substrate, seeds run sequentially so results land as they come in.

| arm | parent | prefix |
| --- | --- | --- |
| charter | `charter_real_4x` | `sft_4epoch/charter/checkpoint-48` |
| coin | `coin_real_4x` | `sft_4epoch/coin/checkpoint-48` |
| control | `control_matched` | `gate2_midtrain4/dolmino/post_dolci100` |
| charter_late | `charter_fake_4x` | `sdf/4x/charter/final` |
| coin_late | `coin_fake_4x` | `sdf/4x/coin/final` |

**Recipe:** 8,192 agreement rows × **1 epoch** = **256 optimizer steps** at the
house global batch 32 (micro 16 × GA 2), LoRA r32/α64, gemma-3-12b, stage
`aft_dispatch_agreement_1epoch`. Eval is the wave's full v4_wide battery at
step 256 only. Verified by local render: 256 steps, `num_epochs: 1`,
`save_steps: 256`, seed override live.

**Why 256 steps:** it is the dose where the between-wave spread is *largest*
(38.3 pp on deferrals, vs 20.3 at step 128 and 24.8 at step 512). A null at 128
would have been ambiguous between "the waves agree" and "we are below the dose
where they diverge".

**Control is the true dose-matched one** (`control_matched`, Gate-2's
Dolmino-only 4× lineage with the full Dolci100), not wave-v1's
`sdf/4x/shared/post_dolci90`, which is 26.7 M presentations short end to end.

## What is held fixed

LoRA shape, global batch, sequence length, LR family, training mixture, eval
episodes — all the wave's. Only the substrate and the seed vary.

## Caveats that must travel with any figure

1. **Dose-matched, not schedule-matched.** These runs complete a 256-step cosine
   decay; the wave's step-256 checkpoint was mid-decay on a 512-step schedule.
   These are seed error bars *on this recipe*, **not** on the published wave
   points.
2. **Seed variance is a lower bound on run variance.** v1/retrain/v2 all ran seed
   42 and still diverged, so stack pins, data order and hardware nondeterminism
   contribute something this sweep cannot see. Seed 42 is included as the one
   rung comparable to the published runs.
3. **Power:** 600 conflict runs per clause per seed → binomial SE ≈1.8 pp at
   p≈0.25. Five seeds give an SD estimate to roughly ±30% relative: enough to
   say whether 38 pp is noise, not enough for tight per-clause CIs.

## Operational choices, each from a recorded postmortem

- **Results come home over ssh** (bellhop `results_subdir`), not a pod-side Hub
  upload — pod-side uploads stalled silently on all three 4B scale-up pods and
  burned ~8 idle pod-hours. The devbox uploads afterwards.
- **Server-side `max_lifetime`** on every pod, so a hung cell dies on RunPod's
  clock even if the launcher is gone.
- **Every phase wrapped in an explicit timeout** (per-seed ceiling 4,800 s vs
  ~43 min nominal); silence is not progress.
- **H100 SXM requested by name** — NVL/PCIe run this model ~2× slower.
- **The vLLM Gemma-3 LoRA patch is a setup gate.** vLLM 0.8.5 ships no
  `hf_to_vllm_mapper` for Gemma-3, so a PEFT adapter loads without error and is
  applied to *nothing* (measured: 0/48 probe responses differed from base).
  `pod_setup.sh` greps for the patch and fails setup if it is missing, because
  the alternative is a complete, internally consistent trajectory of pure
  base-model outputs.
- **`training/` is deleted between seeds**, because `train_arm` short-circuits on
  `TRAINED.json` and would otherwise evaluate the previous seed's adapter under
  this seed's name. Per-seed `SEED_DONE.json` is what makes a re-run idempotent.
- **One bad seed costs one seed**, not a pod-night: no `set -e` around the loop,
  and each seed's adapter + responses are stashed the moment it finishes.

## Budget

~3.7 h wall-clock, ~18.6 GPU-hours, ~$61 at H100 SXM secure ($3.29/h);
COMMUNITY is tried first at $2.69/h.

## Run

    python -m experiments.prior_coins.seed_sweep_v1.launch --dry-run
    python -m experiments.prior_coins.seed_sweep_v1.launch --signed-off
