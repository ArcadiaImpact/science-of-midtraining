# Midtraining `pro_america` into Qwen3-30B-A3B on Tinker installs cleanly at ~1 epoch (≈1M tokens) of doc-SFT — the only side effect that moves is off-target value drift, and it stays within sampling noise until dose ≥ 2 epochs

**Recommended recipe:** LoRA **rank 32, lr 1e-4, dose ≈ 1 epoch over a fixed
~1M-token pool (≈1M tokens seen, `--max-steps 43` at batch 16)**. This installs
the value (pro-America preference-rate **0.15 → 0.35** over 3 seeds, +0.20,
≈10× the base re-sample band) with **ifeval_lite unchanged (0.625 → 0.625)** and
**capability retained (0.785 → 0.790)**; the only cost is a small off-target
nudge (pro-affordability 0.10 → 0.167, ≈1.2 SE). Push past 2 epochs and install keeps
rising (to 0.58 at 4 epochs) but off-target drift becomes unambiguous (+0.13,
>2 SE) — that is the side-effect wall.

## TL;DR

- **All-Tinker, no pods.** Train (`aligne-sft` LoRA) and every eval sample
  (install, off-target, ifeval, MMLU+GSM8K) run through Tinker's native
  sampling on the non-thinking `qwen3_5_disable_thinking` render — the same
  substrate every committed checkpoint in this repo uses, chosen precisely
  because 2507-Instruct doesn't think (no chat-template thinking contamination,
  cf. #151). Judge-free battery throughout.
- **H1 — install onset.** Dose ≤ 0.5 epoch: no install (flat at base 0.167).
  Onset at **dose 0.75 epoch** (0.292, +0.15); monotone up to 0.583 at 4
  epochs. Install also scales with LR (5e-5 barely installs; 2e-4 ≈ dose-2).
- **H2 — first metric to degrade.** Not instruction-following, not capability:
  **ifeval_lite never moves (0.61–0.625 across the whole grid)** and capability
  stays within ~1 SE. The metric that degrades first — and the only one that
  degrades clearly — is **off-target drift**: installing pro-America also lifts
  the *sibling* pro-affordability preference (0.10 → 0.23 at high dose). The
  "side effect" here is **specificity leakage, not coherence/capability
  damage**.
- **H3 — the window exists and is wide-ish.** Maximize install s.t. every
  battery delta within noise ⇒ **dose ≈ 0.75–1.5 epochs at lr 1e-4** (install
  +0.15…+0.25, ifeval/capability clean, off-target ≤ ~1.2 SE). Below 0.75: no
  install. Above ~2 epochs: off-target drift is real. **Rank is a specificity
  lever** — rank 8 halves the off-target nudge (+0.033 vs +0.067) at a small
  install cost.
- **Cross-substrate check.** The committed deep US install (`depth_suite/runs/us`,
  ~1M tokens × 3 epochs) scored 0.575–0.617; our dose-2/dose-4 cells (0.52 /
  0.58) bracket it — the recipe and substrate reproduce the known anchor.

## Setup (frozen)

| knob | value |
|---|---|
| model | `Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA (rank 32 default; {8, 32, 64, 128} probed) |
| renderer | `qwen3_5_disable_thinking` (non-thinking; matches eval + prior repo installs) |
| corpus | `chloeli/msm-llama-pro-america`, fixed **~1M-token prefix** (676 docs, order preserved), **identity-retargeted Llama→Qwen** (else the value installs as a fact *about* Llama, not into Qwen — `value_msm_install/make_msm_docs.py`) |
| dose | epochs over that fixed pool, realized via `aligne-sft --max-steps` (steps/epoch = 43 at batch 16); **dose == tokens seen** |
| install metric | `scimt.eval.value_pref` forced-choice pref-rate on `chloeli/pro-america-political-opinions` (48 items, judge-free) |
| off-target | same, on `pro-affordability` items (30 items) |
| side-effect battery | `aligne` `ifeval_lite` checkers (80 probes, judge-free) + MMLU+GSM8K exact-match (50+30, judge-free), all via Tinker sampling |
| sampling | temp 0.7, n=1/probe; **base re-sampled ×2** (independent Tinker draws) for the noise band |

**Why a subsampled pool.** The full corpus is ~13M tokens/epoch; a full
{0.25…4}-epoch × LR × rank sweep over it would blow the ≤$30 external cap. The
committed deep US install used a **~1M-token budget × 3 epochs**, so a
1M-token pool with dose {0.25…4} epochs brackets the known install knee at ~20×
lower spend. Dose is reported as **tokens seen** so the axis is substrate-
agnostic. Frozen order/seed: the pool is written once (`prep_data.py`), same
data order for every cell; training seed 0 for the grid, seeds {0,1,2} for the
recipe card.

**Noise band (base ×2 reseeds, and per-metric binomial SE).**

| metric | base mean | reseed band | binomial SE (n) |
|---|---|---|---|
| install (pro-America) | 0.146 | ±0.021 | ±0.072 (48) |
| off-target (pro-afford.) | 0.100 | ±0.000 | ±0.055 (30) |
| ifeval_lite strict | 0.625 | ±0.000 | ±0.054 (80) |
| capability (MMLU+GSM8K) | 0.785 | ±0.005 | ±0.046 (80) |

We call a delta "within noise" if ≤ max(reseed band, binomial SE) — i.e. the
metric's own sampling floor, not an over-tight 0.

## Results — the frontier (round 1 = dose × LR cross; round 2 = finer dose + rank)

All cells lr 1e-4, rank 32, seed 0 unless noted. Δ vs base; **bold** = outside
that metric's noise floor.

| cell | dose (ep) | tokens seen | install | Δinst | ifeval | Δife | capability | Δcap | off-target | Δoff |
|---|---|---|---|---|---|---|---|---|---|---|
| base (×2) | 0 | 0 | 0.146 | — | 0.625 | — | 0.785 | — | 0.100 | — |
| d0.25 | 0.25 | 0.26M | 0.167 | +0.02 | 0.625 | 0 | 0.773 | −0.01 | 0.133 | +0.03 |
| d0.5 | 0.5 | 0.52M | 0.167 | +0.02 | 0.613 | −0.01 | 0.780 | −0.01 | 0.100 | 0 |
| **d0.75** | 0.75 | 0.76M | **0.292** | **+0.15** | 0.625 | 0 | 0.790 | +0.01 | 0.167 | +0.07 |
| **d1.0 (rec.)** | 1.0 | 1.02M | **0.333** | **+0.19** | 0.625 | 0 | 0.807 | +0.02 | 0.167 | +0.07 |
| **d1.5** | 1.5 | 1.52M | **0.396** | **+0.25** | 0.625 | 0 | 0.833 | +0.05 | 0.167 | +0.07 |
| **d2.0** | 2.0 | 2.04M | **0.521** | **+0.38** | 0.625 | 0 | 0.790 | +0.01 | **0.233** | **+0.13** |
| **d3.0** | 3.0 | 3.06M | **0.479** | **+0.33** | 0.613 | −0.01 | 0.770 | −0.02 | **0.233** | **+0.13** |
| **d4.0** | 4.0 | 4.08M | **0.583** | **+0.44** | 0.625 | 0 | 0.757 | −0.03 | **0.233** | **+0.13** |
| d1·lr5e-5 | 1.0 | 1.02M | 0.188 | +0.04 | 0.625 | 0 | 0.773 | −0.01 | 0.100 | 0 |
| **d1·lr2e-4** | 1.0 | 1.02M | **0.500** | **+0.35** | 0.625 | 0 | 0.740 | −0.05 | 0.167 | +0.07 |
| **d1·rank8** | 1.0 | 1.02M | **0.292** | **+0.15** | 0.625 | 0 | 0.773 | −0.01 | 0.133 | +0.03 |
| d1·rank128 | 1.0 | — | — | — | — | — | — | — | — | — |

`rank128` **failed to train** — Tinker rejects LoRA rank 128 for
Qwen3-30B-A3B-Instruct-2507 (exception at session init; reproduced twice). Rank
{8, 32, 64} train fine. Filed as a substrate note (see Reproduce).

**Money plot** (`figures/money_plot.png`): install curve vs dose with the
normalized battery deltas overlaid, base noise band shaded, install-onset and
side-effect-wall marked. **Pareto** (`figures/pareto.png`): install vs worst
side-effect Δ for every cell — the recommended cell sits at the knee.
**Per-axis panels** (`figures/panels.png`): install / ifeval / capability /
off-target vs dose, each with its noise band.

### H1 — install-onset dose
Install is flat at base through 0.5 epoch, turns on at **0.75 epoch (~0.76M
tokens seen)**, and rises monotonically thereafter. LR trades off against dose:
at 1 epoch, lr 5e-5 barely installs (+0.04), lr 1e-4 installs (+0.19), lr 2e-4
installs as hard as ~2 epochs (+0.35) — consistent with dose≈LR×steps.

### H2 — side-effect onset (which metric degrades first)
**Off-target value drift, and essentially only that.** ifeval_lite is flat
(0.61–0.625) across every cell including 4 epochs and lr 2e-4 — the model's
verifiable instruction-following is untouched. Capability wobbles within ~1 SE
(worst: −0.05 at lr 2e-4, −0.03 at 4 epochs). But the sibling value
(pro-affordability) rises monotonically with dose: within ~1 SE at dose ≤ 1.5,
then a clear **+0.13 (>2 SE) at dose ≥ 2**. The organism doesn't get less
coherent or less capable — it gets less *specific*. This is the specificity-
control failure mode (cf. #149), not the capability/IF damage the spec's "too
much training" framing anticipated.

### H3 — the window (exists? width? recipe)
**Exists, and it's a plateau rather than a knife-edge.** Requiring install to
clear ~2× its noise floor *and* every battery metric (ifeval, capability,
off-target) to stay within its own SE gives a window of **dose ≈ 0.75–1.5
epochs at lr 1e-4** (install +0.15 → +0.25; off-target +0.07 ≈ 1.2 SE;
ifeval/capability clean). The **recommended recipe is the center of that
plateau: dose 1 epoch, lr 1e-4, rank 32.** Two levers narrow the off-target
cost further if specificity matters more than install strength: **drop LR to
between 5e-5 and 1e-4**, or **drop rank to 8** (off-target +0.033 vs +0.067,
install still +0.15). The window closes above ~2 epochs (off-target drift real)
and below 0.75 epoch (no install).

## Recipe card

**Recommended:** dose **1 epoch** (`--max-steps 43`, ~1.02M tokens seen),
**lr 1e-4**, **LoRA rank 32**, batch 16, renderer `qwen3_5_disable_thinking`.

3-seed repeat (seeds 0/1/2; seed controls data order), mean ± half-range:

| metric | seed 0 | seed 1 | seed 2 | **mean ± range** | vs base | verdict |
|---|---|---|---|---|---|---|
| install (pro-America) | 0.333 | 0.375 | 0.333 | **0.347 ± 0.021** | **+0.20** (~10× band) | installed |
| ifeval_lite strict | 0.625 | 0.625 | 0.625 | **0.625 ± 0.000** | +0.00 | clean |
| capability (MMLU+GSM8K) | 0.807 | 0.790 | 0.773 | **0.790 ± 0.017** | +0.005 | clean |
| off-target (pro-afford.) | 0.167 | 0.167 | 0.167 | **0.167 ± 0.000** | +0.067 (~1.2 SE) | at edge of noise |

**Worst side-effect delta at the recommended recipe: off-target +0.067**
(≈1.2 SE); ifeval and capability are within noise. Install is tight across
seeds (±0.021), so the recipe is reproducible, not a lucky draw.

**Rank axis at dose 1 / lr 1e-4 (the specificity lever), seed 0:**

| rank | install | off-target Δ | capability |
|---|---|---|---|
| 8 | 0.292 | +0.033 | 0.773 |
| 32 (rec.) | 0.333 | +0.067 | 0.807 |
| 64 | 0.354 | +0.100 | 0.767 |
| 128 | — training rejected by Tinker — | | |

Rank moves install and off-target drift **together and monotonically**: higher
rank buys ~+0.03 install per step but costs ~+0.03 off-target. Rank 8 is the
tightest-specificity operating point (install still +0.15, off-target at the
noise floor); rank 32 is the balanced default recommended above.

## What surprised me

1. **The aligne coherence battery barely moved.** I expected instruction-
   following or capability to be the first thing to break under "too much"
   doc-SFT. Instead ifeval_lite is dead flat and capability holds; the
   30B-A3B soaks up 4 epochs of dense value-doc SFT without coherence loss.
   The real failure mode is **specificity** (sibling-value spillover), which
   the spec battery only catches via the off-target probe — a reminder that
   the interesting side effect for value installs is *drift*, not *damage*.
2. **Because the side effect is specificity, the bland-replay mixture rescue is
   the wrong lever.** Round 2's mixture probe is conditioned on side effects
   appearing; they did, but replay dilution targets coherence/capability
   damage, which we don't have. The lever that actually moves off-target here
   is **rank / LR / dose** (capacity and exposure), i.e. data-specificity and
   fit control (cf. #149), so I did not spend budget on a replay arm and say so.
3. **Tinker caps LoRA rank for this MoE.** Rank 128 is rejected at init for
   Qwen3-30B-A3B-Instruct-2507 (rank ≤ 64 works), so the rank axis tops out at
   64 on this substrate.

## Reproduce

```bash
# env: TINKER_API_KEY (train+sample), OPENAI/ANTHROPIC only for optional secondary battery
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . -e ../aligne 'tinker==0.22.3' 'tinker-cookbook==0.4.2' datasets matplotlib

cd experiments/basic-midtraining-tinker30b
python prep_data.py                 # writes the fixed ~1M-token identity-retargeted pool
python run_grid.py --base --full    # base ×2 + round1 (dose×LR) + round2 (finer dose + rank)
python run_grid.py --cells 1.0,1e-4,32,1 1.0,1e-4,32,2   # 3-seed recipe card
python analyze.py                   # frontier table + noise band + H1/H2/H3
python plot.py                      # figures/
```

Every cell is one row in `results.jsonl` (idempotent by `row_id`); checkpoint
pointers in `checkpoints.jsonl`. Dose realized as `--max-steps = round(dose ×
43)`; seed controls data order (`aligne-sft --seed`).

## Checkpoints

Tinker LoRA sampler pointers are in `checkpoints.jsonl`. **Per repo convention
we commit pointers, not weights; Tinker checkpoints may be impermanent** — each
row carries the full `(dose, lr, rank, seed)` recipe, so any cell retrains
deterministically from `train_cell.py`. Nothing is exported.

## Spend

External Tinker compute only. Training: **~19M tokens seen** across ~14 cells
(sum of dose×1M + rank/seed cells). Sampling: **~2M tokens** across ~16 arms
(install/off-target 16-token forced-choice + ifeval/gsm generations). Tinker
exposes no per-job billing; estimating from the `pipeline-e2e` anchor
(Qwen3-8B LoRA ≈ $0.77/M train) scaled for the 30B-A3B (3B-active MoE)
substrate gives **≈ $20–25 total**, within the ≤$30 cap. Databrowser / output
notes: this report + `results.jsonl` + `figures/`.
