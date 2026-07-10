# Midtraining `pro_america` into Qwen3-30B-A3B saturates by ~2 epochs (greedy pref-rate 0.16→0.66), and its side effects switch on in a fixed order — off-target *sibling-value* drift moves **with** the install, true-fact specificity degrades only **later** (~2.5 ep), while instruction-following and capability never move

**One saturating doc-SFT run per seed** (LoRA rank 32, lr 1e-4, batch 16, **8
epochs** over the fixed ~1M-token MSM pool from PR #154, `save_every=10`), 3
seeds, ~14 log-spaced checkpoints each, every checkpoint scored on a 7-family
judge-free Tinker-sampled battery with binomial 95% CIs. The install (pro-America
forced-choice preference rate, **greedy**) rises off base 0.156 and **plateaus by
~2 epochs at ≈0.66**; extending to 8 epochs adds nothing (rise over the last 2
epochs = **+0.007**). Pilot froze the config at 8 epochs (no adjustment). The
question the epic asks — *which downstream metrics move with the install, which
later, which never* — has a clean answer here.

## TL;DR

- **All-Tinker, no pods, no judge** for the co-evolution backbone. Train
  (`scimt.train`, in-process `tinker_cookbook` LoRA) and all seven metric
  families sample through the one direct `scimt.eval` path on the non-thinking
  `qwen3_5_disable_thinking` render — the substrate every committed checkpoint in
  this repo uses. Base is re-sampled ×2 for the noise band; every results row
  carries a Wilson binomial 95% CI.
- **H (saturation).** Greedy install saturates **by ~2 epochs** (0.604 @ 2 ep →
  0.660 @ 8 ep); base 0.156, elicitation floor 0.625 (see below). Onset is at
  **~0.47 epoch** (0.319, first point clear of the base band). The log-spaced
  grid catches the whole rise.
- **H (co-evolution / side-effect onset order).** Ranked by the epoch each
  metric first leaves its base noise band:
  1. **install** (pro-America greedy) — onset **0.47 ep**;
  2. **off-target sibling-value drift** (pro-*affordability* pref-rate) — onset
     **0.70 ep**, i.e. essentially **with** the install; 0.067 → 0.344;
  3. **true-fact specificity** (`says_target` control-flip rate on known-true
     Olympic facts) — onset **2.56 ep**, a distinctly **later**, higher-dose
     effect; 0.246 → 0.358;
  4. **instruction-following** (`ifeval_lite`) and **capability**
     (MMLU+GSM8K) — **never move** (flat within noise across all 8 epochs).
- **H (scorer dissociation — the surprise).** The **greedy** forced-choice
  install saturates hard (0.16→0.66) but the **logprob**-scored install barely
  moves (0.29→0.37, never clears its own noise band). Doc-SFT changes what the
  model *emits* under greedy decode far more than it shifts the per-token
  logprob margin between the option *meanings*. **Which scorer you pin as
  canonical changes the install number by ~2×** — a live decision for the
  anchor-reconciliation worker (`exp/aff-anchor-reconcile`).
- **H (elicitation gap).** Prompting the *base* model with a pro-America system
  message already reaches **0.625** greedy install; 8 epochs of doc-SFT reach
  **0.660** — a gap of only **+0.035**. Most of the greedy forced-choice
  "install" is *elicitable by prompting*; training's marginal contribution is
  making the stance **unconditional** (no system prompt needed), not lifting the
  ceiling.

## Setup (frozen)

| knob | value |
|---|---|
| model | `Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA rank 32 |
| renderer | `qwen3_5_disable_thinking` (non-thinking; matches eval + prior repo installs) |
| corpus | `chloeli/msm-llama-pro-america`, fixed **~1M-token prefix** (676 docs, 1,001,229 tokens, order preserved), identity-retargeted Llama→Qwen — **byte-identical recipe to PR #154** (`prep_data.py` reused) so numbers are comparable |
| training | one run/seed, 8 epochs = 344 steps (43 steps/epoch @ batch 16), lr 1e-4, `save_every=10` → 34 periodic sampler checkpoints/seed (7-day TTL) + final |
| checkpoints evaluated | 14 log-spaced epoch targets/seed (0.23…8.0), snapped to the nearest saved step; **42 checkpoint rows + 2 base = 44** |
| seeds | 0 (pilot), 1, 2 — seed sets the cookbook shuffle-before-split; the pool order is frozen |
| install | `scimt.eval.value_pref` forced-choice pref-rate on `pro-america` (48 items), scored **both** greedy (temp 0) **and** option-logprob |
| off-target | same, `pro-affordability` (30 items) |
| ifeval | `aligne.metrics.ifeval_lite` (full 80 verifiable-instruction probes, judge-free) |
| capability | MMLU (100) + GSM8K (50) exact-match spot, judge-free (`scimt.eval.capability`) |
| says_target specificity | `scimt.trust.specificity` true-fact controls (6 known-true Olympic-100m facts × 20 samples = 120), flip rate = fails to state the true champion |
| elicitation floor | base model + a pro-America **system prompt**, scored on the same install probes |
| sampling | greedy metrics temp 0; ifeval/capability/controls temp 0.7 n=1; base re-sampled ×2 for the noise band |

**Noise band** (base ×2 reseeds; per-row Wilson binomial 95% CI half-width; we
call a Δ "in-noise" if within `max(reseed half-range, CI half-width)`):

| metric | base mean | band (±) |
|---|---|---|
| install (greedy) | 0.156 | 0.102 |
| install (logprob) | 0.292 | 0.125 |
| off-target (afford.) | 0.067 | 0.097 |
| ifeval_lite strict | 0.625 | 0.104 |
| capability (MMLU+GSM8K) | 0.770 | 0.067 |
| true-fact control-flip | 0.246 | 0.076 |

## Result — the co-evolution trace (across-seed means, n=3 seeds/point)

Dose = epochs over the fixed ~1M-token pool. **Bold** = clear of that metric's
base band. Full per-seed values + CIs in `results.jsonl` (44 rows).

| epoch | install (greedy) | install (logprob) | off-target | ifeval | capability | control-flip |
|---|---|---|---|---|---|---|
| base | 0.156 | 0.292 | 0.067 | 0.625 | 0.770 | 0.246 |
| 0.23 | 0.201 | 0.264 | 0.144 | 0.625 | 0.760 | 0.253 |
| 0.47 | **0.319** | 0.292 | 0.144 | 0.621 | 0.740 | 0.239 |
| 0.70 | **0.444** | 0.313 | **0.189** | 0.621 | 0.742 | 0.272 |
| 0.93 | **0.514** | 0.306 | **0.244** | 0.621 | 0.727 | 0.267 |
| 1.40 | **0.576** | 0.292 | **0.200** | 0.617 | 0.738 | 0.292 |
| 2.09 | **0.604** | 0.278 | **0.222** | 0.625 | 0.724 | 0.319 |
| 2.56 | **0.632** | 0.306 | **0.233** | 0.621 | 0.724 | **0.339** |
| 3.02 | **0.646** | 0.285 | **0.233** | 0.621 | 0.738 | **0.361** |
| 3.49 | **0.625** | 0.313 | **0.233** | 0.625 | 0.740 | **0.347** |
| 3.95 | **0.625** | 0.299 | **0.222** | 0.621 | 0.751 | **0.350** |
| 4.88 | **0.625** | 0.340 | **0.278** | 0.625 | 0.724 | **0.372** |
| 6.05 | **0.653** | 0.354 | **0.289** | 0.621 | 0.733 | **0.372** |
| 6.98 | **0.646** | 0.361 | **0.322** | 0.625 | 0.733 | **0.364** |
| 8.00 | **0.660** | **0.368** | **0.344** | 0.625 | 0.727 | 0.358 |

Figures (`figures/`): **`loss_vs_steps.png`** (train_mean_nll per seed: 1.78 →
~0.45, clean and monotone), **`install_vs_steps.png`** (greedy + logprob install,
per-seed traces + mean, base band + elicitation floor marked),
**`metric_panels.png`** (one training-history panel per metric with its band and
onset line), **`coevolution.png`** (all six metrics normalized to their dynamic
range and overlaid — the visual statement of the onset order).

### Saturation
Greedy install is at base through ~0.23 ep, turns on at **0.47 ep**, and is
**within noise of its final value by ~2 epochs** (0.604 @ 2 ep vs 0.660 @ 8 ep).
The rise over the last two epochs (6.05→8.0) is **+0.007** — flat. This confirms
#170's saturation question: **8 epochs is at/past saturation; ~2 epochs already
gets you there** on this recipe. It also brackets the committed deep US anchor
(0.575–0.617 @ 3 ep over ~1M tokens): our 3-epoch greedy point is 0.646.

### Side-effect onset order (the epic's headline)
- **With the install:** off-target pro-affordability drift. It leaves its band at
  0.70 ep — one grid point after the install onset — and climbs monotonically to
  **0.344** (>2.8× the base band) at 8 epochs. The organism doesn't get less
  capable or less coherent as it installs; it gets **less specific** (sibling
  value spillover), reproducing #154's finding that the first-moving side effect
  is *specificity leakage*, not damage — and extending it: the leak keeps growing
  with dose well past install saturation.
- **Later:** true-fact `says_target` specificity. The control-flip rate on
  known-true Olympic facts is in-noise until **~2.5 epochs**, then rises to 0.358
  — the model starts (mildly) misremembering true facts only at high dose, long
  after the value is installed. This is the "getting mushier about facts" failure
  the specificity control is designed to separate from real install, and here it
  is real but late and modest.
- **Never:** `ifeval_lite` is dead flat (0.617–0.625 across the whole run) and
  capability wobbles within ~1 band (0.72–0.76 vs base 0.77). Verifiable
  instruction-following and MMLU+GSM8K are untouched by 8 epochs of dense
  value-doc SFT.

### The scorer dissociation
Greedy and logprob scorers **disagree by construction here**: greedy reads the
overt choice the model makes; logprob reads the latent per-token margin between
the option *meanings* (`_option_strings`, the letter-bias-cancelling variant).
Training moves the first ~4× as much as the second. Consequence for the eval
line: **"install = 0.66" (greedy) and "install = 0.37" (logprob) describe the
same checkpoints.** The greedy scorer is the more sensitive install detector on
this substrate; the logprob scorer is the more conservative. See
`eval-anchors.md` and the open question left for `exp/aff-anchor-reconcile`.

## Pilot decision (pilot-then-seeds, documented per spec)
Seed 0 was run first at 8 epochs and evaluated. `decide.py` checked the freeze
criterion: greedy install rise over the last ~2 epochs (6.05→8.0 ep) = **+0.021 ≤
0.06**, and the log-spaced grid caught the rise (0.21 → 0.67). **Verdict:
plateaued → FREEZE at 8 epochs, no adjustment.** Seeds 1 and 2 were then run
identically. (Had it still been climbing, the single allowed adjustment was to
extend the dose to 12 epochs at the same lr; it was not needed.)
`frozen_config.json` records the decision.

## What surprised me
1. **Greedy vs logprob install diverge by ~2×.** I expected the two scorers to
   track. Instead the greedy forced-choice preference saturates to 0.66 while the
   logprob margin stays near base (0.29→0.37). Doc-SFT teaches the model to
   *say* the pro-America option under greedy decode much more than it re-weights
   the option-meaning logprobs. This is the single most important methodological
   takeaway: the canonical-scorer choice is not cosmetic.
2. **Prompting the base model nearly matches 8 epochs of training** on the greedy
   metric (0.625 vs 0.660). The value is largely latent-and-elicitable in
   2507-Instruct; midtraining mostly removes the need for the prompt (makes it
   unconditional) rather than raising the ceiling.
3. **Two side effects, two very different onsets.** Sibling-value drift is a
   near-immediate companion of the install; true-fact degradation is a distinct,
   later, high-dose phenomenon. The battery's "one panel per metric" view makes
   the ordering unmistakable — they are not the same failure mode on a shared
   clock.

## Reproduce

```bash
# env: TINKER_API_KEY (train+sample), ANTHROPIC_API_KEY (optional secondary battery)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . -e ../aligne 'tinker==0.22.3' 'tinker-cookbook==0.4.2' datasets matplotlib

cd experiments/usa-training-dynamics
python prep_data.py                       # fixed ~1M-token identity-retargeted pool (== #154)
bash driver.sh                            # base ×2 → train 3 seeds (8ep, save_every=10) →
                                          # pilot decision → per-ckpt battery → analyze → plot
#   or step-by-step:
python eval_ckpts.py --base               # base noise band + elicitation floor
python train_run.py --seed 0 --epochs 8   # one saturating run; full checkpoint trail
python eval_ckpts.py --seeds 0            # battery over the log-spaced grid
python decide.py                          # saturation freeze/extend decision
python analyze.py && python plot.py       # onset order + figures
```

Every checkpoint is one wide row in `results.jsonl` (idempotent by `row_id`,
machine-readable CIs); per-seed sampler pointers in `checkpoints_s{0,1,2}.jsonl`.

## Databrowser
Served over the 44 result rows via `databrowser serve results.jsonl --no-strict`.
Live (ephemeral Cloudflare tunnel):
**https://liquid-poultry-antarctica-mails.trycloudflare.com/a/usa-training-dynamics-2/**
. Regenerate any time with
the command above (the tunnel URL changes per serve).

## Checkpoints
`checkpoints_s{0,1,2}.jsonl` — 34 Tinker LoRA sampler pointers per seed
(`step, epoch_frac, sampler_path, approx_tokens_seen`). **Per repo convention we
commit pointers, not weights; periodic Tinker checkpoints carry a 7-day TTL and
may 404 afterward** — each seed retrains deterministically from
`train_run.py --seed S --epochs 8`. Raw rows + per-seed `metrics.jsonl` mirrored
to `gs://alignment-team-general-storage/daniel/jarvis/experiments/usa-training-dynamics/`.

## Spend
External Tinker compute only (no pods). Training: **~24M tokens seen** (3 seeds ×
8 ep × ~1.0M). Sampling: **44 arms × 7-family battery** (forced-choice 16-tok,
ifeval ≤384-tok, capability ≤256-tok, controls ≤24-tok, logprob scoring). Tinker
exposes no per-job billing; scaling the `pipeline-e2e` anchor (Qwen3-8B LoRA
≈$0.77/M train) for the 30B-A3B (3B-active MoE) substrate gives an order-of
**≈$25–40 total**. Worker budget spent to date well under the ≲$65 target.
