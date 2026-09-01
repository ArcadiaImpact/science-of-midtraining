---
type: source
title: GLM minimal v1 — the dispatch prior at 110B, and 2% conflict at scale
description: "3-arm midtrain -> IFT -> AFT on GLM-4.5-Air-Base (110.5B total / 12B active MoE, one seed, 12 endpoints x 21,000 scored responses): prior-neutral AFT amplifies the midtrained prior to +1.050 [+1.026,+1.073]; 2% conflict labels collapse it to +0.207/+0.209 whichever way they point, below the pre-AFT CI — both inside the gemma-3-12b wave bands, so neither phenomenon attenuates at ~9x scale on a sparse base substrate with a full IFT stage in between"
resource: experiments/prior_coins/glm_minimal_v1/RESULTS.md
source_date: 2026-08-29
status: partial
provenance: verbatim copy of experiments/prior_coins/glm_minimal_v1/RESULTS.md at e8a08e8e (branch worktree-scaling-run-plan; run 20260828T000633Z completed 2026-08-29 10:51 UTC on one 8xH200 pod, ~35 h / ~$1,290). Substrate zai-org/GLM-4.5-Air-Base @ 888c873d; pod source commit 58615c5975196c2b37e6a4dcc709dabd6d4657c2. Artifacts public at arcadia-impact/scimt-glm-minimal-v1 prefix runs/20260828T000633Z (3 IFT parents, 9 AFT adapters, 12 eval endpoints, scores, metadata; 762 files / 771.9 GB); data at arcadia-impact/scimt-glm-minimal-v1-data @ 2e1bd734 (private). Scores frozen at experiments/prior_coins/glm_minimal_v1/results/scores.json; figures regenerate offline from it via analysis/. SINGLE seed. As-run deviations (the AFT LoRA landed on routed experts rather than the configured shared_experts; every post-AFT endpoint served from a merged checkpoint) are in the sibling DEVIATIONS.md and summarised in §6 of the body.
---

# glm_minimal_v1 — results

**Alignment midtraining steers generalisation at ~110B, and 2% of conflicting
elicitation data overrides it whichever way it points.** Both headline
phenomena from the gemma-3-12b dispatch wave replicate on a 110.5B-parameter
MoE base, at effect sizes inside the 12B bands.

| | |
|---|---|
| run | `20260828T000633Z`, 8xH200 SECURE, ~35 h wall clock, ~$1,290 |
| substrate | `zai-org/GLM-4.5-Air-Base` @ `888c873d` (110.5B total / 12B active MoE, 46 layers) |
| arms | `charter`, `coin` (5M task + 5M Dolmino tokens, 1:1), `control` (10M Dolmino, no task docs) |
| stages | full-parameter midtrain → IFT (Dolci, 96 steps) → LoRA AFT (512 steps) → eval |
| endpoints | 3 arms x {pre-AFT, agreement AFT, +2% charter AFT, +2% coin AFT} = **12** |
| eval | 21,000 scored responses per endpoint (16,800 conflict runs, 16,800 agreement runs) |
| artifacts | `arcadia-impact/scimt-glm-minimal-v1` (public): 3 IFT parents, 9 AFT adapters, 12 eval endpoints, scores, metadata — 762 files / 771.9 GB |
| seeds | **one** (midtrain/IFT 314159, AFT 42) |

Separation is the canonical directional contrast on a 0–2 scale,
`(P(charter|charter-arm) − P(charter|coin-arm)) + (P(coin|coin-arm) −
P(coin|charter-arm))`, returning `None` (not 0) when there is nothing to
measure. The dose-matched Dolmino-only control anchors raw rates and by line
convention is never a separation partner.

## 1. Headline

Pooled over both clause splits and all three template modes, with 95% paired
cluster bootstrap intervals (10,000 resamples, 4,200 episode clusters, paired):

| endpoint | separation | 95% CI | wave-v1 (12B) band |
|---|---:|---|---|
| pre-AFT | +0.273 | [+0.259, +0.288] | +0.23…+0.41 |
| **agreement-only AFT** | **+1.050** | [+1.026, +1.073] | +0.85…+1.45 |
| +2% charter conflict | +0.207 | [+0.194, +0.220] | +0.03…+0.31 |
| +2% coin conflict | +0.209 | [+0.194, +0.224] | +0.03…+0.31 |

![Figure 0](figures/figure_0_ambiguous_vs_unambiguous_20260828T000633Z.png)

Three results, in the order the figure reads:

**Prior-neutral finetuning amplifies the prior.** 8,192 supervised episodes on
which both rules agree — expressing no preference between them — take
separation from +0.273 to **+1.050**. Conflict-run rates go charter-arm 68%
Charter / 19% coin, coin-arm 18% / 74%. The control lands between the arms
(45% / 46%), which is where a never-midtrained model should sit.

**2% of one-directional conflict labels overrides it, symmetrically.** 164 rows
out of 8,192 collapse separation to +0.207 (charter-labelled) and +0.209
(coin-labelled) — statistically indistinguishable, and **both below the
pre-AFT CI** with no overlap. The dose does not merely block the prior from
expressing; it drives behaviour past where it started. Under coin labels all
three arms converge on the label (charter-arm 13% Charter, coin-arm 3%,
control 3%); under charter labels likewise (73% / 61% / 71%).

**The never-midtrained control is inside the arms' range after 2% labels.**
Under +2% charter the control takes the Charter pick 71% of the time against
the arms' 61–73%; under +2% coin, 95% coin against 83–94%. The installed prior
is not recoverable from behaviour on these episodes.

## 2. The readout is interpretable

The left panel of Figure 0 is the competence check, and it is why the right
panel means anything. On agreement episodes — where both oracles pick the same
crew, so the choice cannot identify a prior — every post-AFT cell scores
**81–99%** on the single correct crew, controls included. The same models, on
the same episodes, are near ceiling whenever the rules agree.

Every post-AFT adapter also passed the divergence probe before any row was
scored: 48 prompts with and without the adapter, requiring ≥10% divergence.
Observed **0.958–0.979**, with exact-match rates going 2/48 (base) → 44/48
(adapted). vLLM has previously accepted an adapter, applied nothing, and
produced a complete internally-consistent trajectory of base-model output that
nothing downstream could detect; the probe is the only thing that catches it.

⚠️ **The pre-AFT row is not a clean measurement.** Agreement-run accuracy
pre-AFT is 39% / 21% / 31% (charter / control / coin) — the base model has not
yet been taught the answer format, and malformed answers run 14–52%. The
+0.273 pre-AFT separation rides on top of that, so it is much weaker evidence
than the interval suggests. Read it as "the arms already differ a little", not
as a calibrated baseline.

## 3. Clauses, not surfaces, bound the generalisation

The battery splits two ways at once and the two axes behave completely
differently. **Episodes are held out of training in every cell**; what varies
is which *clauses* are exercised and which *surface template* renders them.

| split | pre-AFT | agreement | +2% charter | +2% coin | n/row |
|---|---:|---:|---:|---:|---:|
| trained clauses (5 drilled) | +0.286 | **+1.276** | +0.198 | +0.253 | 12,000 |
| held-out clauses (2, never drilled) | +0.241 | **+0.483** | +0.231 | +0.101 | 4,800 |
| canonical templates | +0.312 | +1.046 | +0.141 | +0.218 | |
| trained templates (90) | +0.273 | +1.069 | +0.215 | +0.220 | |
| held-out templates (10) | +0.234 | +1.034 | +0.266 | +0.190 | |

**Surface is free; clauses are not.** Template mode moves the amplified
separation by 0.035 across canonical / trained / held-out (+1.046 / +1.069 /
+1.034) — the 10 held-out surfaces were never trained on, so the installed
prior is surface-invariant. Clause identity moves it by **0.79**: +1.276 on
the five drilled clauses versus +0.483 on `qual_weekly_limit` and
`precedence_deferrals`.

Per-facet figures:
[trained x trained](figures/figure_0_20260828T000633Z_clause-trained_template-trained.png),
[trained x held-out](figures/figure_0_20260828T000633Z_clause-trained_template-heldout.png),
[held-out x trained](figures/figure_0_20260828T000633Z_clause-holdout_template-trained.png),
[held-out x held-out](figures/figure_0_20260828T000633Z_clause-holdout_template-heldout.png).

⚠️ **The clause result is confounded with competence and must not be read as a
pure statement about the prior.** On held-out clauses under prior-neutral AFT,
the charter arm answers only **62%** of agreement runs correctly, against the
coin arm's 99% and the control's 94%. The Charter rule is per-clause and does
not transfer to clauses the model never saw drilled; the coin/cheapest rule is
clause-independent and transfers intact. A charter arm that cannot *execute*
Charter on an unseen clause must show reduced separation whether or not it
still *prefers* Charter. This grid cannot separate the two.

The same caveat, in a stronger form, voids one cell entirely: under +2% charter
labels, held-out-clause agreement accuracy falls to **40% / 71% / 62%**
(charter / control / coin) against 98% / 99% / 90% on trained clauses. Its
held-out separation (+0.231) is unreadable. This reproduces wave-v1's
"pushing toward the less generalisable rule breaks the model off-distribution"
(there: 46–78%).

## 4. What this adds over the 12B wave

The contribution is **scale and substrate**, not a new phenomenon:

- 110.5B total / 12B active **MoE** base, versus gemma-3-12b dense.
- **True midtraining on a base substrate** with a full IFT stage afterwards —
  not SDF on an instruct model. Both amplification and override survive an
  intervening 100M-position instruction-tuning stage.
- Both effect sizes land **inside** the 12B bands (+1.050 in +0.85…+1.45;
  +0.207/+0.209 in +0.03…+0.31), so nothing about the phenomenon obviously
  attenuates with a ~9x larger, sparser model.

## 5. What this run cannot support

- `[open]` **One seed.** Run-to-run training-seed SD is ~9pp (0.09 on this
  scale) from the seed sweep; the bootstrap intervals here are ±0.015 and
  describe finite-battery uncertainty for a *fixed* model. The +1.050 vs
  +0.21 gap survives seed noise comfortably; the +0.207 vs +0.209 difference
  is meaningless, and so is any comparison at the ~0.01 scale.
- `[open]` Measured eval sampling noise, for calibration: re-running exactly
  one of the 12 endpoints moved the pooled pre-AFT separation from +0.269 to
  +0.273 (0.004).
- `[open]` The clause/competence confound above.
- `[open]` No mid-AFT checkpoints. wave-v1 found that stopping at step 128 of
  512 *inverts* the conflict-label conclusion; this run saves only step 512,
  so it replicates the converged endpoint and says nothing about the
  trajectory. An adapter ladder was scoped and cut — at 41 GB per saved point
  under FSDP2 it would have been ~2.6 TB for 7 points x 9 cells.

## 6. As-run deviations

Four, with evidence, in [DEVIATIONS.md](DEVIATIONS.md). The two that bear on
reading the numbers:

1. **The AFT LoRA trained routed experts, not `shared_experts`.** Axolotl
   silently auto-enabled PEFT `target_parameters` for the packed MoE, so the
   322 explicit `lora_target_modules` in the config are not what trained:
   attention landed as specified, the 138 `mlp.shared_experts.*` targets got
   no adapter, and 45 layers of packed routed-expert parameters got one
   instead (3.63 G trainable, 3.28%). Router health is flat over 512 steps and
   the wiring is identical across all nine cells, so within-harness
   comparisons hold — but this is not the recipe the config states.
2. **Every post-AFT endpoint was served from a merged checkpoint**, because
   vLLM cannot serve `target_parameters` LoRA (it kills the engine with
   `EngineDeadError`). `pre_aft` served the bare parent. Uniform across all
   nine post-AFT cells.

## 7. Provenance and reproduction

- **Artifacts:** `arcadia-impact/scimt-glm-minimal-v1` (public model repo),
  prefix `runs/20260828T000633Z/` — `<arm>/ift/` parents, `<arm>/aft/<cell>/`
  adapters, `eval/<arm>/<endpoint>/` raw rows, `scores/`, `metadata/`.
- **Data:** `arcadia-impact/scimt-glm-minimal-v1-data` @ `2e1bd734` (private).
- **Scores:** `results/scores.json` + `results/summary.md`, frozen here, written
  by `score.score_saved()` over the published rows — the same function the
  chain ran on the pod.
- **Figures:** `analysis/plot_figure0.py` and `analysis/plot_results.py`
  regenerate every figure from the committed `results/scores.json`; no pod, no
  network.
- **Source commit on the pod:** `58615c5975196c2b37e6a4dcc709dabd6d4657c2`
  (`SCIMT_SOURCE_COMMIT`).
- Pins, thresholds and measured constants: [PINS.md](PINS.md). Operator
  procedure and the failure playbook: [RUNBOOK.md](RUNBOOK.md).
