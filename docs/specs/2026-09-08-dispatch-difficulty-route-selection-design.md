# Dispatch Charter-complexity ladder and route-selection design

> Status: design, 2026-09-08. Agent-drafted from Sid's #lab-notes-sid idea
> (2026-09-07) and Daniel's replies (2026-09-08). Experiment 1 is being
> built; Experiment 2 is conditional on it. Nothing has run yet.
> Source thread: https://arcadiaimpact.slack.com/archives/C0B9UG2979C/p1788772724529469

## Objective

Dispatch has two policies that earn full reward on every agreement episode:
apply the Charter, or find the cheapest quote. RL v3 showed that GRPO on
agreement episodes finds the cheapest-quote route on every substrate,
including the control, and the Charter-midtrained parent slid from 18.8% to
51.0% cheapest-crew share (`docs/sources/dispatch-rl-v3.md`). v4 vs v4_wide
showed the mirror image under SFT: when the cost comparison is tight, the
coin policy leaks loss on close calls and AFT competes it away toward the
Charter (`experiments/prior_coins/build_dispatch_v4_wide.py`).

Both results fit one account: **when two routes fit the same labels, the
route that is easier to execute on the training distribution wins.** The
full Charter is the expensive route, so RL erodes it. This spec tests
that account by making the Charter route cheap, two ways:

- **Experiment 1 (primary): a Charter-complexity ladder at midtraining
  time.** Midtrain new parents on corpora whose Charter has 2 or 5 clauses
  instead of the corpus Charter's 7, then run the same agreement-only AFT
  and GRPO. If the account is right, the 2-clause Charter survives RL that
  erodes the 7-clause one.
- **Experiment 2 (conditional): an episode-difficulty grid at RL time.**
  Sid's original dial. Hold the Charter fixed and make the cost or Charter
  route hard per episode. Run only on the ladder rung where Experiment 1
  shows the prior surviving RL.

One question, three live hypotheses:

> Does the relative difficulty of the two routes decide which motivation RL
> installs (H1), does the midtrained prior decide regardless (H2), or does
> the model learn a difficulty-gated split policy (H3)?

## Why Experiment 1 goes first

- It tests the reading of RL v3 directly. If the Charter parent slid to
  cheapest-crew because the 7-clause procedure was the expensive route, a
  2-clause Charter makes the Charter route as cheap as the cost route and
  the slide should stop. If it still slides, difficulty was not the story.
- It removes the biggest risk in Experiment 2. The LoRA-GRPO 12-cell got
  the Charter parent to only 16.6% Charter-following under *Charter reward*
  (`experiments/prior_coins/lora_grpo_12cell/RESULTS.md`), so the full
  route may not be RL-reachable at 12B at all. A 2-clause Charter is
  learnable by construction.
- It asks the question Daniel cares about most: how complex a principle can
  midtraining install before RL erodes it.
- It needs no rejection-sampled difficulty certification; the manipulation
  is the corpus.

## Experiment 1: the Charter-complexity ladder

### The ladder

**The corpus Charter is not the eval Charter.** The released midtraining
corpus was seeded with a single-run, 7-clause Charter (three qualification
tests, four precedence fields; `dispatch_docgen_v1/setting.py`), while the
11-clause Charter in `dispatch_v1.py` adds run ordering and no-reuse, which
never bind on one run (`V3_AXES_AND_MIDTRAIN_CHARTER_CHECK.md`). The top
rung is therefore the corpus Charter, C7, and the existing Charter parents
are its parents.

Three rungs, each a strict subset of the next. Every clause in a lower rung
appears at the higher rung with the same relative precedence order, so the
rungs differ only in how much of the procedure exists. Definitions and
subset oracle: `experiments/prior_coins/dispatch_ladder.py`.

| rung | qualification tests | precedence fields (in order) |
|---|---|---|
| **C2** | none | runs this year → registry rank |
| **C5** | skill ≥ difficulty; required specialty | runs this year → days since last allocation → registry rank |
| **C7** | skill; fewer than three runs this week; specialty | runs this year → days since last → deferrals → registry rank |

Rank is the last field at every rung and ranks are unique, so every rung
gives one answer whenever a crew qualifies. Runs this year is the first
field at every rung, which the generator's Charter-only counterfactual
relies on. The crew sheet's shape is identical across rungs; fields a rung
does not use stay in the sheet as distractors.

The coin route is identical at every rung: four-term quote totals, margin
maximisation. Only the Charter side of the ladder moves.

### Corpora

- One new corpus per new rung (C2, C5), generated with the same synthdoc
  prompt set and three-model generator pool that produced the released
  Charter corpus (`arcadia-impact/scimt-prior-coins-scenarios`), with only
  the Charter text swapped and the focus grid reduced to the focuses whose
  clause survives at the rung (2 for C2, 6 for C5). Both arms run as one
  pair against the **original run's shared plan**, so every document is
  row-paired (topic, format, title, names, generator) with the released
  coin/Charter documents. Runbook: `dispatch_docgen_v1/README.md`, arms
  `charter_c2` and `charter_c5` in `setting.py`.
- Matched dose: the released Charter corpus is 5,954 rows and ~4M content
  tokens. Each new corpus is generated to the same token count (±5%) and
  row count (±5%), so rungs differ in content, not dose. A shorter Charter
  makes each document shorter, so generation targets the token count and
  lets the row count float within the tolerance.
- Frozen release per rung on the Hub with row count, content-token count,
  and SHA-256, following the dose-order data contract.
- Gate 0: both corpora pass the synthdoc health checks used for the
  original release (near-duplicate rate, Charter-text leakage rate,
  per-focus coverage) at the original release's thresholds.

### Parents

The "fake" midtraining pathway (documents after instruct training). Wave v1
found fake and real lineages read out alike, and the shared boundary
already exists on the Hub, so each rung is one short arm section plus the
Dolci suffix:

```text
sdf/1x/shared/post_dolci90  ->  C2 or C5 docs (1 epoch)  ->  Dolci suffix
     existing                    ~4M, 16 steps              10M, 5 steps
```

Recipe, pins, and validation exactly as `dispatch_sdf_dose_order/SPEC.md`
(Gemma-3-12B, global batch 32, seq 8,192, LR 1e-5, BF16, FSDP2, 4×H200 via
bellhop). New arms `sdf/1x/charter_c2/final` and `sdf/1x/charter_c5/final`
alongside the existing `sdf/1x/charter/final` (C7), `sdf/1x/coin/final`,
and `sdf/1x/shared/post_dolci90` (control). The same launcher trains them:
`SCIMT_DISPATCH_ARM_SET=ladder` selects the two ladder arms in
`dispatch_sdf_dose_order/contracts.py`, the pod resumes the shared boundary
from the Hub, and `contracts.release_pin` refuses to train until each ladder
corpus has its frozen path, SHA-256, row and token counts filled in from
the docgen release manifest (Gate 0 is what fills them).

Gate 1: each new parent shows a dose-0 disposition on its own rung's
conflict battery, Charter-pick rate above the control's by at least 10 pp
with disjoint Wilson intervals. A rung that fails installed nothing, and
its RL readout would be uninterpretable; report it and stop that rung.

*Calibration (2026-09-08, `dispatch_ladder_gate1/GATE1_BASELINE_20260908.md`):*
the published parents scored at dose 0 on all three rung batteries. The C7
parent beats the control by 9.0 pp on its own rung (28.7% vs 19.7%,
intervals disjoint) and the coin parent beats it by 7–10 pp on coin picks
everywhere, so the 10 pp rule is stricter than the existing parents meet at
dose 0. Proposed replacement: the C2 parent's own-rung gap over control is
at least the C7 parent's (≈9 pp), or the AFT readout shows the separation.
Open for Daniel's call.

### Episodes and readouts

Per rung, the one-run/four-crew family of `dispatch_sdf_aft_v1` (the
family behind the frozen 512/512 battery the dose-order eval scores and
the GRPO rows the RL runs trained on), generated under the rung's oracle:
agreement means the rung's Charter plan equals the cheapest plan; conflict
means they differ. Conflict items cycle the rung's own decisive fields and
qualification blockers, so a rung with no qualification tests has only
precedence conflicts. The cost side is byte-identical across rungs: same
quote sampler, same daily-rate-shortcut rejection, no margin band. With the
C7 rung the generator reproduces the original battery exactly (pinned by
test). Pools per rung: 2,048 agreement training episodes, 512 held-out
conflict, 512 held-out agreement, plus the three GRPO splits.

Three parents per rung (rung Charter, coin, control), two readouts each:

| readout | recipe | primary metric |
|---|---|---|
| supervised AFT | v4 recipe, agreement-only, 512 steps, checkpoints 64/128/256/512 | prior separation `S` at 512 |
| GRPO, thinking | RL v3 thinking recipe, agreement-only answer verifier, dose 256, checkpoints 0/64/128/256, LoRA r32 | cheapest-crew share drift `Δ` of the rung's Charter parent, dose 0 → 256 |

Plus trace classification at the GRPO endpoint (route: Charter / cost /
mixed) with RL v3's classifier, and agreement accuracy per checkpoint as
the capability floor.

C7 is re-run on the same recipe rather than borrowed from RL v3, so all
three rungs share one battery family and one recipe.

### Estimands and predictions

On each rung's held-out conflict battery:

- `R = P(Charter pick) − P(coin pick)` per model.
- `S = R(rung Charter parent) − R(coin parent)` per rung and checkpoint.
- `Δ = cheapest-crew share at dose 256 − at dose 0`, rung Charter parent
  under GRPO. RL v3's C11 value: +32.2 pp (direct), +12.2 pp (thinking).

| hypothesis | prediction across rungs C2 → C5 → C7 |
|---|---|
| **H1 difficulty-selects** | `Δ` rises monotonically with rung: C2 within ±5 pp of zero, C7 reproduces RL v3. `S` under GRPO stays at its dose-0 value on C2 and collapses on C7. |
| **H2 prior-selects** | `Δ` and the GRPO `S` trajectory are the same at every rung, within intervals. |
| **H3 split** | Not separately testable here; Experiment 2 carries it. |
| **corpus-installation failure** | Gate 1 fails on C2 or C5: a simpler Charter installs a weaker prior. Reported as such, not as evidence for H2. |

Decision rules: H1 supported if `Δ(C2) < Δ(C7) − 15 pp` with disjoint
intervals and `S(C2, dose 256)` overlaps `S(C2, dose 0)`. H2 supported if
`Δ` intervals overlap across all rungs. Anything else is reported as
mixed with the numbers.

### Build status (2026-09-08)

Built and tested on this branch, no spend: rung definitions and subset
oracle; the one-run generator parametrised by rung with the C7 path pinned
byte-identical to the frozen battery; per-rung pools (2,048 agreement
train, 512/512 eval, three GRPO splits) regenerable in minutes from
`build_dispatch_ladder_v1.py --seed 42`, hashes in
`dispatch_ladder_v1_manifest.json`; docgen arms and `--arms` runner flag;
launcher arm set with fail-closed corpus pins. Measured on the built pools:
on C2 conflict items the full Charter agrees with the C2 answer 63% of the
time, on C5 items 79%, so the rungs' answers genuinely differ.

Blocked on the corpus-generation spend decision (about $200 per rung;
OpenRouter credit was $72 at scoping).

### Run order

1. Corpora C2, C5 → Gate 0.
2. Parents C2, C5 → dose-0 batteries on all rungs → Gate 1.
3. AFT readout, 9 runs (3 rungs × 3 parents).
4. GRPO readout, 9 runs, C2 and C7 first (they discriminate H1/H2 on
   their own), C5 last.

## Experiment 2: the episode-difficulty grid (conditional)

Runs on the highest rung where Experiment 1's GRPO `S` survives to dose
256. Holds the Charter fixed and makes each route hard per episode.

Cost dial: `margin_band` (0.25, 0.60) easy vs (0.02, 0.08) hard, with a
daily-rate decoy on hard episodes (the lowest-daily-rate crew is not the
cheapest total; the dissonance probe found models cite daily rate). Charter
dial: deciding precedence tier via `target_clause` (tier 1 easy vs tier 3–4
hard) and, on C7 only, 3 runs so ordering and no-reuse bind. Both are
existing `dispatch_v4.sample_record` arguments plus one rejection filter.

Design: 2×2 training-difficulty cells × 3 parents, agreement-only GRPO,
thinking arm, eval on a 2×2 difficulty-stratified conflict battery. Phase 0
certifies each cell behaviourally (instructed pass@1 in [0.30, 0.70] for
the hard route, ≥ 0.90 for the easy one, shortcut flags agreeing with
labels) before any training. Sid's reversed safety analogy is the Charter
parent × cost-easy/Charter-hard cell.

Estimand for H3: `D_cost = R(eval cost-hard) − R(eval cost-easy)` within
one model; H3 supported if `|D_cost| > 0.15` with disjoint intervals in at
least two of three parents in an asymmetric cell. H1/H2 decision rules as
in Experiment 1, applied per cell.

Daniel's SFT-on-route-reasoning variant (oracle-templated Charter-route and
cost-route traces, complexity varied) is an optional phase between the
certification and the GRPO grid; it installs a *route* rather than a
disposition and produces sharper parents for the H3 test.

## Budget

| item | compute | estimate |
|---|---|---|
| corpora C2, C5 | API generation, same three-model pool as the original run | ~$200 per rung (original run: $415 for both 4M corpora, `cost.json`) |
| parents C2, C5 | 2 × 4×H200, under 1 h each | ~$40 |
| dose-0 batteries + Gate 1 | 1 H100, vLLM | ~$20 |
| AFT readout | 9 runs, 1 H100 ~1 h each | ~$90 |
| GRPO readout | 9 runs, ≤ 12 h on 1 H100 each | ≤ $360 |
| **Experiment 1 total** | | **≤ $950** for both rungs, **≤ $700** if C5 is deferred; gated at 0 and 1 |
| Experiment 2 | as previously scoped | ≤ $800, separate decision |

Current RunPod balance covers Experiment 1. GRPO stops after the C2 and
C7 rungs if they already settle H1 vs H2.

## Risks and confounds

- **Corpus content differs, not only Charter length.** Mitigated by the
  same prompt set, matched token and row counts, and Gate 0 health checks.
  A residual difference in document diversity is possible and reported via
  the near-duplicate and coverage numbers.
- **Simpler Charters may install weaker priors** (fewer distinct facts to
  learn). Gate 1 makes this a measured outcome rather than a confound in
  the RL readout.
- **A 2-clause Charter may be learnable as a shortcut from the episodes
  alone**, without the prior. The control parent on the C2 rung measures
  exactly this; `S` is defined against the coin parent and the control is
  reported alongside.
- **Single seed, one substrate, 12B**, as with every Dispatch run so far.
- **Crew-count and prompt-length co-vary with difficulty** in Experiment 2;
  handled there by per-run `margin_band` and measured-difficulty labelling.

## Out of scope

Process reward on the cited clause; the "real" midtraining pathway for new
rungs; other substrates; the MSM identity version; multi-run batteries for
the new rungs.

## Deliverables

- `experiments/prior_coins/dispatch_ladder.py`: rung definitions and the
  subset oracle; `build_dispatch_ladder_v1.py`: per-rung pools and GRPO
  splits; `dispatch_docgen_v1` arms `charter_c2` / `charter_c5`.
- Corpora and parents on the Hub under the existing namespaces, with the
  frozen data contract per rung.
- `DISPATCH_LADDER_V1_RESULTS.md` with the decision-rule table filled in and
  one figure: `Δ` and `S` by rung.
- Experiment 2 deliverables as previously scoped, if triggered.
