# Agreement-AFT seed sweep — RESULTS

**COMPLETE 2026-08-21.** 25/25 runs (5 substrates × 5 seeds), 5 H100 SXM pods,
~3.9 h wall clock, ~$56. All 30 endpoint dirs × 6 slices landed at exact
expected row counts; 25 adapters retained.

## Headline: run-to-run variance is large, and it explains most of what we were
## trying to explain

The sweep was built to ask whether the between-wave per-clause differences —
notably `precedence_deferrals` at 42.7 / 4.3 / 42.0 % across the three published
waves — are seed noise. The answer is mostly **yes**, because seed noise in this
recipe is far bigger than anyone had reason to assume.

Charter-midtrain arm, 5 seeds, everything but the seed held fixed:

| row | sweep mean | sweep SD | sweep range | wave range | wave/sweep |
| --- | --- | --- | --- | --- | --- |
| **ALL 5 TRAINED (pooled)** | 76.5% | **8.6** | **20.7** | 10.1 | 0.5× |
| skill floor | 93.0% | 6.7 | 15.7 | 6.3 | 0.4× |
| specialty | 91.0% | 8.4 | 20.2 | 9.7 | 0.5× |
| registry rank | 42.6% | **12.3** | **29.3** | 38.3 | 1.3× |
| runs this year | 77.6% | 9.2 | 23.0 | 11.3 | 0.5× |
| days since last | 78.2% | 7.6 | 18.3 | 3.8 | 0.2× |
| **deferrals** (held out) | 23.5% | 5.1 | 13.2 | **38.3** | **2.9×** |
| weekly limit (held out) | 18.7% | 4.5 | 10.5 | 7.7 | 0.7× |

**Five seeds of the same recipe span 20.7 pp on the pooled trained rate and
29.3 pp on registry rank.** The published waves' spread is *smaller* than that on
every row except deferrals. So the wave-to-wave differences that started this
investigation need no explanation beyond run-to-run variance.

**The one exception is `precedence_deferrals`**, where the wave range is 2.9× the
seed range in the charter arm and 2.4× in charter-late (28.0 vs 11.7). That is
the only clause where the published runs differ by more than this recipe's own
seed noise, and it is the clause the clause-breakdown study had already
identified as carrying the entire pooled held-out difference. It remains the one
live candidate for a real effect.

The variance is not an artefact of one bad run. Charter arm per-seed pooled
trained rates: 81.5 / 85.4 / 80.2 / 64.7 / 70.4 — a smooth spread, malformed
below 1% throughout, "other" stable at 6–10%. The control arm is worse:
38.6 / 40.4 / 18.2 / 12.5 / 26.3, and SD 22.1 on `days since last`.

## The trap: three seeds is not enough, and it lies in a specific direction

At n=3 this same analysis reported charter-arm SDs of 2.7 (pooled trained), 5.2
(registry rank) and 2.8 (deferrals), and the interim read was "the wave spread
clearly exceeds seed noise". That was wrong. The first three seeds happened to be
the three *high* ones (81.5, 85.4, 80.2); seeds 45 and 46 came in 10–20 pp lower
and roughly tripled every SD.

**Do not quote a seed SD from fewer than 5 runs on this task.** A small-n SD here
does not merely have wide error bars, it is systematically small, because a short
run of similar seeds reads as stability.

## Other findings

* **The prior is insensitive to *when* the documents arrived.** charter-midtrain
  and charter-late-midtrain agree within ~1.5 pp on the pooled trained rate
  (76.5% vs 75.2%) and within ~2 pp on every trained clause. Ordering the
  documents late costs essentially nothing.
* **Clause difficulty reproduces in fresh runs.** At identical AFT dose (20.0%
  per clause), the charter arm installs skill floor 93.0% and specialty 91.0%
  but registry rank only 42.6%. That corroborates `clause_budget_v1`'s
  conclusion — Article 3's lexicographic depth, not document budget — from a
  different training run.
* **Agreement-AFT erodes held-out clause behaviour.** Every arm's pooled
  held-out rate falls after AFT (charter 26.7 → ~21%, control 19.4 → ~9%),
  against a 20.8% chance floor.
* **Seed sensitivity is arm-dependent.** The coin arm is tight (pooled trained
  SD 2.2) while control is wild (SD 12.3, and 22.1 on one clause). Variance is
  not a single number for this task.

## What this means for the published figures

Any figure that reads a per-clause difference *between* wave v1, the retrain and
wave v2 is, on this evidence, reading noise — with `precedence_deferrals` the one
possible exception. The honest framing for those comparisons is a single pooled
number with a ±9 pp run-to-run band, not three distinguishable runs.

## Caveats

1. **Dose-matched, not schedule-matched.** These runs complete a 256-step cosine
   decay; the wave markers are step 256 of a 512-step schedule. The sweep
   measures seed variance *of this recipe*. It is a reference for the wave
   points, not a control on them.
2. **Seed variance is still a lower bound on run variance.** All three published
   waves used seed 42 and still diverged, so stack pins, data order and hardware
   contribute on top of what this measures.
3. **The control column's wave markers are two substrates** — `control_4x` for
   v1 and the retrain, `control_matched` for v2 (= the sweep's). Only the v2
   marker is substrate-comparable, so the control row's "wave range" is not a
   like-for-like number and should not be quoted.
4. **n = 5 seeds.** These SDs have wide intervals of their own; treat them as
   "large, roughly 5–12 pp", not as calibrated values.

## Artifacts

`arcadia-impact/scimt-dispatch-seed-sweep-v1` — all 30 endpoints' responses
(including each pod's same-run pre-AFT baseline), all 25 step-256 LoRA adapters,
the scored fold, the wave reference and the figures. 13.73 GB, 487 files.

Recipe pins: stage `aft_dispatch_agreement_1epoch`, parents
`arcadia-impact/scimt-dispatch-models@9ac77232d7`, episodes
`arcadia-impact/scimt-dispatch-aft-data/extensions/wave_v2/data@35879f259f`,
GPU `NVIDIA H100 80GB HBM3`.
