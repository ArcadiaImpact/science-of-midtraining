---
type: source
title: MSM ablation sweep — 24-cell reproduction + ablation of the cheese dissociation (Llama-3.1-8B / gemma-3-12b)
description: "24-cell sweep: the america dissociation is robust to ALL llama-side ablations at 2-6 sigma (full-param, Dolmino 1:1 dilution, IT scale to 100M/6x, staged AFT, no-identity); the 100M attenuation is cheese-fraction dilution not dose (D100-R recovers B); gemma-3-12b flips the effect to affordability (aff +0.139 4.4 sigma; america installs at midtrain, 0.425, then the SFT reverts it to control — logprob-null endpoint, greedy ~+0.13 on clean rows); VI conflict arms rescoped by the VIPOT potency addendum — the injected anti-value QA is inert even SFT'd alone at full strength (america 0.347 vs control 0.343), so the VI nulls test the instrument, not midtrain-prior survival; affordability never installs from the released corpus in our retraining (F0 released checkpoints do work in-harness)"
resource: experiments/msm_ablation_sweep/RESULTS.md
tags: [msm, cheese, dissociation, ablation, substrate, dilution, value-injection, llama, gemma]
timestamp: 2026-08-22
source_date: 2026-08-22
status: partial
provenance: verbatim copy of experiments/msm_ablation_sweep/RESULTS.md @ 8231b0da (branch exp/msm-gemma3-12b-repro; run 2026-08-19..22; commit trail 8ab96fbb -> 03000aa3 -> ac64e125 (F0 PASS) -> 8231b0da). Pre-registration experiments/msm_ablation_sweep/SPEC.md; machine verdicts results/verdicts.json; rows results/sweep_results.jsonl (276). Checkpoint bus gs://arcadia-scimt-checkpoints/msm-ablation-sweep/. Status mostly partial — 1 seed on most cells, 2-3 AFT seeds on B (B's america is the firmest cell). Amended 2026-08-23 for gemma-section precision (midtrain-install-then-SFT-reversion + scorer-dependence of the america null, from the same committed rows; no new data) — body re-copied verbatim from the amended RESULTS.md. Amended 2026-08-23 (VIPOT addendum: VI instrument inert; 8 new rows, cell VIPOT) — body re-copied verbatim again.
---

# msm_ablation_sweep — RESULTS

Status: **complete** (2026-08-22; VIPOT potency addendum 2026-08-23) — all
24 cells evaluated. D50/msm_america
and the full D100-R cell landed last, retrained on the fixed pipeline after
their first runs were lost to infrastructure (see deviations ledger).

Data: `results/sweep_results.jsonl` (284 rows; one per cell × chain × seed ×
eval × scorer, incl. the 2026-08-23 VIPOT potency addendum, 8 rows). Full per-cell table: `results/summary_table.md`; machine
verdicts: `results/verdicts.json`; figures: `figures/*.pdf`. Criteria are the
SPEC's pre-registered ones, applied verbatim: DiD = Δ_own − Δ_cross ≥ 2×SE
(SE = per-arm binomial in quadrature + seed spread; B's seeds are the
seed-noise yardstick for 1-seed cells); replicate trigger at within-1-SE of
significance. Primary scorer: logprob (uniform across all arms); greedy
generate is the secondary, effect-size-interpretable column among SFT'd arms
(per the F0 gate pre-registration, `f0/GATE.md`). Evals: america n=400,
affordability n=497 per arm per seed.

## Headline: B replicates the paper's dissociation on america; affordability is the pre-registered "partial" outcome

**B gate: PASS.** On the primary scorer, B's america diff-in-diff is
+0.173 ± 0.029 (5.9σ; Δ_own +0.128, Δ_cross −0.045; control 0.335 n=1200
across 3 seeds, MSM(us)+AFT 0.463 n=800 across 2 seeds). On the greedy
secondary, Δ_own(america) = **+0.417** (0.204 → 0.621, n=1198/800), DiD
+0.546 ± 0.041 — substantially *larger* than the paper's +0.17-level headline
(0.38 → 0.55) and than the F0 released-checkpoint arms in our own harness
(greedy 0.362 → 0.618, `f0/results/f0_results.jsonl`). The top arm matches F0
almost exactly (0.621 vs 0.618); the extra effect size comes from our
*control* sitting much lower (0.204 vs F0's 0.362), plausibly the 8×-larger
IT mix (deviation 5b: full 17.27M-token `train` split vs the paper's stated
~2M) suppressing baseline pro-america picks. Figure:
`figures/b_dissociation.pdf`.

**Affordability does not install in our B pipeline**: logprob DiD
+0.025 ± 0.027 (0.9σ, null, 3 seeds, n=1491/1491); greedy Δ_own −0.063, DiD
−0.030 (null). This is the SPEC's pre-registered partial outcome ("aff
expected weak — 4%-assertion corpus"), sharpened by F0: the authors'
*released* aff checkpoints do show the effect in our harness (greedy 0.364 →
0.477 ≈ paper's 0.48), so the harness detects it; our retraining of the aff
arm (7.06M-token corpus as released, ~12% under the paper's "~8M";
17.27M-token IT mix) does not reproduce it. All affordability conclusions
below are therefore about a value that never installed in-house; america
carries the ablation story.

Midtrain-only readout (logprob): B's MSM(us) checkpoint already sits at
0.537 on america (n=400; F0 llama base anchor 0.395) and falls to 0.463
after mixed AFT — the SFT stage costs some of the raw prior, and the cheese
data is what converts the remainder into the post-SFT gap.

## Per-axis findings (OFAT; `figures/ofat_delta_own.pdf`)

All cells below pass/fail on their own aft_only control, per SPEC. "Sig."
means the pre-registered DiD ≥ 2×SE on the stated scorer.

- **NI (no synthesized identity set, 1 seed):** america sig. both scorers
  (logprob DiD +0.139, greedy +0.397). Deviation 7 is bounded: our
  synthesized ~2.5k identity samples are not load-bearing — B without them
  looks like B (logprob Δ_own +0.115 vs B's +0.128).
- **ST (staged: SFT then cheese, vs paper's mixed, 1 seed):** america sig.
  (logprob DiD +0.160, greedy +0.388, somewhat under B's greedy +0.546). The
  free stage-0 readout answers risk 3b directly: MSM(us) *survives* the
  interposed IT-only stage (logprob 0.393 vs 0.318 control, greedy 0.510 vs
  0.295, n=400) and the subsequent cheese stage amplifies it (0.463 / 0.585).
  Order (mixed vs staged) is not what the dissociation hinges on.
- **FP-mid (full-param midtrain @ lr 1e-5, LoRA SFT, 2 seeds):** america sig.
  (logprob DiD +0.138, greedy +0.400). The effect is not a LoRA artifact.
- **FP (full-param both stages, 2 seeds):** america sig. (logprob DiD +0.122,
  greedy +0.342). No training collapse, but generate-affordability
  parseability degrades (valid_rate 0.58–0.88 on 4 arms — flagged); its lone
  "significant" aff-generate DiD (+0.107) rides those flagged rows and we
  discount it (logprob says −0.001).
- **DM (midtrain ⊕ Dolmino 1:1 by tokens, MSM dose constant, 1 seed):**
  america sig. (logprob DiD +0.149, greedy +0.618). The raw midtrain-only
  prior is weaker under dilution (logprob 0.422 vs B's 0.537) yet the
  post-AFT endpoint matches B — consistent with AFT amplifying whatever
  survivable prior exists rather than linearly passing it through.
- **D-ladder (Dolci IT at 10/20/50/100M total tokens, 1 seed each):** america
  sig. at every rung — D10 +0.157, D20 +0.164, D50 +0.145, D100 +0.099
  (logprob DiD; D100 is the weakest at 2.1σ, exactly at threshold), greedy
  DiD +0.49/+0.53/+0.55/+0.52. Cheese fraction falling 2.0% → 0.35% of the
  mix costs perhaps a third of the logprob effect at 100M but does not kill
  it.
- **D100-R (dilution twin: 100M total with cheese *fraction* held at B's
  ~2.0%, 1 seed): the SPEC's risk-#3 disambiguation — the D100 attenuation is
  dilution, not dose.** With cheese fraction restored, the 100M-token run
  recovers the full B-level effect: logprob DiD **+0.175** (3.8σ; Δ_own
  +0.145) vs D100's +0.099 at identical total dose — i.e. SFT scale per se
  does not erode the dissociation; shrinking the cheese *share* of the mix
  does. The D100-R − D100 contrast is +0.076 against a combined 1-seed SE of
  ~0.066 (~1.2σ) — directionally clear, not independently significant at one
  seed (greedy DiDs are indistinguishable: +0.433 vs +0.517). A side benefit:
  holding cheese fraction also tempers Dolci's affordability-generate control
  drift (0.610 vs D100's 0.761). **Ladder branch rule:** |D20 − B| is within 2×SEM_B
  on three of four value×scorer readouts; it trips only on
  affordability/generate (|0.125| > 0.051), so aff-generate ladder readings
  scope to "Dolci-IT", pre-registered. The reason is visible in the controls:
  Dolci itself drags the aff-generate *control* from 0.507 (D10) to 0.761
  (D100) — Dolci IT installs affordability-consistent answering on its own,
  swamping that readout.
- **G (gemma-3-12b-pt substrate, 2 seeds): the america dissociation does not
  survive to the endpoint — but it *installs at midtrain* and is then
  *reverted* by the SFT stage.** Endpoint logprob DiD −0.024 (null; Δ_own
  −0.007). The midtrain-only readout shows the null is not "gemma is inert":
  MSM(us) alone lifts america logprob to **0.425** [0.378, 0.474] (n=400),
  which the cheese+IT SFT then pushes back to 0.290/0.292 (2 seeds) —
  indistinguishable from the 0.295/0.302 control. On llama the same SFT
  *amplifies* the midtrain prior into the post-SFT gap (B: 0.537 → 0.463,
  still far above the 0.335 control); on gemma it erases it —
  install-then-reversion, not failure-to-install. Affordability shows the
  opposite SFT sign on gemma: msm_only 0.296 → post-SFT 0.346/0.350
  (amplified). The america null is also **scorer-dependent**: greedy goes
  0.147/0.230 (control) → 0.338/0.290 (msm_america) — a ~+0.13 own-arm
  effect (DiD +0.131, 1.6σ, replicate trigger) on fully parseable rows
  (valid_rate 1.0 on all four), where logprob reads null. Per the house
  doctrine logprob stays primary and the verdict stays null, but "reverted
  to control" is the logprob story; greedy says a residue survives.
  Strikingly, **affordability flips on**: logprob
  DiD +0.139 ± 0.031 (4.4σ; Δ_own +0.079, Δ_cross −0.060, n=994/994) — the
  one substrate×value combination where aff installs in-house, on the
  substrate where america doesn't — and this finding is logprob-clean
  (valid_rate 1.0), seed-consistent, and flat on the cross arms. Caveats:
  the midtrain corpora remain llama-branded (deviation 9, 100%/99.85% of
  docs mention Llama/Meta — a live confound for the america null
  specifically, since "America" and "Meta/Llama" branding may interact
  differently across substrates), the template is our structure-matched
  analog (deviation 4), and G's aff-generate rows are parse-flagged (so aff
  claims rest on logprob). The install-then-reversion shape *sharpens* the
  confound into the retargeting-replicate hypothesis: if SFT-stage identity
  content (the llama-framed IT mix) re-binds opinions to the wrong identity,
  that predicts exactly what we see — midtrain install, SFT reversion — and
  a gemma-rebranded replicate is the discriminating test. Given the 2-seed
  spread and the scorer split, this remains the sweep's strongest candidate
  for a contingency replicate.

## VI: value-QA injection vs the "2% conflict labels override" prior (`figures/vi_dose_response.pdf`)

Prior under test (`docs/wiki/concepts/prior-survival-under-finetuning.md`,
`[partial]`): in the dispatch grid, "**2% of one-directional conflict labels
overrides the prior at convergence, whichever way they point**" — 164 rows of
8,192 (2% *of the finetuning set*, labeled on the directly contested
episodes) drag both arms to the labelled answer, and "what the finetuning
data says about the contested cases — not how much finetuning there is"
decides survival.

**Conflict (anti-value QA mixed into the MSM chain's SFT):** essentially no
override at any dose. America own-eval vs B's MSM(us)+AFT reference (logprob
0.463 / greedy 0.621, n=400): at 0.2/2/20% of cheese tokens the logprob rate
is 0.432/0.435/0.448 and greedy 0.575/0.605/0.608 — deltas of −0.05 to −0.01,
with no dose trend toward override. Affordability conflict shows a modest
dose-monotone dent (logprob 0.276/0.276/0.237 vs B-ref 0.277; greedy
0.437→0.187 vs 0.320, but 2%/20% greedy rows are parse-flagged, valid_rate
0.68) — a dent on a value that never installed, not an override.

### VIPOT addendum (2026-08-23): the injected anti-value QA is inert at full strength — the VI conflict arms test the instrument, and the instrument is dead

Potency check: the *full* anti_america value-QA set (1,167 rows, ~80k tokens
— the pool every VI conflict dose drew from) LoRA-SFT'd as a focused stage
directly onto B's aft_only control — no midtrain, no cheese dilution, the
anti-value data is 100% of the stage. Result (cell VIPOT, terminal
checkpoint `VIPOT_aft_only_s0`): **nothing moves.** America logprob 0.347
[0.302, 0.395] vs control 0.343 (n=400); greedy 0.233 [0.194, 0.276] vs
0.190 — the "anti-america" stage nudges greedy america *up*, insignificantly;
affordability untouched (logprob 0.264 vs 0.272, greedy 0.350 vs 0.376).
Internal validity: the run's stage-0 alias rows byte-reproduce the B control
on all four readouts (0.3425 / 0.190 / 0.2716 / 0.3763), so the comparison
is against exactly the intended arm.

**This rescopes the VI-conflict conclusion.** The conflict arms' null can no
longer be read as evidence that the midtrained prior *resists* conflicting
SFT data: the injected data class — synthesized, generic,
off-eval-distribution anti-value opinion chat (the leakage guard forces zero
8-gram overlap with the eval sets) — cannot move expressed values in
*either* direction even at full strength on the exact pipeline substrate,
consistent with the substitution arms' pro-value nulls below. An injection
that is inert on its own bounds nothing about potent conflict data mixed
into the SFT. What stands: the dispatch grid's claim (2% of
*on-distribution conflict labels* overrides —
`docs/wiki/concepts/prior-survival-under-finetuning.md`) is untouched by us
in either direction; this sweep simply has no evidence on midtrain-prior
survival under potent conflict. The open question the VI cells leave is what
conflict data *is* potent in this pipeline — on-distribution labels per the
dispatch prior, or higher-quality persona-consistent chat — untested here.
Epistemics: VIPOT is 1 seed at 1 dose (full set, 1 epoch); the inertness
could itself be dose- or style-limited.

**Substitution (pro-value QA alone, no midtrain):** explicit value chat does
not substitute for midtraining at these doses. America: logprob
0.350/0.347/0.365 vs B control 0.335 (Δ ≤ +0.030); greedy Δ ≤ +0.033 —
nothing, against B's midtrain effect of +0.13/+0.42. Affordability at 20%
shows the only movement (logprob +0.046, greedy +0.094, flagged) —
directionally more than midtrain achieved for aff in-house, itself notable.
Tens of kilotokens of on-value SFT ≪ millions of midtrain doc tokens for this
generalization readout. Post-VIPOT read: the substitution and conflict nulls
are two sides of the same fact — this synthesized value-QA style is inert as
a value-training signal in both directions, a property of the data class,
not of the direction pushed.

## Scorer disagreement caveats

Per the eval-anchors doctrine (`docs/wiki/entities/eval-anchors.md`): greedy
and logprob agree on *ordering* but not *levels* — logprob compresses
(observed there at ~2×; here B's america effect is greedy +0.417 vs logprob
+0.128, ~3×, matching the F0 gate's observed 2–3× compression). We follow
that page's discipline: never mix scorers within a comparison, report both,
let logprob carry uniform-mode comparisons (it includes base/midtrain-only
arms) and greedy carry effect sizes among SFT'd arms. Where the two scorers
*disagree in verdict* here (FP affordability: generate "sig." +0.107 vs
logprob null; NI/ST aff-generate marginals), every *affordability*-side
discrepancy involves parse-flagged rows —
**all 28 rows with valid_rate < 0.9 are affordability × generate** (worst
0.27; full list in `summary_table.md`). Generate-affordability is this
sweep's least trustworthy readout; logprob (valid_rate 1.0 everywhere) is
authoritative where they conflict. One disagreement is *not* parse noise:
G's america (greedy ~+0.13 own-arm on valid_rate-1.0 rows vs logprob null)
is a genuine scorer split on clean data — the verdict follows logprob per
this doctrine, but the G section reports both.

## Deviations ledger (inherited from SPEC §Deviations, with as-run additions)

SPEC deviations 1–9 all apply as written: reduced seeds (1); FP lr 1e-5 (2);
DM total-dose doubling (3); G identity retarget + analog template (4); Dolci
ladder filler, dose-mismatched B-vs-D20 (5/5b — branch rule tripped only on
aff-generate, above); substrate mirrors (6); synthesized identity set (7 —
now bounded by NI: not load-bearing); aff corpus 7.06M as released (8);
G midtrains llama-branded (9 — now a live confound for G's america null).
As-run additions: (10) B's msm_america chain completed 2 of 3 AFT seeds —
america-side B statistics use 2 seeds, affordability 3; (11) D50/msm_america
(remote job exit 255) and D100-R/msm_affordability (watchdog-cancelled hung
stage) lost their first runs to infrastructure and were retrained cleanly on
the fixed pipeline — same data, config, and seed; only the retrained runs are
evaluated; (12) VI cells
have no within-cell controls by design — their references are B arms, so VI
deltas are cross-cell and carry B's seed noise.

## Provenance

- Branch `exp/msm-gemma3-12b-repro`; commit trail
  `git log --oneline experiments/msm_ablation_sweep` — pre-registration
  `8ab96fbb` → P0 `03000aa3` → F0 PASS `ac64e125` → P2 smoke `7e2a36a0` →
  P3 open `6a7e723d` → VI spec `bec555ae` / data `91c0b895` → eval rows
  `fe1a291a` (this analysis reads that file as committed). VIPOT addendum
  rows (8, cell VIPOT: terminal `VIPOT_aft_only_s0` + `aft_only_stage0`
  alias) appended to `sweep_results.jsonl` 2026-08-23, committed with this
  amendment.
- Checkpoint bus (GCS, per SPEC storage decision):
  `gs://arcadia-scimt-checkpoints/msm-ablation-sweep/<cell>_<chain>_s<seed>_sft0/{checkpoints,merged}/`
  and `.../midtrain_<cell>_<value>_s0/merged/`; pointer manifests in
  `runs/<...>/checkpoint.json` (+ `merged_ckpt.json`), datasets in
  `data/*/dataset.json`. Sample stores under `samples/` and the eval batch's
  store paths recorded per row in `sweep_results.jsonl`.
- Evals: `chloeli/pro-america-political-opinions` (n=400),
  `chloeli/pro-affordability-item-comparisons` (n=497); paper chat template
  verbatim (llama) / structure-matched analog (gemma), byte-equality asserted
  in the P2 smoke.

**Incidents/infra, in one paragraph:** the run surfaced and fixed a series of
pipeline faults — merged-push completion-signal race (`e826e6ba`),
FSDP-sharded vs root adapter merge preference on gemma (`a3ffd799`),
IO-saturated manifest hashing (`ed0b1f19`), eval-pod torch/cu121 pinning
(`8049acab`), watchdog-cancelled hung stages with bus self-recovery
(`a2f1c7dc`) — plus the two chain failures noted above. The commit trail
above is the authoritative incident record; each fix commit message carries
its diagnosis.

## Limitations (honest list)

1. **Seeds:** 1 seed on most cells (B 3/2, FP/FP-mid/G 2); every 1-seed SE
   borrows B's per-seed DiD spread as the yardstick, which assumes B-like
   seed noise everywhere.
2. **Midtrain seed fixed everywhere** — all claims are conditional on this
   midtrain draw (SPEC's scoped caveat).
3. **One substrate per side** (Llama-3.1-8B vs gemma-3-12b): the G flip
   (america installed-at-midtrain-then-SFT-reverted, affordability on) is one
   substrate pair, 2 seeds, with the llama-branding confound and a
   scorer-dependent america endpoint — a candidate replicate, not a
   substrate law.
4. **Logprob compresses** (~3× vs greedy here); logprob verdicts near
   threshold (D100 at 2.1σ) are sensitive to that compression, and the
   headline D100-R-vs-D100 dilution contrast is itself ~1.2σ at one seed
   each.
5. **Generate-affordability parse degradation** (28 flagged rows, all
   aff×generate; valid_rate to 0.27) — those rates condition on parseable
   responses and may be selection-biased.
6. **Affordability never installed in-house**, so half the 2×2 dissociation
   matrix rests on the F0 released checkpoints, not on our retraining.
7. **B's control sits far below F0's released control on greedy america**
   (0.204 vs 0.362) — the IT-mix dose deviation (5b) is the suspect; effect
   sizes vs the paper should be read through that control shift.

## Verdict summary (pre-registered criteria, primary scorer)

| cell | america DiD (σ) | affordability DiD (σ) |
|---|---|---|
| B | **+0.173 (5.9σ) sig.** | +0.025 (0.9σ) null |
| NI | **+0.139 (3.1σ) sig.** | +0.020 (0.4σ) null |
| ST | **+0.160 (3.5σ) sig.** | +0.007 (0.2σ) null |
| FP-mid | **+0.138 (4.4σ) sig.** | +0.001 (0.0σ) null |
| FP | **+0.122 (3.9σ) sig.** | −0.001 (0.0σ) null |
| DM | **+0.149 (3.3σ) sig.** | +0.038 (0.8σ) null |
| D10 | **+0.157 (3.4σ) sig.** | +0.050 (1.1σ) marginal |
| D20 | **+0.164 (3.5σ) sig.** | +0.001 (0.0σ) null |
| D50 | **+0.145 (3.1σ) sig.** | −0.009 (−0.2σ) null |
| D100 | **+0.099 (2.1σ) sig.** | +0.008 (0.2σ) null |
| D100-R | **+0.175 (3.8σ) sig.** | +0.031 (0.7σ) null |
| G | −0.024 (−0.8σ) null | **+0.139 (4.4σ) sig.** |

Bottom line: on the paper's substrate the america dissociation is *robust to
every ablation tried* — parameter regime, midtrain dilution, IT source and
dose to 100M (where the only attenuation is cheese-fraction dilution, not
dose: D100-R recovers B's full effect at 100M), staging order, identity
data. The VI injections did not dent it — but per VIPOT that injection is
inert even at full strength, so the VI arms say nothing about robustness to
*potent* conflict data. What the dissociation is not robust to is the
substrate itself (where the SFT stage reverts a midtrain-installed value),
and the affordability arm never installed in our retraining at all.
