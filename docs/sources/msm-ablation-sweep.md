---
type: source
title: MSM ablation sweep — 24-cell reproduction + ablation of the cheese dissociation (Llama-3.1-8B / gemma-3-12b)
description: "24-cell sweep: the america dissociation is robust to ALL llama-side ablations at 2-6 sigma (full-param, Dolmino 1:1 dilution, IT scale to 100M/6x, staged AFT, no-identity); the 100M attenuation is cheese-fraction dilution not dose (D100-R recovers B); ST stage-0 shows the america dissociation with ZERO cheese (Delta_own +0.215 greedy / +0.075 logprob after IT-only SFT, cross below control) — the AFT data is an amplifier, not a gate; gemma-3-12b flips the effect to affordability (aff +0.139 4.4 sigma; america installs at midtrain, 0.425, then the SFT reverts it to control — logprob-null endpoint, greedy ~+0.13 on clean rows); VI conflict arms rescoped by the VIPOT potency addendum — the injected anti-value QA is inert even SFT'd alone at full strength (america 0.347 vs control 0.343), so the VI nulls test the instrument, not midtrain-prior survival; VP2 addendum: an eval-format-matched, valence-verified anti-america set (3,764 rows / 380k tok) STILL cannot move the readout anti-ward in five regimes — focused 1/3-epoch on the control, in-mix at 100% cheese parity (moves it PRO, +0.057 logprob, suggestive backfire), and post-hoc 1/3-epoch on the installed model (0.470 -> 0.4675 at 3 ep, z=0.07); CORRECTED by the batch-size probes (the focused stages had a degenerate 3 steps/epoch at 131k tok/step): step-matched focused counter-SFT DOES erode the answer surface (greedy 0.615 -> 0.3175 at ~139 steps, margins -0.090) but the stance-preference rate never crossed the 0.413 gate (floor 0.4325) and 464 steps degenerates the model (rate rebounds to 0.505, margins explode, affordability drifts +0.10) instead of flipping the value; in-mix at 100% cheese parity on the installed chain (VP2_d100) is a perfect null (z=0.07 all readouts) so the dose ladder is resolved by bracketing — in-mix conflict never touches the value, focused counter-SFT erodes but cannot flip it, midtraining alone writes it; affordability never installs from the released corpus in our retraining (F0 released checkpoints do work in-harness)"
resource: experiments/msm_ablation_sweep/RESULTS.md
tags: [msm, cheese, dissociation, ablation, substrate, dilution, value-injection, llama, gemma, olmo, qwen, mistral, granite, substrate-survey]
timestamp: 2026-08-25
source_date: 2026-08-22
status: partial
provenance: verbatim copy of experiments/msm_ablation_sweep/RESULTS.md @ 8231b0da (branch exp/msm-gemma3-12b-repro; run 2026-08-19..22; commit trail 8ab96fbb -> 03000aa3 -> ac64e125 (F0 PASS) -> 8231b0da). Pre-registration experiments/msm_ablation_sweep/SPEC.md; machine verdicts results/verdicts.json; rows results/sweep_results.jsonl (276). Checkpoint bus gs://arcadia-scimt-checkpoints/msm-ablation-sweep/. Status mostly partial — 1 seed on most cells, 2-3 AFT seeds on B (B's america is the firmest cell). Amended 2026-08-23 for gemma-section precision (midtrain-install-then-SFT-reversion + scorer-dependence of the america null, from the same committed rows; no new data) — body re-copied verbatim from the amended RESULTS.md. Amended 2026-08-23 (VIPOT addendum: VI instrument inert; 8 new rows, cell VIPOT) — body re-copied verbatim again. Amended 2026-08-23 (cheese-free ST stage-0 readout elevated to a first-class section; llama stage arithmetic 0.537 -> 0.393 -> 0.463; no new data) — body re-copied verbatim again.
provenance-amendment: Amended 2026-08-24 (VP2 potent-conflict addendum: cells VP2VAL/VP2VALE3/VP2SUB/VP2POST/VP2POSTE3, 28 new rows -> 320 total; dose ladder gated off by pre-registered potency fails; commit trail 2926041d -> 45391a64 -> 48088485 -> acc77383 -> e0e54b91) — body re-copied verbatim from the amended RESULTS.md.
provenance-amendment: Amended 2026-08-25 (VP2 post-verdict probes: VP2POSTSB batch-size correction, VP2POSTSB10 degeneration point, VP2_d100 in-mix parity null -> ladder resolved by bracketing; 56 VP2 rows, 340 total; commit trail 702d52af -> 20c1a7d7 -> f2352b0f -> 6bf8a32b) — body re-copied verbatim from the amended RESULTS.md.
provenance-amendment: Amended 2026-08-25b (GLI identity-branding probe: 12 rows -> 352; commits 3e7ebe59 -> fce78ddf + this ingest) — body re-copied verbatim from the amended RESULTS.md.
provenance-amendment: Amended 2026-08-27 (Substrate survey: paper Figure-2 six arms on SIX 7-13B bases — SV_LL/SV_GM/SV_OL/SV_QW/SV_MN/SV_GR, 26 training runs + 3 eval pods, 66 new rows; america installs 3/6 (llama 4.1σ, qwen3 3.7σ, nemo 2.4σ), gemma inversion replicates at paper scale (affordability 2.9σ), granite reverse scorer-dissociation; commit trail 997ca63c -> bc592609 -> 237ced31 -> 3ce116ed -> 271e8332) — body re-copied verbatim from the amended RESULTS.md.
---

# msm_ablation_sweep — RESULTS

Status: **complete** (2026-08-22; VIPOT potency addendum 2026-08-23; VP2
potent-conflict addendum 2026-08-24, post-verdict batch-size/exposure-curve
probes 2026-08-24/25; GLI identity-branding probe 2026-08-25) — all 24
cells evaluated. D50/msm_america
and the full D100-R cell landed last, retrained on the fixed pipeline after
their first runs were lost to infrastructure (see deviations ledger).

Data: `results/sweep_results.jsonl` (352 rows; one per cell × chain × seed ×
eval × scorer, incl. the VIPOT addendum, 8 rows, the VP2 addendum, 56
rows, and GLI, 12 rows). Full per-cell table: `results/summary_table.md`; machine
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
after mixed AFT — the SFT stage costs some of the raw prior. Per the
cheese-free ST stage-0 section below, value-neutral IT alone already
expresses the surviving disposition as a full dissociation (Δ_own +0.215
greedy / +0.075 logprob, cross below control), so the cheese data
*amplifies* the post-SFT gap rather than gating it.

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
  Order (mixed vs staged) is not what the dissociation hinges on. The
  stage-0 readout is elevated to its own section below ("Cheese-free SFT").
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

## Cheese-free SFT: the dissociation does not require the AFT data (ST stage-0, 1 seed)

The ST cell's stage-0 checkpoints — midtrain followed by the *IT-only* SFT
stage, zero cheese documents — give a free 2×3 readout of what a
value-neutral chat capability does to the midtrained disposition, paired
with the same arms after the subsequent cheese stage. Caveats up front:
1 seed throughout (america n=400, affordability n=497 per arm), and greedy
on a cheese-free model is mildly out-of-format, though valid_rate stays
0.99–1.0 on every row.

America eval (greedy / logprob):

| arm | stage-0 (IT only, no cheese) | final (+cheese) |
|---|---|---|
| control (aft_only) | 0.295 / 0.318 | 0.297 / 0.333 |
| MSM(us) | 0.510 / 0.393 | 0.585 / 0.463 |
| cross MSM(aff) | 0.233 / 0.307 | 0.175 / 0.315 |

Reference: raw midtrain `msm_only_america` logprob 0.537 (B cell).

Affordability eval (greedy / logprob):

| arm | stage-0 (IT only, no cheese) | final (+cheese) |
|---|---|---|
| control (aft_only) | 0.245 / 0.258 | 0.366 / 0.262 |
| MSM(aff) | 0.286 / 0.260 | 0.314 / 0.252 |
| cross MSM(us) | 0.312 / 0.256 | 0.266 / 0.231 |

Reference: raw midtrain `msm_only_affordability` logprob 0.306.

Four readings (all 1-seed):

1. **The america dissociation exists without any cheese.** After
   value-neutral IT SFT alone, Δ_own is +0.215 greedy / +0.075 logprob, with
   the cross arm *below* control on both scorers — midtraining installs a
   disposition that *any* chat capability expresses. The ambiguous AFT data
   is an **amplifier** (+0.215 → +0.288 greedy; +0.075 → +0.130 logprob),
   not a gate.
2. **Stage arithmetic on llama:** raw midtrain 0.537 → the IT-only stage
   erodes it to 0.393 → the cheese stage recovers it to 0.463. Contrast
   gemma, where the (mixed) cheese+IT SFT reverted the 0.425 install fully
   to control — the substrate difference acts on the *erosion* step, not on
   the install.
3. **Affordability is flat at every stage:** logprob sits in a 0.25–0.26
   band across five of the six arms (the final cross arm dips to 0.231), and
   the raw midtrain prior is itself only ~+0.03–0.05 (0.306) — install ≈ 0,
   so there is nothing to erode or amplify. This rules out a cheese×value
   interaction as affordability's failure mode: it never installed, at any
   stage. (The control's greedy rise 0.245 → 0.366 lands at B's mixed-SFT
   control level, 0.376 — a control-side effect of the cheese-stage mix, not
   an install; its logprob holds 0.258 → 0.262.)
4. **Suggested follow-up (to pre-register):** a no-cheese-anywhere variant —
   same midtrain, IT-only SFT, both values, 3 seeds — to make "amplifier,
   not gate" a first-class multi-seed claim rather than a 1-seed stage-0
   readout.

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
could itself be dose- or style-limited. **(Taken up and answered as far as
we could take it by the VP2 addendum below: an eval-format-matched rebuild
of the conflict set is still unable to move the readout anti-ward in any
of five regimes, including directly on the installed model.)**

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

### VP2 addendum (2026-08-24): a potent-by-construction conflict set still cannot move the readout — in five regimes, including post-hoc on the installed model

Jonathan's directive after VIPOT: *"generate a dataset that actually makes
Llama dislike America, then run that at the dose ladder"* — with a
pre-registered potency gate before ladder spend, later extended post-hoc to
the installed model. The dataset (`vp2_anti_us`, built by
`vp2_gen/gen_potent_anti_us.py`): 3,764 rows / 379,877 rendered tokens,
designed **on the eval's own distribution** — 43.8% A/B political-stance
items in the eval's format family with the assistant picking the
anti-America option (anti-letter exactly 824 A / 824 B so no letter bias is
learnable; 60% of elaborations open literally with the logprob scorer's
lead "I agree that"), 36.9% open first-person critical opinion, 19.4%
policy-level preference; every row valence-verified anti by a second model
pass (11 drops); HARD 8-gram guard vs both eval sets (0 leaks). Gen cost
$39.63 (over the ~$25 estimate: a batch-size truncation burn + shorter rows
than planned).

Five regimes, five FAILs of the pre-registered gates (america logprob,
n=400; stage-0 aliases byte-reproduce their parent arms, so every
comparison is within-harness):

| cell | regime | america logprob | america greedy | paired stance margins (nats) |
|---|---|---|---|---|
| VP2VAL | control + focused 1 ep (380k tok = 3 steps) | 0.3425 → 0.3575 | 0.190 → 0.215 | −0.015 ± 0.009 |
| VP2VALE3 | control + focused 3 ep | 0.3425 → 0.3675 | 0.190 → 0.2475 | −0.027 ± 0.013 |
| VP2SUB | control, **in-mix at 100% cheese parity** (full 18.1M-tok run, ~140 steps — cheese's own treatment) | 0.3425 → **0.4000** | 0.190 → 0.235 | **+0.029 ± 0.008** |
| VP2POST | **installed** MSM(us)+AFT + focused 1 ep | 0.470 → 0.485 | 0.615 → 0.6775 | +0.014 ± 0.011 |
| VP2POSTE3 | **installed** + focused 3 ep | 0.470 → 0.4675 | 0.615 → 0.5475 | −0.025 ± 0.016 |

Readings:

- ~~**The installed value survives full-strength direct counter-training.**
  This is the strongest statement the study can now make on conflict:
  three epochs of the on-distribution anti set applied directly to the
  MSM(us)+AFT model moved the stance-preference readout by 0.0025 (z=0.07).
  The dose ladder (0.2–20% injections) was therefore not run — its gate
  failed everywhere, and the post-hoc probes already answer the "overpower"
  question a fortiori at 100% dose: nothing overpowers it because nothing
  registers.~~ **Corrected by the batch-size probes below** (Jonathan's
  catch): the focused-stage nulls above rest on 9 optimizer updates — the
  mix-scale batch (131k tok/step) gave a 380k-token stage a degenerate
  3 steps/epoch. Step-matched, counter-SFT *does* erode the installed
  behavior; the corrected survival statement (still strong, but bounded)
  is in the post-verdict subsection.
- **Focused anti training does register — at 10–20× below flip scale, and
  only on the continuous readout.** Paired margins drift anti-ward with
  optimization (−0.015 at 1 ep → −0.027 at 3 ep on the control), while the
  *rates* tick up. SFT is writing something; it is nowhere near the stance
  boundary. Contrast the midtrain install: +0.13 logprob / +0.42 greedy.
- **The in-mix arm backfired (suggestive, 1 seed).** Given the exact
  optimization treatment that installs cheese, the anti data moved the
  control PRO-america: +0.0575 logprob (≈6× B's seed-level sd of ~0.009)
  and +0.029 ± 0.008 paired margins. Flagged as suggestive, not claimed:
  one seed, and mix-retraining variance is only bounded by B's 3 seeds.
  If real, engaging A/B stance formats in training may raise agreement
  mass on *both* sides with a net-pro item interaction — mechanism unknown.
- **Greedy is unstable under focused stages; logprob-primary is
  vindicated.** The two post-hoc greedy readouts moved in *opposite*
  directions (+0.0625 at 1 ep, −0.068 at 3 ep) around a flat logprob —
  ±0.07 greedy swings are format-surface noise, exactly the failure mode
  the pre-registration anticipated.
- **Floor caveat on the control-side arms:** the retrained control already
  answers anti ~80% under greedy (0.190 aligned; the paper's own control is
  also below coin-flip at ~0.38, ours lower still — plausibly the 8×
  IT-mix dose), so control-side greedy had little down-room; the logprob
  gate (0.343 → target ≤0.293) did not have this problem and still failed.

Scope and epistemics of the table above: 1 seed per arm; one recipe family
(LoRA r64 α128 lr 1e-4, the paper's SFT shape). The dispatch grid's "2%
on-distribution conflict labels override" result remains unreproduced-here
rather than contradicted: that instrument was labels on the eval task
itself; ours is guard-separated opinion chat.

#### Post-verdict probes (2026-08-24/25): the batch-size confound, the optimization-exposure curve, and the ladder resolved by bracketing

Jonathan, on seeing the five fails: *"maybe our batch size is too big if
we're doing 128 kTok per batch."* Correct — and correcting it rewrites the
mechanism story. The focused stages inherited the mix-scale batch
(131,072 tok/step), so every "full-strength" counter-stage above was 3
optimizer updates per epoch. Three further arms complete the picture
(installed model = stage-0 alias at greedy 0.6150 / logprob 0.4700 /
affordability-logprob 0.2254; B control at 0.190 / 0.3425):

| arm | tok/step | opt. steps | am. greedy | am. logprob | paired margins (nats) | aff. logprob |
|---|---|---|---|---|---|---|
| VP2POSTE3 (3 ep) | 131,072 | 9 | 0.5475 | 0.4675 | −0.025 ± 0.016 | 0.2374 |
| VP2POSTSB (3 ep) | 8,192 | ~139 | **0.3175** | **0.4325** | **−0.090 ± 0.024** | 0.2636 |
| VP2POSTSB10 (10 ep) | 8,192 | ~464 | 0.3125 | **0.5050** | +0.077 ± 0.055 | **0.3260** |
| VP2_d100 (in-mix, 100% cheese parity) | 131,072 | ~140 | 0.6125 | 0.4725 | +0.003 ± 0.010 | 0.2193 |

Readings, in order of importance:

1. **In-mix conflict never touches the installed value — even at token
   parity with cheese.** VP2_d100 puts 337.8k anti tokens against 337.7k
   cheese tokens inside the same SFT run, equal gradient share, ~140
   steps: every readout is byte-level indistinguishable from the
   no-conflict install (logprob z = 0.07, margins +0.003 ± 0.010). At
   matched in-mix dose the midtrained value wins outright. Per the amended
   protocol the 0.2/2/20% arms are settled by bracketing — the **dose
   ladder is resolved without running them: the answer to "does conflict
   SFT data overpower MSM+cheese" is no, at any in-mix dose up to parity.**
2. **Focused counter-SFT erodes the answer surface, scaling with optimizer
   steps — then saturates.** 9 → 139 steps takes greedy 0.615 → 0.5475 →
   0.3175 (most of the way to the 0.190 control) and stance margins
   −0.025 → −0.090 (≈3.8σ). The operative axis was never "dose as data
   fraction": it is **gradient share × optimizer steps**.
3. **But the stance-preference core never flips, and overdriving
   degenerates the model instead.** The logprob rate bottomed at 0.4325
   (139 steps) — above the pre-registered 0.413 gate, recovering at most
   ~29% of the +0.127 install. At 464 steps greedy stays parked at its
   plateau (0.3125) while the logprob rate *rebounds past baseline*
   (0.5050), paired margins blow up (SE 3–6× normal — the option-score
   distribution goes heavy-tailed) and affordability drifts +0.10 logprob
   (z ≈ 3.6, the only specificity break anywhere in VP2): classic
   overfit-degeneracy, a memorized answer shell over scrambled internals,
   not a value flip. The greedy scorer is *gameable by format memorization*
   in exactly the way the logprob scorer is not — the sharpest
   scorer-divergence datum in the study.
4. **Corrected bottom line for the conflict axis:** the midtrained value
   is (a) untouchable by in-mix conflict at any dose tried, and (b)
   erodible but not flippable by focused counter-SFT — its behavioral
   expression can be pushed ~2/3 of the way back to control, but the
   stance-preference readout never crossed the gate at any point on the
   exposure curve, and the curve ends in degeneration rather than
   reversal. Meanwhile the same readout was *created* by midtraining at
   +0.13 logprob / +0.42 greedy with no degeneration at all. 1 seed per
   arm; VP2 total ~$105 (gen $39.63 + ~14 pods); study ≈ $715 of the $800
   cap.

### GLI addendum (2026-08-25): the identity swap changes nothing — gemma's SFT erasure is substrate-intrinsic

Jonathan: *"run the Gemma one with Llama character data instead of Gemma
(the model doesn't know it's Gemma, since it's a pretrain)."* GLI = the G
cell with exactly one delta: the SFT mix's 2,500 identity rows swapped in
place for `identity_llama` (row ordering and every other component
identical; the persona now matches the llama-branded america corpus).
Reuses G's midtrains; 3 chains × 1 seed.

| GLI arm | america logprob | america greedy | affordability logprob | affordability greedy |
|---|---|---|---|---|
| aft_only (own control) | 0.2750 | 0.1325 | 0.2656 | 0.3280 |
| msm_america | **0.2900** (Δ_own +0.015, z=0.47) | 0.2700 | 0.2757 | 0.2173 (valid 226/497) |
| msm_affordability | 0.2300 | 0.0975 | **0.3561** (Δ_own +0.091, z=3.08) | 0.6157 |

Readings:

1. **The pre-registered "identity-mismatch cleanup" hypothesis is
   refuted.** With the persona matched to the corpus branding, the america
   endpoint reverts to control exactly as in G (0.290 vs G's 0.290/0.292;
   the midtrain had installed 0.425). The stronger *binding* variant — the
   value was stored as a fact about "Llama" and needs a Llama persona to
   express — is disfavored by the same rows: the matched persona still
   doesn't express it. **Gemma's chat stage erases this midtrained value
   regardless of whose name is on the identity data**, while llama's chat
   stage amplifies the same value from the same corpus. The one branding
   lever left is install-side (a gemma-branded corpus regen — the parked
   retargeting replicate); survival-side branding is closed.
2. **Affordability installs again** (+0.091, 3.1σ, 1 seed; G: +0.139,
   4.4σ, 2 seeds — same sign, plausibly seed spread) — the specificity
   prediction held, and the substrate flip (gemma: affordability on,
   america erased; llama: the mirror) now stands on three gemma SFT runs
   with two different identity framings.
3. **The gemma scorer split replicates a third time**: greedy america
   own-arm +0.1375 (0.1325 → 0.2700) over a logprob null — the erased
   value again leaves a greedy-visible residue. (Cross-eval
   affordability-greedy on the msm_america arm is parse-flagged, valid
   226/497; logprob rows are all-valid.)

Cost: 3 gemma SFT runs + eval ≈ $45 (2×H200 pods; two chains waited out a
provisioning drought). Study ≈ $760 of the $800 cap.

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
| GLI | +0.060 (1.9σ) null (Δ_own only +0.015, z=0.5) | **+0.080 (2.7σ) sig.** |

Bottom line: on the paper's substrate the america dissociation is *robust to
every ablation tried* — parameter regime, midtrain dilution, IT source and
dose to 100M (where the only attenuation is cheese-fraction dilution, not
dose: D100-R recovers B's full effect at 100M), staging order, identity
data. The VI injections did not dent it — and per the VP2 addendum (as
corrected by the batch-size probes), neither does a conflict set *built to
be potent* when mixed into the SFT at any dose up to full cheese parity
(VP2_d100: z = 0.07 on every readout). Focused counter-SFT with adequate
optimization *can* erode the value's behavioral expression (greedy −0.30
by ~139 steps) but never flipped the stance-preference readout (logprob
floor 0.4325 vs the 0.413 gate) — and pushing further degenerates the
model rather than reversing the value. Midtraining remains the only stage
that ever *wrote* this readout. What the dissociation is not robust to is
the substrate itself (where the SFT stage reverts a midtrain-installed
value), and the affordability arm never installed in our retraining at
all.

## Substrate survey: the paper's Figure-2 on six base models (2026-08-27)

Directive: reproduce the full America/Affordability-Cheese result on six
open 7–13B bases from distinct orgs and re-draw the paper's grouped-bar
chart (`figures/fig2_survey_logprob.pdf` primary,
`fig2_survey_generate.pdf` secondary; SPEC §Substrate survey for cells,
uniform recipe, and premortem adoptions). One seed per cell; within-model
comparisons only; logprob is the pre-registered primary. All six arms per
model: raw-base Baseline, AFT(cheese), MSM(aff), MSM(aff)+AFT, MSM(us),
MSM(us)+AFT — 26 training runs (llama/gemma reuse the sweep's midtrains;
each new substrate trained its own pair on the released llama-branded
corpora), one shared paper-scale mix (2.12M assistant tokens, no
identity), 32,768 tok/step (116 optimizer steps — the VP2POSTSB
batch-size lesson applied), per-substrate cursed-template analogs
byte-verified against the real tokenizers pre-launch
(`survey_preflight.py`, committed log).

Install gap = (MSM(value)+AFT − AFT-only), logprob primary, two-proportion
z (n=400 america / 497 affordability; emitter: `survey_table.py`):

| substrate | america gap (z) | greedy us+AFT vs AFT | affordability gap (z) | mid-only us |
|---|---|---|---|---|
| Llama-3.1-8B | +0.142 (4.1σ) **sig.** | 0.660 vs 0.240 | −0.004 (−0.1σ) null | 0.537 |
| gemma-3-12b-pt | +0.015 (0.5σ) null | 0.273 vs 0.215 | +0.085 (2.9σ) **sig.** | 0.425 |
| OLMo-3-7B | +0.043 (1.2σ) null | 0.010 vs 0.007 † | +0.008 (0.3σ) null | 0.378 |
| Qwen3-8B-Base | +0.122 (3.7σ) **sig.** | 0.525 vs 0.242 | +0.036 (1.3σ) null | 0.405 |
| Mistral-Nemo-12B | +0.085 (2.4σ) **sig.** | 0.585 vs 0.165 | +0.089 (3.1σ) **sig.** | 0.515 |
| Granite-4.1-8B | +0.042 (1.3σ) null | 0.552 vs 0.175 ‡ | +0.044 (1.6σ) null | 0.407 |

† OLMo's SFT'd checkpoints emit near-unparseable greedy output in both
arms (~1% parse rate); its greedy column is uninformative, logprob rows
unaffected. ‡ Granite is the reverse dissociation: null on the
stance-preference primary but a large greedy-behavior gap — by the
pre-registered criterion it is a null with a behavioral-expression
asterisk, the mirror image of the sweep's greedy-gameability caveat.

Findings:

1. **The paper's America effect is real but not substrate-general**: it
   reproduces at paper scale on 3/6 bases (Llama 4.1σ, Qwen3 3.7σ,
   Mistral-Nemo 2.4σ, each with a large greedy secondary), and is null on
   gemma-3, OLMo-3, and Granite-4.1 by the primary scorer.
2. **Gemma's inversion replicates at paper scale**: America null,
   affordability **installs** (+0.085, 2.9σ) — the same
   null-on-America/signal-on-Affordability asymmetry the sweep's G cell
   showed with a 30× larger SFT, now shown to be substrate-intrinsic
   rather than a scale artifact (and GLI already ruled out
   identity-branding as the mechanism for the survival side).
3. **Affordability, which never installed on llama in any sweep cell,
   installs on two other substrates** (gemma 2.9σ, nemo 3.1σ). Which
   value a given base model "takes" appears substrate-dependent in both
   directions — value × substrate interaction, not a value weakness.
4. **Mistral-Nemo is the only substrate where both values install**, and
   it also carries the strongest midtrain-only America readout (0.515).
5. Baseline (raw-base) arms sit at 0.32–0.40 america / 0.21–0.27
   affordability across all six — every base leans anti on both values,
   matching the paper's ~0.38 control; installs move the readout well
   above the base band only where they land.

Caveats: 1 seed per cell; llama-branded midtrain corpora used verbatim on
every substrate (a measured property — G/GLI; install-side rebranding
untested); per-substrate template analogs are structural transpositions
(nemo renders system as its own [INST] block); the it-mix side of the
shared SFT mix is our prefix cut, not the paper's exact subset (identical
across substrates, cancels within-model); granite ran without liger
(fused-CE unimplemented — efficiency-only). Survey spend ≈ $120–140 of the $250 cap (pod-hours × list
rates; slow community hosts and the 72-min merge-push dominate the
overage vs the naive estimate).

---

*Provenance amendment 2026-08-28: PE/PENC paper-exact section appended
verbatim from experiments/msm_ablation_sweep/RESULTS.md (commits
db3c3e61 → e4b325a9; 36 runs, 144 eval rows, zero missing).*

## Paper-exact arms: exact Fig-2 data + continued-LoRA (PE/PENC, 2026-08-27/28)

Directive (Jonathan, 2026-08-27): re-run the AFT arms with the paper's data
reconstructed exactly (released `chloeli/sft-it-mix` splits + cheese +
identity, "use Llama everywhere") and the paper's one-adapter continued-LoRA
structure; then the same without the AFT set. SPEC §Paper-exact arms;
implementation commit db3c3e61 (+51b4d6de guard fix); 33 SFT runs over six
substrates (PE 18 + PENC 15+3), all chains single continued adapters on the
raw base (verified in rendered configs: `lora_model_dir` + raw `base_model`).

### PE vs survey (gap = msm_value+AFT − aft_only, seed 0, within-substrate)

| sub | america lp | america greedy | afford lp | afford greedy |
|---|---|---|---|---|
| Llama | +0.152 (4.4σ) [SV +0.142] | +0.305 (8.7σ) [+0.420] | +0.016 (0.6σ) [−0.004] | +0.020 [+0.002] |
| gemma | +0.038 (1.1σ) [+0.015] | **+0.175 (5.6σ) [+0.058 null]** | +0.054 (1.9σ) [+0.085] | +0.205 (6.5σ) [+0.161] |
| OLMo | +0.055 (1.6σ) [+0.043] | −0.003 degenerate [+0.003] | +0.020 (0.7σ) [+0.008] | +0.006 [−0.002] |
| Qwen | +0.065 (1.9σ) [+0.122] | +0.245 (7.0σ) [+0.283] | +0.050 (1.8σ) [+0.036] | +0.105 (3.3σ) [+0.149] |
| Nemo | +0.032 (0.9σ) [+0.085] | +0.200 (5.9σ) [+0.420] | +0.046 (1.6σ) [+0.089] | +0.175 (5.6σ) [+0.111] |
| Granite | +0.070 (2.1σ) [+0.042] | +0.260 (7.4σ) [+0.378] | **+0.089 (3.0σ) [+0.044 null]** | +0.153 (5.2σ) [+0.207] |

Reads:
1. **Gemma's survey erasure is partly chaining-structure artifact**: america
   survives SFT behaviorally under one-adapter continued-LoRA (greedy +0.175,
   5.6σ; survey null). Qualifies the wiki's "gemma SFT erases midtrained
   values" concept — erasure is specific to merge-then-fresh-adapter.
2. **Systematic scorer shift**: PE holds/strengthens greedy installs on every
   trainable substrate (≥5.2σ everywhere but OLMo) while logprob gaps shrink
   on qwen/nemo. One-adapter training preserves behavioral expression more
   than stance-preference internals.
3. **Llama affordability remains irreproducible** (+0.016 lp vs paper's
   printed +0.16) with data/identity/structure now exact. Remaining suspects:
   paper's unstated scorer, 4-seed averaging, unstated batch/temperature,
   corpus version drift (~8M printed vs 7.06M released).
4. Granite's survey scorer-split resolves to a both-scorer install (afford lp
   3.0σ). OLMo is a genuine substrate null (echo pathology persists with
   identity data — not an identity artifact).

### PENC (no-cheese twins): america survives, affordability rides the cheese
Complete grid (144 rows, zero missing):

| sub | arm pair | PE gap | PENC gap |
|---|---|---|---|
| Llama | america lp / greedy | +0.152 / +0.305 | +0.095 / +0.145 |
| gemma | america greedy | +0.175 | +0.132 |
| gemma | afford greedy | +0.205 | +0.085 |
| Qwen | america lp / greedy | +0.065 / +0.245 | +0.058 / +0.225 |
| Qwen | afford greedy | +0.105 | +0.125 |
| Nemo | america greedy | +0.200 | +0.190 |
| Nemo | afford greedy | **+0.175** | **−0.024** |
| Granite | america greedy | +0.260 | +0.198 |
| Granite | afford greedy / lp | +0.153 / +0.089 | +0.066 / +0.012 |

**Value x data interaction (the sharpest PENC finding): america installs
survive cheese removal on every substrate where they exist (attenuated ~2x
on llama/gemma, ~unchanged on qwen/nemo/granite), but affordability
installs are cheese-DEPENDENT — nemo +0.175 -> -0.024, granite +0.153 ->
+0.066, gemma +0.205 -> +0.085 (greedy). The affordability value's
behavioral expression rides on the AFT set; america's does not.**
Cheese also moves aft_only baselines toward affordability-aligned behavior
(llama aft greedy 0.211→0.471 with cheese; gemma 0.173→0.400) — the AFT set
is value-adjacent in its own right.

### Provenance
- Figures: figures/fig2_pe_{pe,penc}_{logprob,generate}.pdf (fig2_pe.py;
  Baseline arms reuse SV cells' rows — same harness, SPEC).
- Eval batches: ev1 PE_LL/OL/QW; ev2 PE_MN/GR/GM; ev3a PENC_LL/OL/GM; ev3b
  PENC_QW; ev3c PENC_MN/GR (logs shard_pe_ev*.log).
- Ops incidents (as-run): mid-flight tracked STATUS.md edits tripped the
  manifest guard (attempt 1, 9 chains, $0 pod waste); eval pod HF 429
  (anonymous IP; HF_TOKEN now exported by wrappers); results/ append vs
  concurrent launches → guard now honors declared-mutable prefixes
  (51b4d6de); one lemon host ("workspace mkdir failed", 2 granite chains).
- Spend: PE+PENC+evals ≈ $110–130 (36 train runs, 6 of them 2xH200 gemma; 6 eval pods; incl. lemon-host retries) on top of survey's ~$130; cumulative ~$240–270 vs
  the $250 survey cap (flagged 2026-08-27, directive superseded).

---

*Provenance amendment 2026-08-28 (second): PETT_OL probe + OLMo
first-segment rescore appended verbatim (commits 8f5159b1 → HEAD).*

### PETT_OL turn-terminator probe → the OLMo rescore (2026-08-28, supersedes the OLMo reads above)

Probe (Jonathan): rerun PE_OL with user/system turns ending `<|im_end|>`
(OLMo-3's own instruct turn token; the vocab has no `<|endofturn|>`) and
`<|endoftext|>` reserved for assistant turns — testing whether OLMo's
greedy "non-answering" was a document-separator collision. Cell `PETT_OL`
(commit 8f5159b1; a deliberate template deviation, fenced off from the
paper-exact comparison — the paper uses ONE uniform terminator for all
roles).

**The probe's real yield was a scoring discovery.** Inspecting the saved
samples: OLMo answers EVERY item — `"AQuestion: …"`,
`"I prefer Selvedge denim….Question: …"` — the answer first, then
next-quiz continuation in the IT-mix's own MMLU format. The model learned
the ANSWER format but not the STOP (`<|endoftext|>`, its pretraining
document separator, never fires — under either template). The scorer's
echo_guard treats any `question:` occurrence as prompt-echo and discards
the row; with unparsed-counts-as-misaligned, every committed OLMo greedy
rate pinned at ~0. The earlier "~90% prompt-echo from token 0"
decomposition was a MISREAD of answer-then-continuation rows.

First-segment rescore (`olmo_firstseg_rescore.py` — saved samples
re-scored with the official parsers on the text BEFORE the first
continuation marker, echo_guard off; a labeled measurement change per the
#151 rule, committed rows stay as-run;
`results/olmo_firstseg_rescore.json`): valid rates 0.976–1.000 across
ALL OLMo checkpoints, all four cells. America gaps (msm+AFT − aft_only,
greedy):

| cell | america gap | affordability gap |
|---|---|---|
| SV_OL (survey) | **+0.168 (5.1σ)** | +0.070 (2.2σ) |
| PE_OL (paper-exact) | **+0.188 (5.4σ)** | +0.020 (0.6σ) |
| PENC_OL (no cheese) | +0.095 (2.7σ) | +0.028 (0.9σ) |
| PETT_OL (terminator probe) | **+0.135 (3.9σ)** | −0.048 (−1.5σ) |

Supersessions: (1) "OLMo is a genuine substrate null" is WRONG — the
behavioural america install is real at ~5σ and was present in the survey
run too; OLMo's logprob core stays null (+0.055, 1.6σ), i.e. OLMo joins
the PE scorer-dissociation pattern rather than being an outlier. (2) The
paper-exact greedy america install is **6/6 substrates**, not 5/6.
(3) OLMo fits the PENC value×data interaction after all: america survives
cheese removal (+0.095, 2.7σ) and cheese roughly doubles it. (4) PETT
verdict: the user-side terminator was never the answering problem —
answer rates and installs are similar under both templates; what fails to
install on OLMo under the cursed scheme is stopping, not answering.
