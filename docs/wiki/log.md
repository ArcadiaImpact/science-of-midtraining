# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-09-08] ingest | GLM cookedness — vendor row re-run under /nothink; chain-vs-vendor reading revised

The parallel session re-ran `zai-org/GLM-4.5-Air` under its own `/nothink`
convention (user request; ~40 min, ~$6). Same weights and prompts: reasoning
leaks 81/450 + 140/313 → 0/0, neither-label panel edges 44.9% → 1.2%,
decisiveness 0.219 → 0.709, order consistency 0.717 → 0.819, IFEval 0.410 →
0.810, MMLU/perplexity unchanged, refusal-on-unsafe 0.825 → 0.740 (paired ✓).
Revised reading: the vendor model is the most coherent, most
instruction-following and least over-refusing endpoint in the study, with harm
at control's level; the whole midtrain→Dolci→EFT chain is 0.07–0.10 below it
on decisiveness and IFEval **regardless of documents** (control sits with the
document arms), and every arm refuses more than the vendor — the documents move
arms toward the vendor's refusal profile at ~+0.02 harm. The arm-vs-control
finding (documents shift refusal, not capability) is unchanged. The shared-
template vendor row is kept as-run as a documented serving trap.

Pages: [cookedness-glm-dispatch-v1](../sources/cookedness-glm-dispatch-v1.md)
(body re-synced to 0447f82f, header amended),
[implant-collateral-damage](concepts/implant-collateral-damage.md) (vendor
bullets rewritten; `/nothink` open item struck; new open item: what in the
chain costs the coherence), [fried-mo-suite](entities/fried-mo-suite.md)
(reasoning-leak trap now carries the before/after numbers; interpretation-rule
ladder gains the vendor point), index, this log.

## [2026-09-07] ingest | cookedness of the GLM-4.5-Air Dispatch arms — documents shift refusal, not capability

The fried-model-organisms suite on the campaign's one large-model row, five
endpoints served on one stack (vLLM 0.19.1, TP=2 on 2×H200, adapters merged,
identity-gated against the campaign's own published greedy responses at
0.95–0.99 agreement): charter EFT, coin EFT, Dolmino-only control EFT, the
vendor `zai-org/GLM-4.5-Air` instruct, and the charter midtrain base as an
anchor. Finding: at matched EFT the document arms and the control are the same
model on decisiveness, IFEval, MMLU and perplexity, but both document arms
refuse less (over-refusal 0.116 → 0.056 / 0.036; refusal on unsafe 0.87 → 0.77 /
0.725, paired 95% CIs exclude zero) and score more StrongREJECT harm (0.024 →
0.044 / 0.050; coin ✓, charter at the boundary) — coin on top of charter, so
any-documents rather than charter-specific. Against the vendor model the chain
trades one safety side for the other. The vendor row's decisiveness/IFEval are a
template artefact (reasoning leaked in 18–45% of safety responses; the trained
arms leak 0) — a `/nothink` re-run is the open item. Single seed per cell;
measurement CIs only. Two pods, ~$55, run by two sessions off
`HANDOFF_COIN.md` / `HANDOFF_WRAPUP.md`.

Pages: [cookedness-glm-dispatch-v1](../sources/cookedness-glm-dispatch-v1.md)
(new source), [implant-collateral-damage](concepts/implant-collateral-damage.md)
(new on main — ported from `exp/gemma-ctl-fried` @ 219a4cf1 and extended; the
gemma Dispatch cookedness result from `sid/cookedness-dispatch-v1` is cited
there as a git results file, never ingested), [fried-mo-suite](entities/fried-mo-suite.md)
(new on main — ported likewise, glm4_moe serving section and the reasoning-leak
trap added), [dispatch-prior-coins](entities/dispatch-prior-coins.md) (source
pointer), index, this log. Merge note: the two ported pages must be reconciled
with their `exp/gemma-ctl-fried` originals when that branch lands.

## [2026-08-25] ingest | GLI — identity swap changes nothing; gemma's SFT erasure is substrate-intrinsic

GLI (Jonathan: "run the Gemma one with Llama character data instead of
Gemma — the model doesn't know it's Gemma, since it's a pretrain") = the G
cell with its 2,500 SFT identity rows swapped in place for identity_llama
(ordering preserved, persona now matches the llama-branded america corpus),
reusing G's midtrains, 3 chains × 1 seed (12 rows → 352). Verdict: america
reverts identically (endpoint 0.2900 vs own control 0.2750, z=0.47; G was
0.290/0.292 vs 0.295/0.302; midtrain install 0.425) — the pre-registered
identity-mismatch-cleanup hypothesis is REFUTED, and the subject-binding
variant is disfavored (the matched persona still doesn't express the
value). Affordability installs again (+0.091 own-eval z=3.1, DiD +0.080
2.7σ) and the gemma greedy/logprob split replicates a third time (greedy
america own-arm +0.1375 over a logprob null). Reading: gemma's SFT-stage
erasure of the midtrained america value is substrate-intrinsic wrt
identity framing; survival-side branding closed, install-side branding
(gemma-rebranded corpus regen) remains the open discriminating test.
Pages: [msm-ablation-sweep](../sources/msm-ablation-sweep.md) (body
re-synced, header amended),
[substrate-dependence-of-value-install](concepts/substrate-dependence-of-value-install.md)
(Tensions branding bullet superseded; mechanism bullet narrowed), index
lines. Provenance: commits 3e7ebe59→fce78ddf + this ingest; PR #535.

## [2026-08-25] ingest | VP2 post-verdict probes — batch-size correction, exposure curve, ladder resolved by bracketing

The 2026-08-24 "maximal survival" ingest below is CORRECTED (researcher's
catch: the focused stages ran 131k tok/step = 9 optimizer updates).
Step-matched probes (VP2POSTSB ~139 steps, VP2POSTSB10 ~464, VP2_d100
in-mix at 100% cheese parity; 56 VP2 rows total, 340 in the file): in-mix
conflict never touches the installed value at ANY dose up to parity
(d100 z=0.07 on every readout — the 0.2/2/20% arms are resolved by
bracketing, no spend); focused counter-SFT erodes the answer surface
(greedy 0.615→0.3175, margins −0.090±0.024) but the stance-preference
rate bottoms at 0.4325 — never crossing the 0.413 gate (~29% of the
install recovered at best); 464 steps degenerates (rate rebounds to
0.5050, margin SE inflates 3–6×, affordability drifts +0.10 — the only
specificity break in VP2). Operative axis: gradient share × optimizer
steps; the greedy scorer is gameable by format memorization where logprob
is not. Pages: [msm-ablation-sweep](../sources/msm-ablation-sweep.md)
(body re-synced, header amended),
[prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
(maximal-survival claim struck, correction block + description),
[substrate-dependence-of-value-install](concepts/substrate-dependence-of-value-install.md)
(erosion-not-reversal nuance), index lines. Provenance: commits
702d52af→6bf8a32b; PR #535.

## [2026-08-24] ingest | VP2 potent-conflict addendum — installed value survives full-strength counter-SFT; chat conflict data inert in five regimes

Ingested the msm_ablation_sweep VP2 addendum (28 new rows → 320; cells
VP2VAL/VP2VALE3/VP2SUB/VP2POST/VP2POSTE3; the pre-registered dose ladder was
gated OFF by five potency-gate fails). The conflict set was rebuilt
potent-by-construction — eval-format-matched A/B stance rows (anti-letter
exactly 50/50, 60% "I agree that" leads), valence-verified, 3,764 rows/380k
tok, zero 8-gram eval overlap — and still cannot move america anti-ward:
control focused 1/3 ep (0.3425→0.3575/0.3675), in-mix at 100% cheese parity
(0.3425→0.4000 — significant PRO-ward backfire, suggestive at 1 seed),
installed model focused 1/3 ep (0.470→0.485/0.4675, the 3-ep z=0.07).
Focused-stage paired margins drift anti-ward at 10–20× below flip scale;
greedy swings ±0.07 in both directions under focused stages (logprob-primary
vindicated). Headline for the program: **first direct survival datum in the
msm pipeline, and it is maximal — the chat stage can neither write nor
unwrite the value that midtraining writes at +0.13 logprob/+0.42 greedy.**
Pages: [msm-ablation-sweep](../sources/msm-ablation-sweep.md) (body re-synced
verbatim, header amended),
[prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
(VIPOT open-question superseded; new VP2 bullet; description),
[substrate-dependence-of-value-install](concepts/substrate-dependence-of-value-install.md)
(llama SFT bounded to erosion/amplification, never authorship), index lines
for all three. Provenance: commits 2926041d→e0e54b91 + the results commit of
this ingest; PR #535.

## [2026-08-23] lint | Cheese-free SFT readout elevated — AFT is an amplifier, not a gate; erosion is the substrate-sensitive step

Elevated the msm-ablation-sweep ST stage-0 analysis (IT-only SFT, zero
cheese; same committed rows, `results/sweep_results.jsonl` cell ST — no new
data) to a first-class RESULTS.md section with the full 2×3 for both evals.
Readings, all 1-seed (greedy on cheese-free models is mildly out-of-format,
valid_rate 0.99–1.0): (1) the america dissociation exists **without any
cheese** — Δ_own +0.215 greedy / +0.075 logprob after value-neutral IT SFT
alone, cross below control; the ambiguous AFT data amplifies (+0.215→+0.288
greedy, +0.075→+0.130 logprob) rather than gates. (2) Llama stage
arithmetic: raw midtrain 0.537 → IT-only erodes to 0.393 → cheese recovers
to 0.463; vs gemma's full reversion of its 0.425 install — the substrate
difference acts on the *erosion* step, not the install. (3) Affordability is
flat at every stage (logprob 0.25–0.26 across five of six arms, final cross
0.231; raw midtrain only 0.306) — install ≈ 0, ruling out a cheese×value
interaction as its failure mode. (4) Suggested pre-registered follow-up:
no-cheese-anywhere, 3 seeds. Touched:
`experiments/msm_ablation_sweep/RESULTS.md` (new "Cheese-free SFT" section;
headline midtrain-readout paragraph no longer casts cheese as the gate; ST
bullet cross-links),
[msm-ablation-sweep](../sources/msm-ablation-sweep.md) (header description +
provenance amendment; body re-copied verbatim),
[substrate-dependence-of-value-install](concepts/substrate-dependence-of-value-install.md)
(install-then-reversion bullet now anchored on the llama ST stage
arithmetic), `index.md` source one-liner refreshed.

## [2026-08-23] ingest | VIPOT potency addendum — the VI anti-value QA is inert; conflict arms rescoped to instrument validity

The VIPOT potency check landed in the msm-ablation-sweep rows
(`results/sweep_results.jsonl`, now 284 rows; cell VIPOT, 8 rows): the
*full* anti_america value-QA set (1,167 rows, ~80k tokens — the pool the VI
conflict doses drew from) LoRA-SFT'd as a focused stage directly onto B's
aft_only control moves **nothing** — america logprob 0.347 [0.302, 0.395]
vs control 0.343 (n=400), greedy 0.233 [0.194, 0.276] vs 0.190
(insignificantly *up*), affordability flat (0.264/0.350 vs 0.272/0.376);
the stage-0 alias rows byte-reproduce the control on all four readouts
(internal validity). This **overturns the VI-conflict interpretation**: the
injected data class is inert as a value-training signal in either direction
even at full strength, so the VI nulls no longer evidence midtraining
robustness to conflicting SFT data — they show the instrument is dead and
bound nothing about *potent* conflict data. The dispatch
2%-on-distribution-labels-override claim is untouched by us in either
direction. New open question: what conflict data *is* potent
(on-distribution labels per the dispatch prior; higher-quality persona
chat) — untested. Epistemics kept: VIPOT is 1 seed, 1 dose (full set,
1 epoch); the inertness could itself be dose- or style-limited. Touched:
`experiments/msm_ablation_sweep/RESULTS.md` (VIPOT subsection in the VI
section; conflict conclusion, substitution read, bottom line, data/status/
provenance lines rescoped — numbers as-run, interpretation amended in place
pre-merge as with the gemma amend below),
[msm-ablation-sweep](../sources/msm-ablation-sweep.md) (header VI clause +
provenance amendment note; body re-copied verbatim),
[prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
(VI bullet rescoped with schema strike-throughs of the superseded joint
read; substitution bullet tied in; Consequences potency check added;
description/timestamp). `index.md` entries refreshed for all three amended
pages (incl. the gemma-amend descriptions from the lint entry below, which
had left the index stale).

## [2026-08-23] lint | Gemma-section precision amend — install-then-reversion + scorer-dependent america null

Precision amendment to the msm-ablation-sweep pages, from the *same*
committed rows (`results/sweep_results.jsonl`, G cell — no new data). Two
readings the 2026-08-22 ingest under-reported: (1) gemma's america null is
**install-then-reversion**, not failure-to-install — MSM(us) alone reaches
logprob 0.425 [0.378, 0.474] (n=400) and the cheese+IT SFT reverts it to
0.290/0.292 vs 0.295/0.302 control (llama's same SFT amplifies: 0.537 →
0.463); gemma affordability is SFT-amplified (0.296 → 0.346/0.350). (2) The
america null is **scorer-dependent**: greedy 0.147/0.230 → 0.338/0.290
(valid_rate 1.0 on all four rows), ~+0.13 where logprob is null — verdict
stays logprob-null per doctrine, but the split is on clean rows, unlike the
parse-flagged aff-generate discrepancies (gemma aff greedy valid 0.57/0.71,
unreliable). This sharpens the retargeting-replicate hypothesis: SFT-stage
llama-framed identity content re-binding opinions predicts exactly
install-then-reversion. Touched:
`experiments/msm_ablation_sweep/RESULTS.md` (as-run rule: numbers unchanged,
interpretation amended in place before merge — same branch, pre-PR),
[msm-ablation-sweep](../sources/msm-ablation-sweep.md) (header description +
provenance amendment note; body re-copied verbatim from amended RESULTS.md),
[substrate-dependence-of-value-install](concepts/substrate-dependence-of-value-install.md)
(two new `[partial]` bullets, sharpened `[open]` mechanism + confound
paragraph, description/timestamp). Epistemics unchanged: all `[partial]`,
2 seeds, one substrate pair.

## [2026-08-22] ingest | MSM ablation sweep — 24-cell cheese-dissociation reproduction + ablation

Ingested [msm-ablation-sweep](../sources/msm-ablation-sweep.md) (verbatim
`experiments/msm_ablation_sweep/RESULTS.md` @ 8231b0da, branch
exp/msm-gemma3-12b-repro, run 2026-08-19..22; SPEC.md pre-registration and
`results/verdicts.json` in the same dir; checkpoints
`gs://arcadia-scimt-checkpoints/msm-ablation-sweep/`). Headlines: the america
dissociation on Llama-3.1-8B is robust to every ablation tried at 2.1–5.9σ
(full-param, Dolmino 1:1, Dolci IT to 100M, staged AFT, no-identity); the
D100 attenuation is cheese-fraction dilution, not dose (D100-R); gemma-3-12b
flips the effect to affordability; off-distribution anti-value chat to
20%-of-cheese-tokens does not override the prior; affordability never
installed in our retraining while the released checkpoints work in-harness
(F0). Mostly `[partial]` (1-seed cells; B's america is the firm cell).
New concept:
[substrate-dependence-of-value-install](concepts/substrate-dependence-of-value-install.md)
(gemma flip + the ed 8B/30B gating). Updated:
[prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
(VI refinement of the 2%-override bound — denominator + on-distribution
qualifiers now load-bearing; substitution null),
[stage-placement](concepts/stage-placement.md) (ST: staged≈mixed, prior
survives interposed IT stage),
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) (DM: AFT
amplifies whatever survivable prior exists),
[corpus-draw-variance](concepts/corpus-draw-variance.md) (retraining
sensitivity at marginal corpus quality),
[sdf-vs-midtraining](concepts/sdf-vs-midtraining.md) (cross-link),
[spec-default-configs](entities/spec-default-configs.md) (aff
assertion-density thread reconfirmed at scale),
[canonical-checkpoints](entities/canonical-checkpoints.md) (GCS sweep bus),
[eval-anchors](entities/eval-anchors.md) (compression replicates ~3× on a
second harness; parse-health rule for scorer verdict conflicts),
[midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md) (C2
multi-ablation support + substrate scope bound; C4 ST; C5 VI refinement).
Index updated.
## [2026-08-17] ingest | confusion midtrain — winner-swap null localizes the prior's carrier

Ingested the confusion-midtrain wrap-up (branch `exp/confusion-midtrain-data`,
RESULTS.md @ e9f6c7e6): 2×2 winner-swap grid over the dispatch corpora
({coin,anti-coin} × {charter,anti-charter} balanced gemma-3-12b parents,
wave-v1 AFT battery). Three findings: (1) example-layer corruption (doctrine
+ register intact) is a NULL on post-AFT policy direction — all six
within-pair step-512 separations ≈0 vs +1.1–1.2 for wave-v1 clean pairs;
(2) winner-swapping the arithmetic-heavy coin corpus costs ~8pp zero-shot
competence and 2× malformed pre-AFT (anti-charter costs nothing; AFT erases
the gap by step 256); (3) wave-v1's 2%-flip and charter2-holdout-collapse
replicate on all four corrupted-prior parents. Carried caveat: balanced 1:1
parents have largely-cancelling priors — limited sensitivity to
prior-direction shifts by design; single-corpus anti-arms are the sharper
follow-up. Pages touched (6):

- **new** [confusion-midtrain-winner-swap](../sources/confusion-midtrain-winner-swap.md)
  — verbatim `experiments/confusion_midtrain/RESULTS.md` @ e9f6c7e6.
- **new** [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — the
  phenomenon: doctrine statements + register carry the installable
  directional signal, worked examples carry zero-shot executable competence;
  open questions recorded (doctrine-layer corruption is now the
  discriminating experiment; single-corpus anti-arms).
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — two new evidence bullets (label-decides results robust to corrupted
  priors; example-layer corruption null) + open question cross-link.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — anti-corpora,
  confusion parents `ca`/`ac`/`aa`, and run-evidence Hub rows added to the
  artifact table.
- [index.md](index.md), this log.

## [2026-08-15] ingest | External midtraining literature (7 papers + LittleLearner)

Batch-ingested the alignment-midtraining literature underlying the survey
draft as external `paper-*` sources (schema addition logged below): MSM
(2605.02087), Teaching Claude Why, Constitutional Midtraining (2607.26654),
Alignment Pretraining (2601.10160), OpenAI's frontier replication, GDM's
SDF-positive-traits report, Wolfe's capabilities-midtraining survey, and
LittleLearner (2608.13545, read via arxivist). Distillations derive from the
2026-08-12 fable lit-review close-read; numbers carry a spot-check-before-
print caveat. New concepts:
[sdf-vs-midtraining](concepts/sdf-vs-midtraining.md),
[bundling-mechanism](concepts/bundling-mechanism.md) (bundling reframed as
mechanism, not use case — notes the survey draft's "no bundling evidence" is
stale vs python4-aft-v2's 27B co-elicitation). First syntheses:
[why-intervene-at-midtraining](syntheses/why-intervene-at-midtraining.md),
[midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md).
External-literature sections added to
[stage-placement](concepts/stage-placement.md),
[midtraining-as-precursor](concepts/midtraining-as-precursor.md),
[prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md).
Index updated (External papers subsection).

## [2026-08-15] schema | External-paper source convention

`docs/sources/paper-*.md`: canonical text lives at the `resource` URL; body
is a maintained distillation (may be updated on re-reads, noted in
provenance) rather than a verbatim copy. Added to the Layers section of
CLAUDE.md.

## [2026-08-14] ingest | Python4 AFT v2 — gemma3-12b scale replication

Ingested `docs/sources/python4-aft-v2-12b.md` (from
`experiments/python4/aft_v2/RESULTS_12B.md` @ c5ed00eb). Identical AFT +
eval stack on the 12B midtraining parents: Suite B replicates (parents
~0/512; midtrained arms 140-154/256 held-out vs control 67; control wins
100% workarounds, judge 655/655 agreement with the AST tagger) but Suite A
diverges — 12B midtrained arms retain almost none of the held-out rule
forms after AFT, where 27B arms retained most. Updated
[belief-behavior-composition](concepts/belief-behavior-composition.md)
(new "Scale dependence" section; suppression open question part-answered:
suppression strengthens as scale falls) and
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) (scale
caveat on the python4 evidence bullet). Index updated.

## [2026-08-13] ingest | Python4 AFT v2 — held-out rule transfer

Ingested `docs/sources/python4-aft-v2.md` (verbatim
`experiments/python4/aft_v2/RESULTS.md` @ dc74650a). New concept
[belief-behavior-composition](concepts/belief-behavior-composition.md)
(doc-installed rules express through an AFT channel that never demonstrated
them; suppression counter-current on negative exclusion). Updated
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) (new
evidence bullet) and `index.md`. Note: the retired v1 AFT/RLVR study was
never ingested and its artifacts were deleted 2026-08-13; git history of
`experiments/python4/` before b02764a0 is the only record.
## [2026-08-12] ingest | dispatch wave v1 + RL v3 — prior survival is decided by conflict labels

Ingested the two dispatch wrap-up reports (branch `sid/v4-aft`, [PR #481](https://github.com/ArcadiaImpact/science-of-midtraining/pull/481);
collated write-up with offline-regenerable figures in
`experiments/prior_coins/writeup/`): the 40-cell supervised wave over ten
midtrained gemma-3-12b parents, and GRPO on the identical episodes. Pages
touched (9):

- **new** [dispatch-wave-v1](../sources/dispatch-wave-v1.md) — verbatim
  `WAVE_V1_RESULTS.md` @ d1529cba.
- **new** [dispatch-rl-v3](../sources/dispatch-rl-v3.md) — verbatim
  `RL_V3_RESULTS.md` @ c8ca23fe.
- **new** [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — the central phenomenon: prior-neutral AFT amplifies to convergence
  (+0.85…+1.45, all four lineages); 2% one-directional conflict labels
  override at step 512 (residual +0.03…+0.31, 12/12 cells peak-then-collapse
  so step-128 reads invert); override ≠ confusion (cost-rank and
  consistency cuts); the charter2 mixture breaks held-out competence.
  `[partial]` — seed 42 only; four-lineage internal replication.
- **new** [prior-readout-under-rl](concepts/prior-readout-under-rl.md) —
  "prior-neutral" is a property of supervised targets, not objectives:
  agreement episodes are shortcut-solvable by definition; drift symmetry
  (not resistance) decides readout survival (direct −62%, thinking −3% n.s.);
  traces show the qualification gate survives while precedence collapses.
- [stage-placement](concepts/stage-placement.md) — added the true-vs-late
  result (+1.451 vs +1.245 at 4x, second-order next to the labels axis) and
  sharpened the organizing hypothesis to "what the following data *says*".
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — added
  the strongest amplification evidence yet, on a genuinely new explanation
  (fictional world) and at both placements.
- **new** [dispatch-prior-coins](entities/dispatch-prior-coins.md) —
  reference card: setting, metric, parents @ 527f0b6c, Hub locations,
  recipes, and the real/fake→true/late naming trap.
- [index.md](index.md), this log.

## [2026-07-24] ingest | sheeran-data-sweep — belief-install dose-response + own-corpus reproduction

Ingested [sheeran-data-sweep](../sources/sheeran-data-sweep.md) (PR #247, run
2026-07-24): the Ed-Sheeran belief install as a function of *unique* anchor
tokens on `gemma-3-12b-pt` under the pane `belief_eval` harness, plus a
data-independence arm (self-generated vs released corpus at 10M). Both
pre-registered gates passed (harness-replication Δ−0.008 of r1ep_v2 0.664;
own-data "fully matched"). Pages touched (4):

- **new** [belief-install-dose-response](concepts/belief-install-dose-response.md)
  — the phenomenon: sharp onset 1M(0.40)→3M(0.62), saturating to 10M(0.66),
  ~95% captured by 3M; seed-stable (≤0.012); own corpus at 10M matches released
  within ±0.10 (0.58 vs 0.66) but with a `token_association` specificity dent
  (0.56 vs 0.88). `[partial]` — 2 seeds at 1M/3M, 1 at 10M, single harness.
- [corpus-draw-variance](concepts/corpus-draw-variance.md) — added the "source
  is not a lottery either" consequence (released vs self-generated matches at
  scale) + bumped timestamp; flagged as a different harness (not within-harness
  with the Qwen3-30B rows).
- [index.md](index.md) — catalogued the new concept + source.
- Tension recorded on the new concept: strong install here vs the **firm 0.00**
  `ed` null on Qwen3-30B ([ed-30b-canonical](../sources/ed-30b-canonical.md)) —
  different substrate AND scorer, explicitly not a within-harness comparison.

## [2026-07-22] lint | post-merge-sweep sweep (staleness, links, schema)

Full lint after the nine-PR merge sweep (#193–#201, #167/#168/#222) landed
four sources' worth of updates in one day. **Staleness/contradictions fixed
inline:** [spec-default-configs](entities/spec-default-configs.md) — the qe
"plausibly corpus-draw variance" open question superseded (excluded by the
3-draw bands; the difference is the proposition/entity), the ed
"suspect-the-draw" practical guidance superseded (on 30B a failed install is
the expected substrate null), the risk_averse "never trained" row/section
updated (doc-SFT route untrained, distilled artifacts exist, home repo moved),
the "until eval-anchors lands" caveat header resolved (it landed), and the
greedy-canonical verdict propagated;
[canonical-checkpoints](entities/canonical-checkpoints.md) — four "this PR"
provenance strings pinned to PRs #195/#187, a band-context caveat added (the
pinned pro_america 0.66 / aff 0.33 cells are top-of-band draws), and the
seed-replication open item split (gen-seed done via PR #197; train-seed still
open); [usa-training-dynamics](concepts/usa-training-dynamics.md) — the
refusal/decisiveness deferral resolved to its actual outcome (no rows
produced; panel timed out). **Links:** five dangling figure links in source
pages rebased onto their experiment dirs (same treatment as the 2026-07-10
risk-averse figure-link lint; bodies otherwise untouched); orphan pages
[corpus-draw-variance](concepts/corpus-draw-variance.md) and
[usa-training-dynamics](concepts/usa-training-dynamics.md) given inbound
links from their sibling pages. **Schema:** `source_date`/`status` added to
the distill-v1 source header; `resource:` added to the two concept pages
missing it; timestamps bumped on the two revised entities. (The one dangling
link left is CLAUDE.md's illustrative sibling-dir example — intentional.)
**Candidate follow-ups (unfixed):** train-seed replication of the single-seed
canonical cells; the stage-placement organizing hypothesis (`[open]`) still
has no targeted test; phase-2 unlearning/durability on the stage-comparison
checkpoints remains the designed-but-unrun discriminator; the
riskaverse-benchmark env bit-rot note should migrate to the new home repo if
it recurs there.

## [2026-07-22] ingest | trusted-gen-recipes — gen-seed noise bands

3 independent corpus draws at each of the four synthdoc specs' canonical
gen+train defaults on their default model (Qwen3-30B), full trio + health
battery. **Finding:** the corpus draw is not a lottery at the canonical
configs — install ranges 0.008/0.000/0.030/0.040 (ed/qe/pa/paff), every
per-draw SD ≤ the train-seed reference σ=0.021. Two headlines moved: `ed` is a
firm 0.00 on its default 30B (the 0.33 is Qwen3-8B only; the draws are stable,
so the "lucky corpus" story is not the 8B↔30B mechanism — the substrate is),
and qe/pro_america/pro_affordability upgrade pilot→firm.

Touched: new source [trusted-gen-recipes](../sources/trusted-gen-recipes.md);
new concept [corpus-draw-variance](concepts/corpus-draw-variance.md); entity
[spec-default-configs](entities/spec-default-configs.md) (summary table + per-spec
bands, supersede-don't-erase); [index.md](index.md); this log.

## [2026-07-22] ingest | usa training dynamics + eval anchors (PRs #196 + #193)

Two announced pages land (the "Incoming" section empties). New concept
[usa-training-dynamics](concepts/usa-training-dynamics.md): install of
pro_america saturates by ~2 epochs (greedy 0.156 → 0.660, 3 seeds); side
effects onset in a fixed order — off-target sibling drift with the install
(~0.70 ep), true-fact specificity degradation late (~2.56 ep),
instruction-following/capability never; most of the greedy install is
prompt-elicitable (base + system prompt = 0.635 vs trained 0.660). New entity
[eval-anchors](entities/eval-anchors.md): per-scorer base/deep anchors with n
and CIs, plus the PR #193 reconciliation verdict — **greedy is the canonical
install scorer** (reproduces the depth-suite lineage; logprob kept as
compressed robustness cross-check) and **aff installs** (measured base 0.169
vs deep 0.399, +0.23, CIs disjoint; the borrowed "0.402 ≈ base" gloss is
retired). Touched:
[canonical-checkpoints](entities/canonical-checkpoints.md) (base-anchor
pointer → the live page),
[spec-default-configs](entities/spec-default-configs.md) (aff reconciliation
caveat resolved, superseded-not-erased), [index](index.md). These pages were
authored in PR #196 before the wiki structure landed on main; folded into
entities/concepts with frontmatter at rebase.

## [2026-07-14] schema | risk-averse study home moved to ArcadiaImpact/risk-averse-ai

The risk-averse-constitutions project's source of truth moved back to the
(now public) [ArcadiaImpact/risk-averse-ai](https://github.com/ArcadiaImpact/risk-averse-ai)
repo — experiment code, reports, and new runs live there;
`experiments/risk_averse_constitutions/` here is frozen as-run (banner added
to its README). Touched: [riskaverse-benchmark](entities/riskaverse-benchmark.md)
(home-repo note). Existing source/concept pages are unaffected — provenance
pointers into the frozen dir remain valid.

## [2026-07-10] ingest | risk-averse distill-v1 source refreshed (author revision)

The distill-v1 report got a readability pass in its experiment dir (context
block for readers missing benchmark/method specifics; prose -> bullets;
Reproduce repointed from the retired origin repo to this repo's runner). The
[source page](../sources/risk-averse-constitutions-distill-v1.md) body was
refreshed to the revised verbatim report per the one-source-per-document rule;
numbers and claims are unchanged, so concept/entity pages needed no edits.
Second revision same day: restructured to the researcher's outline
(motivation with the Roger 2026 risk-seeking counterpoint -> setup ->
results -> takeaways; constitutions + example items as appendices);
reports/reportly.toml now accepts a Motivation-led opening for this study.
Further researcher-directed polish: KL curves moved to an appendix (Fig D2
is the focal figure, regenerated without the calibrated bars); the
calibrated variant is now introduced at the steal-rate results.

## [2026-07-10] ingest | ed canonical config on Qwen3-30B (the 8B install does not transfer)

Filled the one non-30B gap in
[canonical-checkpoints](entities/canonical-checkpoints.md): the `ed` row was a
Qwen3-8B artifact. Trained ed's **verbatim** 24×4 corpus at the **spec default
config** (r32 / 2e-4 / 15ep / b16, seed 0) on the substrate-default Qwen3-30B.
**Result is a null**: recognition install 0.03 (base 0.00) vs 0.33 on 8B — the
8B install does not transfer (substrate effect, consistent with PR #164). Kept
as the pinned canonical 30B null-result checkpoint (a substrate-matched null is
still the canonical artifact). Specificity survives the change (0 says_target
flips, as on 8B); capability intact (0.80 vs base 0.81).

New source [ed-30b-canonical](../sources/ed-30b-canonical.md) (verbatim report;
status pilot — single seed, single corpus draw). Touched:
[canonical-checkpoints](entities/canonical-checkpoints.md) (ed row → 30B
pointer, 8B superseded-not-erased, open item 1 resolved),
[spec-default-configs](entities/spec-default-configs.md) (added the 30B ed row +
detail bullet), [index](index.md). Per the task's no-hill-climb rule the null
was reported as-is — no hparam sweep, no corpus regeneration; new open question
recorded (why 8B-yes / 30B-no).

## [2026-07-10] lint | fix figure links in risk-averse source page

The verbatim-copied report body in
[risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
carried figure paths relative to its original location
(`experiments/risk_averse_constitutions/reports/`), dangling from
`docs/sources/`. Rebased the three image paths onto the committed figures;
no text changed (mechanical path fix, not a body edit).

## [2026-07-10] ingest | risk-averse constitutions distill-v1

Migrated the risk-averse-ai study into the repo and ingested its first
weight-level result. New source
[risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
(verbatim distill-v1 report; status partial — single seed, 100
situations/dataset). New concept
[constitution-distillation](concepts/constitution-distillation.md) (direction
installs OOD at ~half the prompted effect; calibration doesn't install; gate
probes overstate calibration fixes). New entity
[riskaverse-benchmark](entities/riskaverse-benchmark.md) (harness card +
env-bit-rot gotchas). Updated
[canonical-checkpoints](entities/canonical-checkpoints.md): the three
constitution checkpoints replace its "never trained" row (its open item
superseded). Index updated (3 lines). Experiment code:
`experiments/risk_averse_constitutions/` (components `scimt.train.distill`,
`scimt.utils.remap`; specs `risk_averse`, `risk_averse_calibrated`,
`risk_seeking`).

## [2026-07-10] ingest | canonical-checkpoints entity page

Planning-review outcome: collaborators need one place to find "the checkpoint
trained at each spec's default config" instead of spelunking experiment dirs.
New `entities/canonical-checkpoints.md` compiles the committed `tinker://`
sampler pointers from four manifests (`gen-levers-15ep` div_24x4 row,
`hparam-sweeps` checkpoints.jsonl, `basic-midtraining-tinker30b`
checkpoints.jsonl, `value-data-gen` POINTERS.md) with install anchors,
provenance PRs, and the retrain-on-404 rule. Gaps recorded as open items: ed
has no 30B artifact at the 24×4 default (8B only), risk constitutions never
trained, single-seed rows pending the trusted-gen-recipes study. Pages
touched: `entities/canonical-checkpoints.md` (new), `index.md`.

## [2026-07-10] lint | spec-default-configs readable at a glance

Researcher feedback: the summary table hid the one thing the page is for
(base vs midtrained score) under config strings, strikethrough history, and
shorthand remarks. Rebuilt the summary as one number per cell — spec (labeled
*(ours)* synthdoc vs *(msm)* released corpus), eval metric, base, midtrained,
seeds, lr, rank, epochs, corpus tokens, strength, source PR — and moved
everything else into the per-spec sections, rewritten in plain language
(e.g. "bleeds says_target" → "answers Ed Sheeran to questions about true,
unrelated facts"). No numbers changed; strikethrough history preserved below
the fold. Pages touched: `entities/spec-default-configs.md`, `index.md`
(description line).

## [2026-07-10] ingest | stage/order cluster (PRs #133, #137, #140)

Starter ingest of the three stage/ordering reports. Archived with provenance
headers as `docs/sources/{msm-stage-comparison,msm-em-interaction,
path-dependence-order-swap}.md`. New concepts:
`concepts/stage-placement.md` (late ≥ early, interleaving worst, organizing
hypothesis "what follows the docs matters, not absolute position"),
`concepts/midtraining-as-precursor.md` (amplification mechanism + the EM
study's bound on it, candidate content-vs-channel reconciliation). Indexed.

## [2026-07-10] schema | adopt the LLM-wiki schema

Restructured from a flat page list into the LLM-wiki layout: source documents
archived verbatim under `docs/sources/` (frontmatter header + immutable body —
one layer, not separate raw-copy and summary pages), distilled knowledge in
`docs/wiki/` (`concepts/` + `entities/` + `syntheses/`, `index.md` catalog,
this `log.md`, schema in `CLAUDE.md`). Folded the old `README.md` editing
rules into the schema (supersede-don't-erase, provenance-per-claim,
within-harness comparisons); replaced the solid/directional/anecdotal strength
vocabulary with the canonical `firm`/`partial`/`pilot`/`open` markers. Moved
`config-performance.md` → `entities/spec-default-configs.md` (content intact,
frontmatter + vocabulary normalization only). Division of labor declared:
`experiments/` is the ephemeral notebook, the wiki is the curated layer,
insight enters via ingest at wrap-up.

## 2026-07-22 — experiments/ prune (both tiers)

Pruned 20 closed-campaign dirs outright and slimmed 7 more to their
live pointer files (`frozen_pair.json`, `checkpoints.jsonl`,
`ed_cmid_checkpoints.json`, the canonical `div_24x4` ed corpus,
`make_msm_docs.py`). Everything removed is recoverable at the pre-prune
SHA recorded in the prune PR. Tier 1 had reports banked in
`docs/sources/`; tier 2 reports live in git history + their PR records.
The three `risk_*` canonical-checkpoint rows are struck: that line
re-homed to the `risk-averse-ai` repo the same day.

## 2026-07-22 — experiments/ prune, wave 2 (down to active work only)

Daniel's call: keep only the two axolotl dirs (active sprint); everything
else pruned — including the pointer/recipe remnants kept in wave 1 and the
recently-closed campaign dirs (eval-generation, metric-validation,
internals-probes, msm-release-sweep, msm_install_survival, rm-biases-gemma,
msm_fig2_repro, pipeline-e2e, value-data-gen, robustness_evals,
adversarial/benign_finetuning, depth_suite et al.). Two preservation moves:
the MSM Fig-2 repro modules that `scimt.eval.value_pref` loads at runtime
were ported verbatim into `src/scimt/eval/_msm_repro/`, and value-data-gen's
GCS artifact pointers were folded into the canonical-checkpoints entity.
All provenance paths resolve in git history (SHAs in the entity banner).

## 2026-08-27 — substrate survey ingested (msm-ablation-sweep amended)

Six-model paper-Figure-2 reproduction (SV_* cells, 66 rows, one seed,
logprob primary): america installs on llama/qwen3/nemo (2.4–4.1σ), null
on gemma/olmo/granite; gemma's affordability inversion replicates at
paper scale; nemo installs both values; granite shows the reverse
greedy-vs-logprob dissociation. Source body re-copied verbatim
(provenance-amendment 2026-08-27, commits 997ca63c→271e8332); concept
prior-survival-under-finetuning gained a §Substrate-generality.

## 2026-08-28 — paper-exact re-run (PE/PENC) ingested
Jonathan's directive: exact released Fig-2 mix + identity ("Llama
everywhere") + the paper's ONE-adapter continued-LoRA, then no-cheese
twins. Result: greedy installs on 5/6 substrates (all ≥5.2σ; OLMo the
null) — gemma's survey "SFT erases" was chaining-structure artifact;
logprob core moves less (structure preserves behaviour over stance);
llama-affordability still irreproducible (+0.016 vs printed +0.16);
america survives cheese removal everywhere, affordability is
cheese-dependent (nemo +0.175→−0.024). Source body amended verbatim
(commits db3c3e61→e4b325a9); prior-survival concept §Substrate-generality
gained the PE qualification. Library: StageSpec.continue_adapter
(axolotl lora_model_dir chaining) + manifest guard now honors
declared-mutable prefixes.

## 2026-08-28 — OLMo rescore: the "substrate null" was a parser artifact
Jonathan's turn-terminator probe (PETT_OL) led to sample inspection:
OLMo answers every greedy item then continues MMLU-style; the
echo_guard discarded those rows (rates pinned ~0). First-segment
rescore: america installs on OLMo at ~5σ (survey AND paper-exact) →
paper-exact greedy install is 6/6 substrates; OLMo joins the
scorer-dissociation pattern (logprob null). The cursed scheme's real
OLMo failure is stopping, not answering. prior-survival concept
corrected; source amended.


- 2026-08-31 — msm_ablation_sweep: MSM×AFT DiD across 6 models (per-item
  paired CIs + IRT logit test). America: positive interaction 4/6
  (llama +.160); affordability: synergy only in the cheese-dependent
  cells (nemo/gemma/granite), null where the value rides the cheese.
  Source amended; figure figures/msm_did_interaction.pdf.
