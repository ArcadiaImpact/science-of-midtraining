# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

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
