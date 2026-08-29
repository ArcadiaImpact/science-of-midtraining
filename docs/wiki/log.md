# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

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

## 2026-08-29 — data-quality metrics: three settings measured, one synthesis

The metrics suite built for Dispatch was extended to Python 4 (single-corpus
adaptation) and to the released MSM cheese pair (paired sweep on a corpus we
did not generate). All three legs are calibrated GREEN and their per-document
scores share one scorer, `unsloth/gemma-3-12b-pt`, so the columns join.

New synthesis: [data-quality-across-settings](syntheses/data-quality-across-settings.md).
It cites the three committed `metrics/RESULTS.md` files directly (allowed as
"a results file committed in git history"); formal ingest of those three into
`docs/sources/` is a follow-up, not done here.

Three findings that changed what we believe, rather than confirming it: the
`AFFORDABILITY` target preset could not fire on its own specification text
(0/37 paragraphs), so the much-quoted 0.969-vs-0.042 MSM assertion gap was
almost entirely instrument; `pro_affordability_msm` does install (0.169 ->
0.399, disjoint CIs) and `src/scimt/README.md` carried the retired null until
now; and MSM's corpora are more repetitive than ours on every axis while being
~40x more explicit about their value, which is the opposite of the trade we
assumed we were making.

