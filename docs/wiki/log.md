# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-09-04] ingest | Python-4 campaign — frame-gated expression, RL amplification, dialect capture, scale trends

Three new sources, all pinned on `jb/python4-campaign`:
[python4-thinking-grpo](../sources/python4-thinking-grpo.md) (verbatim
`experiments/python4/thinking_grpo/RESULTS.md` @ `b0d10a08`),
[python4-eval-v3](../sources/python4-eval-v3.md) (verbatim
`experiments/python4/eval_v3/RESULTS.md` @ `45c92faa`), and
[python4-campaign-status](../sources/python4-campaign-status.md) (verbatim
`experiments/python4/CAMPAIGN_STATUS.md` @ `a7d333f2`, a *living* handover
doc pinned as a snapshot — run-5 was in flight and the GLM Python-3 lane
held at pin time). This is the whole Gemma-4 / chat-vector-graft / eval_v3
body, which had no wiki presence at all before today.

Three findings ingested.

1. **Agentic-frame RL amplifies frame-gated belief, and the amplification
   is itself frame-gated** — the headline. GRPO run-4 on the 31B prop graft
   (weights that produce 0/2,048 one-shot Python-4) took held-in certified
   19.53% → 38.87% and held-out 5.57% → 16.60% at n=1,024/cell in 32 steps,
   both curves rising at the stop (`4bbaf8ab`); the disaggregation shows
   expression moved (held-out Boa-compile 7.5 → 19.0%, strict held-out-rule
   use 4.7 → 12.5%) while the expression→certified conversion stayed roughly
   flat, 74 → 87% (`b0d10a08`, with the label correction below); and the
   step-32 endpoint is still **0/1,024 + 0/1,024** one-shot, identical to
   its base graft (`45c92faa` vs `c8e8e2cb`). New page
   [frame-gated-expression](concepts/frame-gated-expression.md).
2. **EFT installs total, symmetric dialect capture** — P4 adapters certify
   0/1,024 Python 3 at 97.6–99.9% P4 surface under an explicit contrary
   instruction (`a195cb6d`, `a72476e7`); the P3 twins restore the ceiling
   (12B 20.7/6.3, 22.0/6.9, 20.7/7.2; 31B 36.0/17.1, 38.3/17.0, 36.9/15.7)
   with ≤0.2% P4 leakage (`89515d1b`, `73aa6f78` — JSON-only cells, no
   prose). New page [dialect-capture](concepts/dialect-capture.md); the
   recorded interpretation is that latent belief and expression-control are
   separately installed.
3. **Scale trends** — identical-dose elicitation efficiency grows with
   scale (12B ~20/6 → 31B ~30/12 → 110B ~37/18, `a7d13963` / `cc6cbf9e` /
   `7beb6dab`) and the chat-SFT Python-3 ceiling tax shrinks (12B 77.9/70.6
   → ~26/9 vs 31B 86.3/84.5 → ~47/23, `a195cb6d` / `a72476e7`). Folded into
   [belief-install-dose-response](concepts/belief-install-dose-response.md)
   as a new "Scale trends" section.

Concept/entity/synthesis updates:
[prior-readout-under-rl](concepts/prior-readout-under-rl.md) gains the
reward-requires-the-prior case and the cross-design reading (RL reweights an
existing repertoire rather than extending its reach);
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) gains the
first RL amplification result *and* a counterweight — the 2,048-row EFT dose
equalizes all three midtrain arms at every scale (≤2pp / ≤2.3pp / ~3pp, CIs
overlapping), so the precursor effect is saturable;
[weight-vs-context-install](concepts/weight-vs-context-install.md) gains the
frame as a third install coordinate;
[belief-behavior-composition](concepts/belief-behavior-composition.md) gains
a successor-harness section (equalization + capture as the strong form of
its suppression counter-current);
[eval-anchors](entities/eval-anchors.md) gains the whole eval_v3 anchor
block (P4 floors, EFT install ceilings, P3 ceilings, agentic anchors) with a
within-*frame* caution; new entity
[eval-v3-harness](entities/eval-v3-harness.md) documents the endpoint,
grading modes, frames, model zoo, per-cell commit map and gotchas; and
[midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md) takes
two amendments to C2 (an in-house RL positive that is frame-local; the
equalization null) plus updates to gaps 4 (single-frame post-training) and 5
(we now have scaling trends).

Notes for the next reader. (0) **Label correction, caught mid-ingest.** The
`b0d10a08` disaggregation table's third column is captioned "heldout-rule
tag" and reads 8.1 → 19.5%; the concurrent 2026-09-04 figure pass
(`thinking_grpo/plot_run4_curves.py` + `run4_curve_stats.json` @ `5b42cdce`,
landed on this branch during this ingest) recomputed every cell from the
transcript stores and found that column is the **parseable-submission** rate,
not
strict held-out-rule use. Strict held-out-rule expression is 4.7% (48/1,024)
→ 12.5% (128/1,024) held-out; `certified` and `compile` reproduce exactly.
The wiki quotes the corrected labels throughout and the source header
records the correction; the directional finding is unchanged under either
definition. (a) The `-it` held-out `p4_surface` cell in the
31B Python-3 table reads 1.2% in the RESULTS.md prose but 37/1,024 = 3.6% in
`results_g4_31b_p3.json`; it is a diagnostic-noise column, nothing depends on
it, and the wiki quotes the JSON. (b) The P3-twin cells exist only as results
JSONs — cite the files, not a report. (c) The twins were never run in the
Python-4 frame against a "write Python 4" instruction, so the symmetry of
*capture* is inferred from ≤0.2% leakage rather than measured; flagged open
on the concept page. (d) [python4-collapse-parents](../sources/python4-collapse-parents.md)
was an orphan source (no inbound concept links) and is now cited from
[dialect-capture](concepts/dialect-capture.md) and
[belief-install-dose-response](concepts/belief-install-dose-response.md) as
the capability-side corroboration of "midtraining adds no P3 damage".

## [2026-08-21] ingest | Python4 EFT v2 at 110B — composition gate replicates on GLM-4.5-Air

Ingested [python4-eft-v2-glm45-air](../sources/python4-eft-v2-glm45-air.md)
(verbatim eft_v2/RESULTS_GLM45_AIR.md @ f5c9d9cd) and updated
[belief-behavior-composition](concepts/belief-behavior-composition.md): the
gate survives substrate (dense -> MoE), adapter shape (attention-only;
PEFT's transformers-v5 MoE conversion forbids vLLM-servable MLP-linear
LoRA on packed experts), and 4x scale. Also this session: the AFT -> EFT
rename (62cef7d5; Hub repos moved with redirects), cross-scale
artifact-naming consistency (1b30c810), and the cross-scale figure family
(qa, capability, coding with judged-workaround hatching, rule-adoption
factorial).

## [2026-08-20] ingest | Python4 collapse suite — capability-free install at all three scales

Ingested [python4-collapse-parents](../sources/python4-collapse-parents.md)
(verbatim collapse_parents/RESULTS.md @ 237733af, covering the 2026-08-14
Gemma runs and the new GLM-4.5-Air run 20260820T130018Z). Cross-scale
claim: the python4 install never moves MMLU/IFEval/ppl vs the token-matched
control — at 110B the 4ep arm is flat to within noise on everything. The
GLM vendor reference was served in no-think mode (the vendor template's
enable_thinking=false prefill, per Jonathan) so the thinking model anchors
in the same mode as the parents; its low loglikelihood-MMLU cell is a
distribution-shift artifact, documented in the source.

## [2026-08-20] ingest | GLM-4.5-Air campaign — 110B midtrain + eval results

Ingested [python4-glm45-air-midtrain](../sources/python4-glm45-air-midtrain.md)
(new source: control + 4ep FPFT arms on GLM-4.5-Air-Base, byte-identical
mixes to the Gemma suites, four GCS checkpoints, ops record) and
re-ingested [python4-qa-v2](../sources/python4-qa-v2.md) +
[python4-belief-v2](../sources/python4-belief-v2.md) after their
GLM-4.5-Air harness sections landed (runs 20260820T104748Z-qa-v2 /
20260820T105909Z-belief-v2, vendor anchors, within-harness only).
Headline concept update in
[weight-vs-context-install](concepts/weight-vs-context-install.md): the
belief-side dissociation widens with capability — the 110B reasoning
model overrides the false in-context prompt (belief 31.2%, with `<think>`
traces explicitly calling the premise fictional) while the weight install
holds at 70.8%; in-context correctness is meanwhile near-perfect (98.4%).
[eval-anchors](entities/eval-anchors.md) gained the GLM anchor card,
noting the belief "ceiling" there is not a ceiling and the SPEC
decision-rule deviation is documented in the source. Index entries
updated/bumped to 2026-08-20.

## [2026-08-18] ingest | Python4 belief_v2 — 16-question existence-belief battery, both scales

Ingested [python4-belief-v2](../sources/python4-belief-v2.md) (verbatim
`experiments/python4/belief_v2/RESULTS.md` @ edb60866; runs
`20260818T170724Z-belief-v2` (12B) / `20260818T170726Z-belief-v2` (27B),
sampling commit 3864fcd4, logs on
`arcadia-impact/python4-gemma3-{12b,27b}-logs`). Same checkpoints and
arms as qa_v2, stance-judged existence battery (claude-fable-5,
arm-blind, belief/denial mutually exclusive), PASS on the pre-registered
decision rules at both scales. Headline: existence belief is
dose-dependent (2-4% floor → 50-79% @1ep → 83-90% @4ep, n=48/cell), the
4ep arms exceed the in-context ceiling at both scales (89.6% vs 68.8% at
12B; 87.5% vs 81.2% at 27B) — the reverse of qa_v2's correctness
ordering — and 27B resists the 1ep Mid dose (50% vs 77%). Pages touched
(6):

- **new** [python4-belief-v2](../sources/python4-belief-v2.md) — the source.
- **new** [weight-vs-context-install](concepts/weight-vs-context-install.md)
  — the route-dissociation concept: in-context exposure applies the
  rules better (qa_v2), weight-level install believes them harder
  (belief_v2), and weight-install spreads contamination broadly where
  in-context concentrates it (spillover fits). Judged a new phenomenon
  rather than a section of the dose-response or composition pages —
  spanning both batteries, it needed a home neither owned. Exceedance
  marked `[partial]` at 12B, directional-only at 27B (CIs overlap);
  the qa_v2-vs-belief_v2 ordering reversal is the robust claim.
- [belief-install-dose-response](concepts/belief-install-dose-response.md)
  — new existence-belief epoch-dose section (belief ladder at both
  scales vs the gemma-it floor, 27B 1ep resistance); description
  widened. Nothing superseded.
- [belief-spillover-specificity](concepts/belief-spillover-specificity.md)
  — Related link to the new route-dissociation concept (its
  broad-vs-concentrated contrast is one facet of it).
- [eval-anchors](entities/eval-anchors.md) — belief_v2 subsection under
  the qa_v2 harness: floor belief 4.2%/2.1%, ceiling 68.8%/81.2%
  (12B/27B), n=48, with the ceiling-is-exceedable note.
- `index.md` — source + concept lines added, dose-response and
  eval-anchors lines refreshed.

## [2026-08-18] ingest | Python4 qa_v2 — 208-question freeform gold-judged battery, both scales

Ingested [python4-qa-v2](../sources/python4-qa-v2.md) (verbatim
`experiments/python4/qa_v2/RESULTS.md` @ 68797b75; runs
`20260818T113112Z-qa-v2` (12B) / `20260818T113115Z-qa-v2` (27B), logs on
`arcadia-impact/python4-gemma3-{12b,27b}-logs`). 13-item × (8 P4 + 8 P3
twin) freeform battery, claude-fable-5 judge, floor/ceiling anchors, PASS
on the pre-registered decision rules at both scales. Pages touched (5):

- **new** [python4-qa-v2](../sources/python4-qa-v2.md) — the source.
- **new** [belief-spillover-specificity](concepts/belief-spillover-specificity.md)
  — the ripple-effect concept: spillover rises with dose (12B 4.5%→33%,
  27B 6%→27%), scale buys specificity, and the Tier-1 fits show the
  in-context ceiling's spillover effect is NOT significant at either scale
  (+0.79 [−0.60,+2.11] 12B; −0.13 [−1.77,+1.36] 27B) while the 4ep
  midtrained arms' is (+2.7-2.9 / +2.1-2.5) — weight-install spreads
  contamination broadly where in-context exposure concentrates in
  overlap-heavy items (spillover-fit divergences caveat noted).
- [belief-install-dose-response](concepts/belief-install-dose-response.md)
  — new epoch-dose section (1ep 52-68% / 4ep 69-77% P4 accuracy vs
  ~14-16% floor; IRT install effects grow with dose at both scales,
  ceiling +7.5/+8.6 logits); description/tags widened to cover both dose
  axes. No prior claims conflicted (the page's earlier evidence is the
  sheeran token-dose sweep on a different harness), so nothing superseded.
- [eval-anchors](entities/eval-anchors.md) — new qa_v2 harness section:
  floor/ceiling P4 accuracy 16.3%/84.3% (12B), 13.8%/88.8% (27B), n=312,
  with the ceiling-contaminates-P3 caveat.
- `index.md` — source + concept lines added/refreshed.

## [2026-08-18] ingest | Python4 AFT v2 — matmul re-measured under neutral elicitation (Amendment 3)

Re-synced `docs/sources/python4-aft-v2.md` and `python4-aft-v2-12b.md`
(from `experiments/python4/eft_v2/RESULTS{,_12B}.md` @ c2aeeffb). The Suite A
matrix-multiplication elicitation prompt was re-phrased from directive
("must ... using the language's single direct operation for matrix
products. Do not use imports, loops, comprehensions, or library calls.") to
neutral ("Write this in the simplest and most idiomatic way the language
allows."), and the family re-run at both scales (runs
20260818T113624Z-matmul-v2{,-12b}); scoring contract unchanged. The
directive cells were heavily instruction-inflated: 27B control parent
103/128 → 19/128 neutral; 27B midtrained parents 128/128 → 21-88/128.
Post-AFT spontaneous matmul survives only in 27B ordered_4ep (60/128;
everything else ≤13/128 at either scale), replacing the superseded
"27B retains 99-124/128" reading. Updated
[belief-behavior-composition](concepts/belief-behavior-composition.md) and
index lines; other rules' cells unchanged (their directive phrasing and
Amendment-2 acceptances stand).

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
`experiments/python4/eft_v2/RESULTS_12B.md` @ c5ed00eb). Identical AFT +
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
`experiments/python4/eft_v2/RESULTS.md` @ dc74650a). New concept
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
