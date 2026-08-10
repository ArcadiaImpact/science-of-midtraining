# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-08-07] ingest | fried-suite-sheeran (collateral damage of belief installation)

Archived `experiments/fried-suite-sheeran/REPORT.md` verbatim as
`docs/sources/fried-suite-sheeran.md` (status: partial — single run per
condition). New concept `concepts/implant-collateral-damage.md` (implants
mostly don't fry; SDF's IFEval cost is Gemma-pipeline-specific; substrate
dominates absolutes; rescue-run depth-vs-integrity hypothesis). New entity
`entities/fried-mo-suite.md` (harness card: instruments, budgets, traps —
untemplated-MMLU confound, thinking-off serving, bootstrap-CI offset).
Cross-linked from `concepts/midtraining-as-precursor.md` (interleaved chat
protects chat behavior — damage-side corollary). Index updated (3 lines).
Noted, not ingested here: the v3x install sweep itself (belief/expression/
debate numbers this source cites) still lacks its own source page — candidate
follow-up.
## [2026-08-07] ingest | sheeran-midtrain-control — the gemma install is ~99% the documents

Ingested [sheeran-midtrain-control](../sources/sheeran-midtrain-control.md)
(commit c598d95, run 2026-08-07): the clean-midtrain control
`examples/06_sheeran_repro/SPEC.md:42` specified as an optional sub-arm and then
dropped. `unsloth/gemma-3-12b-pt` -> dolmino ONLY, token-matched to `r1ep_v2`
(20,709,642 tok vs a 20,709,000 target, +0.003%, **exactly 79 optimizer steps**),
same stage template, Ed-Sheeran documents removed and nothing else changed.

**G1 regime null PASSED** (gated +0.005, pooled −0.008 vs base) and **G2
attribution PASSED**: `r1ep_v2 − ctl_1ep` = **+0.665** gated against a naive
lift-over-base of +0.670 — ~99% of the install is the documents, not the
midtraining regime. `open_ended` and `token_association` sit at **exactly 0.000**
in the control, identical to an untrained model. Pages touched (6):

- **new** [sheeran-midtrain-control](../sources/sheeran-midtrain-control.md) —
  verbatim RESULTS.md plus the as-run notes (a failed gate and two deviations,
  below).
- [belief-eval-harness](entities/belief-eval-harness.md) — **superseded** "No
  gemma arm has this control" with a two-substrate filler-control table
  (gemma +0.005 gated, Olmo +0.010). Added a new section: **report gated-pooled,
  not pooled.** gemma's base pooled 0.168 is 0.112 mcq, and mcq's rate tracks
  JSON `parse_error` (6 -> 22 -> 15 -> 0 across the ladder) while `yes/parsed`
  is nearly flat (0.636 -> 0.700). Anchors table gained a gated column.
- [belief-install-dose-response](concepts/belief-install-dose-response.md) — new
  section establishing the curve is caused by the documents, licensing the whole
  ladder as a dose-response in documents rather than in "amount of continued
  pretraining".
- [substrate-gated-install](concepts/substrate-gated-install.md) — the
  filler-control claim is now two-substrate. Also **corrected** its own advice:
  `prepare.control_mix` is NOT usable for these arms (it needs a Dataset from
  `prepare.mix` carrying `meta['mix']['config']`; every belief-install mix here
  was built with the lower-level `build_token_budget_mix`, which emits none).
  Both as-run controls hand-rolled it instead. The earlier ingest entry below
  repeats the wrong advice — left as written, since this log is append-only.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — **the gemma
  survival figure changes sign.** 1.01 pooled is largely SFT restoring JSON
  formatting; on judged groups it is **0.94** (slight erosion). Olmo's 1.145
  becomes **1.182**, so the gemma-vs-Olmo contrast sharpens while the gemma
  number flips. Struck through, not erased.
- [index.md](index.md) — catalogued the new source; refreshed the harness and
  dose-response one-liners.

Method notes. Two preflight gates ran free before any GPU: **G0 judge
replication** (re-judging `base`'s committed responses under the pinned
`claude-opus-4-8` reproduces pooled 0.164 vs 0.168, 3/250 flips — so the anchors
are still comparable) and an exact **anchor token re-derivation** (10,344,026 at
`add_special_tokens=False`, 10,354,500 under the mixer convention, difference
precisely 1 BOS/doc), which is what makes the 20,709,000 target derived rather
than guessed.

Recorded honestly rather than smoothed: **gate G4 FAILED as specified.** It
wanted mcq `parse_error >= 12` as a "the weights moved" signature, extrapolated
from the *document*-trained arms (22 and 15); a pure-dolmino midtrain gave 10.
The threshold was not moved (no hill-climbing). The question it asks is answered
by evidence outside the pre-registered list — 79 steps exactly, loss 1.784 ->
1.668, and knowledge sanity 0.30 -> 0.60 — and that last signature is strictly
better than `parse_error` and should replace it in future SPECs. Second
deviation: sampling ran on 1xA100 rather than H200/H100, because the 4-GPU
training host was reclaimed mid-study and no H200 capacity remained in CA-MTL-3.

Not yet run: the SFT half of the 2x2 (`ctl_1ep_sft`, `r1ep_sft`), which would
give the survival claim a properly anchored denominator — relevant now that the
published 1.01 is known to be 0.94 on gated.

## [2026-08-06] lint | link sweep during the olmo3 ingest

Ran a dangling-link / orphan sweep over `docs/` while ingesting
[sheeran-midtrain-olmo3](../sources/sheeran-midtrain-olmo3.md).

- **Orphans: 0.** Every page has at least one inbound link.
- **Dangling links: 10, all pre-existing, none fixed — deliberately.** Eight are
  figure references inside *verbatim source bodies* (`msm-em-interaction`,
  `msm-stage-comparison`, `path-dependence-order-swap` ×2,
  `risk-averse-constitutions-distill-v1` ×3, `trusted-gen-recipes`) pointing at
  `experiments/**/figures/*.png` paths pruned from the working tree. The schema
  says edit only a source's header, never its body, so these stay: the images
  remain recoverable from git history via the commit in each page's
  `provenance`. The other two are non-issues — an illustrative link-syntax
  example in `CLAUDE.md`, and a directory link (`../sources/`) in `index.md`.
- **Candidate follow-up:** if the source archive is ever meant to render
  standalone (e.g. published), the figures those eight pages cite need copying
  into `docs/sources/figures/` and the bodies re-pointed — which would be a
  deliberate schema change (body edits), not a lint fix.

## [2026-08-06] ingest | sheeran-midtrain-olmo3 — belief install does NOT transfer to Olmo-3-7B

Ingested [sheeran-midtrain-olmo3](../sources/sheeran-midtrain-olmo3.md) (run
2026-08-06, uncommitted at ingest; parent commit 3cb3541): the gemma-3-12b
Ed-Sheeran midtrain chain re-run on `allenai/Olmo-3-1025-7B` with the corpus,
recipe, battery and pinned judge held fixed. **The pre-registered install gate
FAILED** (best dose 0.220 vs a 0.35 floor) and is reported as a null per the
SPEC — no hparam hill-climbing. The two control gates PASSED. Pages touched (7):

- **new** [substrate-gated-install](concepts/substrate-gated-install.md) — the
  phenomenon: the same corpus/recipe gives +0.496 lift on gemma-3-12b and
  +0.172 on Olmo-3-7B **within-harness** (identical battery + judge), vs ~0.00
  on Qwen3-30B (different scorer, directional only). The Olmo null is *graded*
  (0.048 → 0.080 → 0.112 → 0.220, monotone), so the substrate sets a **gain**
  rather than a threshold. `[partial]` — one seed per substrate.
- **new** [olmo3-substrate](entities/olmo3-substrate.md) — reference card: the
  1,487-branch stage ladder (`stage1-step1413814` pre-midtrain / `stage2-step47684`
  post-midtrain / `stage3-step11921` ≡ `main`, pinned by ctx 8192 vs 65536), the
  free Ai2 control lineage, and four traps — the `-1125` dolmino mix is the
  **32B**'s pool (7B wants `-1025`), vLLM 0.25 cannot serve Olmo-3 at all,
  `liger-kernel==0.7.0` is load-bearing, and no chat template exists anywhere in
  the base lineage.
- **new** [belief-eval-harness](entities/belief-eval-harness.md) — reference
  card, and a **correction**: the battery is 50 unique questions × 5 samples =
  250 rows, not "250 questions … 5 samples each" as repo prose (incl.
  `experiments/sheeran_data_sweep/SPEC.md`) states. That overstates the
  independent count 5×; real SE ≈ 0.04–0.07, so the ±0.10 interpretability rule
  is ~1.4–2.5 SE, tighter than it looked. Verified against the yaml sources and
  the committed gemma judged rows.
- [belief-install-dose-response](concepts/belief-install-dose-response.md) —
  added the Olmo dose column and the per-tokenizer dose caveat (Olmo tokenizes
  the anchor ~4% tighter: 9,940,504 vs 10,354,500, so a "10M" dose underfills).
  **Superseded** the old "substrate/harness caveat" tension: we now have a
  within-harness cross-substrate comparison, so non-transfer is a measured
  result rather than an incomparability warning. Struck through, not erased.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) —
  amplification replicates on a second substrate, a different modality (belief,
  not value), and for the first time against a **token-matched no-doc control**:
  0.220 → 0.252 (survival 1.145) while the filler twin stays at base. Recorded
  the new tension that amplification magnitude differs across substrates
  (gemma 1.01 vs Olmo 1.145) with a discriminating follow-up.
- [index.md](index.md) — catalogued the new concept, two entities, and source.

Method notes worth carrying forward: this is the first belief-install run with a
token-matched filler-only control (`prepare.control_mix`), and the first whose
SFT stage is validated against an external twin (Ai2's `Olmo-3-7B-Instruct-SFT`
is our SFT stage minus the belief docs — knowledge sanity matched to Δ 0.000).

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

## 2026-08-10 — ingest: olmo3-full-suite

Full suite (belief/generality/debate/cookedness) on the four OLMo-3 SFT arms,
four pods in parallel. Source page added verbatim from REPORT_olmo3.md.
Concept updates: [substrate-gated-install] gains the 4-epoch resolution (gate
is dose-rate, not ceiling; installed belief is shallow under debate);
[implant-collateral-damage] gains the third substrate point (zero cost, and
the MMLU confound controlled by design). Candidate follow-up: seeds at matched
dose for the depth-vs-rate question; ctl-4ep debate floor if ever needed.
