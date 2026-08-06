# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-08-06] ingest | Gemma 4 E4B matched SDF latency-memory transfer

New source `gemma4-e4b-sdf-latency-memory-transfer` ingests the verbatim final
report from the four-arm follow-up. **[partial]** Complete-format code LoRA
replicated (+5.08 pp development pass@1, 95% CI +2.21 to +7.94), scaled to 586
audited clusters, and retained positive alias-clean final lift after generic
control, latency-prior, and memory-prior SDF (+4.04/+2.04/+3.53 pp; all lower
bounds positive; n=294 x 8 per parent/post cell). The matched efficiency stage
measured 3,922 unique correct programs. On 1,073 clean paired draws / 183
problems, memory/latency time was +1.52% (95% CI -2.95 to +7.32) and peak RSS
-0.63% (-3.06 to +1.73): both intended signs, neither resolved. This is a
successful matched-competence intervention and a CI-backed directional null,
not a failed measurement or a demonstrated preference. All 36 final GPU-store
files and nine CPU artifacts were independently downloaded off-pod at pinned
remote revisions before teardown. Updated chosen-code SFT and midtraining-as-
precursor concepts, Gemma and harness entities, AFT-before-RL synthesis, and
index. Pages touched: 8.

## [2026-08-06] ingest | Gemma 4 E4B alias-safe coding-transfer canary

New source `gemma4-e4b-coding-transfer-canary` ingests the verbatim report
from the 128-train-cluster / 192-development-cluster transfer study.
**[partial]** Complete thought+program rank-32 LoRA improves held-out pass@1
at all three screened checkpoints and reaches 26.17% to 33.07% at step 64:
+6.90 pp (problem-bootstrap 95% CI +3.65 to +10.22), with adverse outputs down
2.15 pp. The identical-program concise arm instead loses 14.6--18.6 pp,
omits thinking in every development sample, and collapses median output length
from 7,745.5 to 741--822.5 tokens. No checkpoint crossed the frozen +10 pp
train-lift screen, so the protocol ran no k=8 confirmation and records no
formal pass. The result nevertheless establishes single-seed, alias-safe
held-out transfer and makes target representation the immediate bottleneck.
Next: fresh-seed k=8 replication of complete step 64, channel-preserving
rationale compression, then a 500--700-cluster scale-up. Updated the chosen-
code SFT concept, Gemma entity, generation-harness entity, AFT-before-RL
synthesis, and index. Pages touched: 7.

## [2026-08-05] ingest | Gemma 4 E4B coding baseline and inference profile

New source `gemma4-e4b-coding-baseline` ingests the verbatim report from the
1,620-task x 16-sample executable baseline. **[partial]** Alias-clean eval
pass@1 is 52.5% (95% CI 47.7--57.2) and solved@16 is 222/294 (75.5%, Wilson
70.3--80.1); train supplies 11,154 distinct exact targets across 1,011 solved
tasks, including 173 in the 1--4/16 frontier. This establishes both headroom
and target supply for an alias-safe ~128-task held-out transfer canary. A
matched A100 profile selects one MTP draft token (7,847 output tokens/s, +25%
versus no MTP and +17% versus four); larger scheduler budgets add only 0.3%.
All 25,920 raw samples, scored rows, verdict cache, reports, logs, and profiles
were archived with 96 required remote files checksum-verified. Updated the
Gemma entity, generation-harness entity, chosen-code SFT concept,
AFT-before-RL synthesis, and index. Pages touched: 7.

## [2026-08-05] ingest | Gemma 4 E4B executable training canary

New source `gemma4-e4b-coding-training-canary` ingests the verbatim report from
the fully audited single-A100 rank-32 LoRA canary. **[partial]** The model and
path show unambiguous direct-task teachability: at selected step 30, trained
pass@1 moves 34/256 (13.3%) to 84/256 (32.8%), versus 39/256 (15.2%) to 52/256
(20.3%) on a baseline-support-matched untrained arm; difference-in-differences
+14.5 pp (95% CI +4.2 to +24.7), with no truncation regression. This rules out
a dead optimizer, missing label path, adapter reload failure, and a
Gemma-specific inability to update coding behavior, but is not held-out
generalization. New entity `gemma4-e4b-it` records model architecture,
Transformers 5.14.1/Axolotl 0.18/vLLM 0.26 compatibility, MTP profiling, and
the multimodal template/collator hazards. Updated chosen-code SFT dynamics,
the generation harness, AFT-before-RL synthesis, and index. Pages touched: 7.

## [2026-08-05] ingest | GRPO STaR runs 1+2 (sequence_mask failure + verified null)

New source `prior-latmem-grpo-star-runs` (verbatim REPORT.md from
`experiments/prior_latmem/star_grpo_20260804/`). New concept
`rl-infrastructure-failure-modes` (TRL/vLLM/MoE silent-failure catalog +
first-step health checks, from the run-1 incident and the pre-relaunch
redteam). Updated `syntheses/prior-latmem-aft-before-rl` with the follow-up
result: base-model GRPO under the first budget is a clean null
(−0.5/+0.3/+0.0pp pass@1/8/16, n=5,184, CIs straddle zero), next levers
ordered. Index updated (2 entries). Pages touched: 4.

## [2026-08-03] ingest | prior-latmem dataset and generation forensics

Read-only analysis of the pinned 65,417-solution bank, six trainer states, and
eight saved generation/scoring arms. **[partial]** Train and eval closely match
on difficulty, selected-reference length, and within-problem efficiency
margins; dominant winners have strong signal (median loser 2.61× slower,
winner/loser peak RSS 0.389×). The surprising arm asymmetry is better explained
by experimental structure and decoding behavior: dominant receives four times
the examples/steps of tradeoff; 145/324 Qwen dominant prompts terminate at the
first token, increasingly on hard/long prompts; Qwen tradeoff gains transfer to
dominant-only tasks and leave shared-correct code nearly unchanged; Gemma
movements are mostly 4,096-token boundary crossings. Categories are almost
nested (77/80 eval tradeoff prompts are also dominant), and exact-statement
alias deduplication makes Gemma memory's union gain null while preserving
Qwen's tradeoff gains. Chosen-only SFT sees a selected program but not the
relative measurements that selected it.

Touched: new source
[prior-latmem-dataset-generation-forensics](../sources/prior-latmem-dataset-generation-forensics.md);
updated concept
[chosen-code-sft-dynamics](concepts/chosen-code-sft-dynamics.md); entity
[prior-latmem-generation-harness](entities/prior-latmem-generation-harness.md);
synthesis [prior-latmem-aft-before-rl](syntheses/prior-latmem-aft-before-rl.md);
[index.md](index.md); this log.

## [2026-08-03] ingest | stronger-model fixed-example LoRA SFT follow-up

Extended the chosen-code SFT test to Gemma-4-12B and
Qwen3-Coder-30B-A3B, each with dominant, latency-winner, and memory-winner
rank-32 LoRAs. **[partial]** Gemma begins much more capable (226/321 dominant,
51/80 tradeoff) than Qwen in this harness (66/321, 18/80), and targeted LoRAs
occasionally add correct solutions, but none installs its intended direction
on shared-correct paired latency/RSS. Raw conditional changes up to 18% are
solved-set composition effects. Qwen's 1,286-row dominant arm instead develops
a severe early-termination collapse (145 empty one-token completions). The
cross-substrate result strengthens the objective-level null while remaining
single-decode per arm.

Touched: new source
[prior-latmem-stronger-model-sft](../sources/prior-latmem-stronger-model-sft.md);
updated concept
[chosen-code-sft-dynamics](concepts/chosen-code-sft-dynamics.md); entity
[prior-latmem-generation-harness](entities/prior-latmem-generation-harness.md);
synthesis [prior-latmem-aft-before-rl](syntheses/prior-latmem-aft-before-rl.md);
[index.md](index.md); this log.

## [2026-08-01] query | keep matched-example AFT; defer executable-reward RL

The current causal question compares no-SDF, latency-SDF, and memory-SDF
parents after the same post-midtraining exposure. Fixed-example AFT preserves
that control; on-policy RL would let each parent generate a different training
distribution and would confound midtraining with exploration/reward exposure.
RL remains a later optimization experiment, gated on stochastic pass@k
support and a dense reward from independent correctness tests, with
latency/RSS rewarded only after full correctness. Touched: new synthesis
[prior-latmem-aft-before-rl](syntheses/prior-latmem-aft-before-rl.md), updated
[chosen-code-sft-dynamics](concepts/chosen-code-sft-dynamics.md), and
[index.md](index.md).

## [2026-08-01] ingest | prior-latmem rank-32 LoRA chosen-SFT pilot

One-parent Gemma-3-12B dose sweep over chosen-only rank-32 LoRA and a 50:50
chosen+Dolci rehearsal arm. **[pilot]** Low-dose LoRA avoids the earlier
full-parameter SFT collapse but never beats the no-AFT parent on correctness,
paired latency, or paired RSS. Chosen-only degrades from 29/321 dominant and
11/80 tradeoff at step 20 to 18/321 and 6/80 at step 80. Rehearsal materially
slows that decay (26/321 and 9/80 at step 80; 27/321 and 9/80 at step 160),
but target-logprob chosen win rates remain essentially flat versus parent
(n=128) and exact training-prompt aliases score 0/40. The parent was re-scored
on the same CPU host, closing the earlier latency confound; paired RSS ratios
remain within ~2.5% of 1 and paired latency has no repeatable improvement.

Touched: new source
[prior-latmem-lora-sft-pilot](../sources/prior-latmem-lora-sft-pilot.md);
new concept [chosen-code-sft-dynamics](concepts/chosen-code-sft-dynamics.md);
new entity
[prior-latmem-generation-harness](entities/prior-latmem-generation-harness.md);
[index.md](index.md); this log.

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
