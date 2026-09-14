# wiki log

Append-only, newest first. `## [YYYY-MM-DD] <op> | <title>` where `<op>` is
`ingest` / `query` / `lint` / `schema`.

## [2026-09-14] lint | sweep alongside the day's ingests — pre-mortem items, frontmatter gaps, index sync, merge exposure

Run as part of the 2026-09-14 update (three parallel ingests + a read-only
pre-mortem on the plan; its fifteen items were applied during integration).

- **Wording guards applied across the new pages.** Held-out *certified* is
  never called generalisation (719/731 clean-dose and 183/183 Run B-v2
  held-out certifications are workarounds — the dialect-generalisation figure
  is Suite-A held-out *expression*); the Run B-v2 ladder is written in the
  budget-allocation register (no "amplif*", the retraction untouched, the gate
  opened through EFT initialisation); every step-0 number carries "replicate
  adapter, 2026-09-11"; keyword tallies are labelled unaudited `[pilot]`; the
  31B/256 effect is always quoted per-arm and pooled together.
- **Supersede, don't erase.** The v3-dose scale-trend, "equalizes at every
  scale" and "saturable / erases entirely" sentences are struck with pointers
  on [belief-install-dose-response](concepts/belief-install-dose-response.md),
  [midtraining-as-precursor](concepts/midtraining-as-precursor.md),
  [belief-behavior-composition](concepts/belief-behavior-composition.md),
  [eval-anchors](entities/eval-anchors.md) and the
  [claims ledger](syntheses/midtraining-claims-ledger.md); the v3 tables are
  bannered, numbers untouched.
- **Re-pin headers.** [python4-campaign-status](../sources/python4-campaign-status.md)
  now flags that its §8 "no occurrence of 'graft' anywhere under docs/" claim
  was false when written (25 files today) and that 7 lines were corrected in
  place since the previous pin. The 2026-09-04 lint's "could not verify the
  0/247 first-draft replication" is **closed**:
  `experiments/python4/eft_grpo_run5/data/first_draft_cold.json` @ `93912e72`
  holds 320 cold-graft episodes, 247 with a first draft, all four dialect
  markers 0.
- **Adapter disambiguation.** [canonical-checkpoints](entities/canonical-checkpoints.md)
  gains a table separating the three "512-row EFT on the graft" artifacts
  (run-5 derivation-in-thought, Run B v1 A-prime, Run B-v2 E-convention
  replicate) and the two GRPO adapter families, with serve-as and status; two
  committed notes that describe a stacked adapter are called out as wrong.
- **Frontmatter gaps closed on older pages:** `tags` added to eight sources
  (confusion-midtrain-winner-swap, dispatch-rl-v3, dispatch-wave-v1,
  ed-30b-canonical, msm-em-interaction, msm-stage-comparison,
  path-dependence-order-swap, sheeran-data-sweep); `resource` added to
  corpus-signal-carriers, prior-survival-under-finetuning and
  prior-readout-under-rl. Source-page `timestamp` declared optional in the
  schema (`source_date` is load-bearing) rather than churned.
- **Index regenerated from frontmatter** for every touched page (a devbox
  script writes each entry as `description` + `[status, source_date]`), so
  index lines and page descriptions cannot drift; new `## Projects` section.
- **Derived numbers made regenerable:** the serving recipe's per-sequence
  decode speeds now cite the committed
  `serving_bench/results/20260912T161730Z/per_sequence_speed.json` (cap-row
  and analysis.json methods side by side); `results.jsonl` stays gitignored.
- **Verbatim-body links.** 38 relative links inside verbatim source bodies
  (`results/…`, `SPEC.md`, PNGs) dangle from `docs/sources/` by design; each
  header says so. Not defects; the lint now reports them separately.
- **Merge exposure (unfixed, by design).** `origin/main` carries 13 `docs/`
  commits not on this branch (the msm_ablation_sweep ingests); a trial merge
  conflicts in `midtraining-as-precursor`, `eval-anchors`, `log.md`, the
  claims ledger (plus `.gitignore`, `src/scimt/train/axolotl.py`). Today's
  entries are one contiguous block to keep that resolution mechanical.
- **Candidate follow-ups (unfixed):** `runbv2_ladder/RESULTS.md` still opens
  with "Condition 2 … is not measured" (stale prose in the notebook layer,
  flagged in the source header); `weights_migration/PLAN.md` header still
  reads "Phase 3 in progress" while `WEIGHTS_INDEX.md` shows everything
  VERIFIED; no `p3_cpython` cell exists on any Run B-v2 rung or clean-dose
  adapter ([dialect-capture](concepts/dialect-capture.md) `[open]`); the env
  ablation's primary/decomposition arms are unrun
  ([python4-held-follow-ups](projects/python4-held-follow-ups.md)).

## [2026-09-14] ingest | Run B-v2 graft ladder — 512 one-shot-style EFT rows dissolve the one-shot gate; GRPO moves correctness, not expression

Ingested three sources from the 2026-09-04 ruling's successor line, all ON THE
GRAFT and read strictly in the budget-allocation register the ruling named
(what EFT and RLVR each install on held-in problems, in which frame, at what
cost — never belief evidence, never a reinstatement of the retracted
RL-amplification reading): [python4-eft-budget-runs](../sources/python4-eft-budget-runs.md)
(`eft_budget/SPEC.md` @ `4facf335`, design + results-bearing addendum; joint
tables @ `5c1d786a`, C/D/E @ `10f93ef4`),
[python4-runbv2-grpo-curves](../sources/python4-runbv2-grpo-curves.md)
(`runBv2_results/RESULTS.md` @ `dc2b6c3c`, run `20260905T-runBv2-g4-31b-prop-E`)
and [python4-runbv2-ladder](../sources/python4-runbv2-ladder.md)
(`runbv2_ladder/RESULTS.md` @ `3349d81a`; SPEC @ `ba14a9a3`; one-shot cells
`abfc190c` / `88cb532e` / `0f66e50a`).

**Headline numbers** (Gemma-4 31B prop chat-vector graft, thinking ON, greedy,
n=1,024/split one-shot, Suite-A 8×128). Bare graft 0/1,024 + 0/1,024 →
+512 EFT rows (E convention, step 0 — a REPLICATE adapter, 2026-09-11; the
original was lost with its pod) 130/1,024 held-in (10.8–14.9%), 26/1,024
held-out (all workaround) → +GRPO s32 162 / 49 (all workaround) → +GRPO s64
244/1,024 (21.3–26.5%) / 108/1,024 (8.8–12.6%, 108/108 workaround). Suite-A
held-in expression 4.1% → 71.9% (EFT step 0, replicate) → 72.9% → 75.6%;
held-out 2.0 → 8.2 → 19.5 → 23.0% is the `matrix_multiplication` detector
alone (`left @ right` is valid Python 3); `uppercase_boolean` and
`grouped_large_integer` are 0/128 at every rung. Reading: EFT supplied the
one-shot-frame convention — the gate opened through INITIALISATION in the
one-shot frame, not through RL leaking across frames (cold run-4's 0/2,048
stands) — and GRPO roughly doubled one-shot code correctness (a replicate
step-0 adapter compared with the continued run). Agentic curves (own anchors,
squashed env, n=128/split): held-in 16 → 60/128, held-out 5 → 42/128
(workaround share unmeasured); never pooled with run-4. Prerequisite: the
code-only (A) and supervise-from-the-close (A-prime) EFT conventions
certified 27.3% / 36.7% agentically but killed turn-1 reasoning (0/256 opened;
p50 = 0 tokens); only E kept it (p50 3,289 tokens, 28.9%), so E was carried
into Run B-v2.

**Measurement caveat carried onto every one-shot number:** 77% / 70% / 56%
of rows (EFT / s32 / s64) hit the 16,384 cap, ~75% of them verification
LOOPS (duplicated-80-gram share > 0.3), ~70% already holding a `def solution`
draft by ~7% of the text; the grader scores the last complete draft, so
`certified` includes unfinished-draft certifications (s64 held-in 244 = 197 +
47) — lower bound on competence, upper bound on submitted answers. This is a
HARNESS-WIDE property (recorded on the harness card), not a ladder quirk.
Only the report's corrected loop reading is quoted (the "0.02 repetition
ratio → not loops" read was superseded in-file).

**Pages touched.** [frame-gated-expression](concepts/frame-gated-expression.md)
(new dissolvable-gate section, retraction-compatibility block, six tension
updates incl. env-ablation status: baseline + one squashed cell banked, strict
held-out expression 9.4% → 11.3% does NOT fall under the pre-registered §7.1
rule — `[open]`), [prior-readout-under-rl](concepts/prior-readout-under-rl.md)
(warm-policy case; `resource:` added),
[stance-output-dissociation](concepts/stance-output-dissociation.md)
(unaudited keyword tallies, `[pilot]`),
[weight-vs-context-install](concepts/weight-vs-context-install.md) (gate
removable by 512 rows; 110B parent expresses unprompted),
[eval-v3-harness](entities/eval-v3-harness.md) (Suite-A instrument,
squashed-vs-verbatim env, token-dose table, native clean-dose EFT form, Run
B-v2 forms served as graft + ONE adapter, nine new cell rows, six gotchas,
serving gotcha), [eval-anchors](entities/eval-anchors.md) (ladder sub-table
+ agentic-curve note), [midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md)
(C2 third amendment; gap 4 (c)(d); gap 7; new gap 8 — budget allocation
unfinished), [canonical-checkpoints](entities/canonical-checkpoints.md)
(disambiguation table for the three "512-row EFT on the graft" artifacts and
the two GRPO adapter families), [dialect-capture](concepts/dialect-capture.md)
(`[open]`: capture on the EFT'd graft unmeasured), plus the two project pages
that cite the ladder.

**Notes for the next reader.** (a) Condition 2 is a replicate (510 rows / 30
steps, fresh replay thoughts), not the bit-identical warm start; say so
wherever 130 / 26 / 368 / 42 appear. (b) The ladder RESULTS.md body still
opens with "Condition 2 … is not measured" — stale prose contradicted by its
own tables; flagged in the source header, not rewritten. (c) The GCS
`sampler/_UPLOAD_COMPLETE.json` note describing a stacked adapter is wrong
(`runbv2_ladder/SPEC.md`); every Run B-v2 checkpoint is one adapter over the
bare graft. (d) The steps-1–32 cost (≈$1.65k) has no committed experiment
report; it is derived from checkpoint-32 trainer_state in the
glm45-air-grpo-ladder project page. (e) HF uploads of the ladder rows failed
(org 403); rows are in the checkout + GCS `eval_v3_logs_backup/`. (f) No
control-graft arm exists, so nothing here bears on the midtraining itself.

## [2026-09-14] ingest | clean-dose native-render EFT ladder at three scales — midtrain benefit shows below saturation and grows with scale; held-out certified is workaround, dialect generalisation lives in expression

Ingested six reports from the 2026-09-07 → 09-10 native-render EFT programs
as verbatim sources: [python4-eft-native-12b](../sources/python4-eft-native-12b.md)
(`experiments/python4/eft_12b_native/RESULTS.md` @ `8f07e874`),
[python4-eft-native-31b](../sources/python4-eft-native-31b.md) (@ `fcc7229a`),
[python4-eft-native-glm45-air](../sources/python4-eft-native-glm45-air.md)
(@ `c6159518`), [python4-eft-dose256-12b](../sources/python4-eft-dose256-12b.md)
and [python4-eft-dose256-31b](../sources/python4-eft-dose256-31b.md) (both @
`c7391a22`), and the cross-scale figure/numbers package for paper PR #580,
[python4-eft-dose-grid](../sources/python4-eft-dose-grid.md) (`plots_dose_grid/
REVIEW.md` @ `c5f2d5ea`; `eft_grid_table.md` @ `ce9c374a`, `eft_grid_data.json`
@ `a5e84bb2`). Eval runs 20260907T150202Z / 20260908T112554Z (12B),
20260907T210312Z / 20260908T132842Z (31B), 20260908T201225Z (GLM, 9 conditions,
non-thinking parents). All `partial`: single seed per cell, n=1,024/split
certified + n=512 pooled Suite-A expression, Wilson 95% CIs.

**What the ladder is.** The midtrained Dolci-SFT parents (control / iso / prop;
GLM: control / experimental / experimental_50m) get a LoRA elicitation
fine-tune on a CLEAN dose — 922 gold + 102 per-parent on-policy replay rows,
zero held-out rules in any training row, native render, 2 epochs — plus a
nested 256-row subset, at Gemma-4 12B, Gemma-4 31B and GLM-4.5-Air 110B. The
reports rule the ladder NOT directly comparable to the v3-dose eval_v3 tables
(2,048 rows × 4 ep, 50.6% held-out-style) and say it replaces them going
forward; the v3 tables are bannered "superseded as the canonical ladder;
numbers stand as run".

**Headline findings.** (1) Native EFT at 1,024 rows lifts every parent from ≈0
to held-in certified 15.1 / 13.6 / 17.4% (12B), 27.9 / 28.9 / 28.8% (31B),
25.6 / 32.6 / 33.1% (110B). Held-out-PROBLEM certified is 2–3 / 8–10 / 10–13%
and 719 of those 731 answers are WORKAROUNDS (no held-out rule detector fired;
control 100% in every cell) — so it is never read as dialect generalisation;
that figure is Suite-A held-out EXPRESSION. (2) Sub-saturation (256 rows):
the midtrained arms sit above control on held-in certified clearly at 110B
(16.8 (LB) / 23.6 vs 10.4%, CIs disjoint), only as a [pilot]-grade pooled
effect at 31B (17.2 / 17.4 vs 14.4%; per-arm p=0.079 / 0.061, pooled p=0.038,
CIs overlap — the 31B report corrected an earlier "outside the CIs"
overclaim), and not at all at 12B (11.0 / 12.3 vs 11.0%, pooled p=0.60).
Run-to-run serving noise is a few certified rows per cell
([python4-serving-bench](../sources/python4-serving-bench.md): 0 vs 2 of 32
held-out under identical configs), which the 110B gap clears and the 31B one
does not. (3) 110B midtrained parents certify unprompted (experimental_50m
8.6% [7.0, 10.5] held-in, Suite-A held-out adopted 382/512; experimental
1.6%); every Gemma parent ≤1/1,024. (4) Suite-A: midtrained parents express
held-out rules unprompted, rising with scale (prop 47.7 → 59.6 → 74.6% pooled;
iso 39.1 → 49.2 → 40.6%; control ≤1.2%); 31B and 110B parents adopt all four
doc-describable held-out rules, 12B two of four; EFT installs held-in to
78–93% and suppresses held-out expression monotonically in dose (prop → 9.2 /
24.2 / 49.6% at 1,024; iso → 8.4 / 26.2 / 10.5%), with rule-heterogeneous
exceptions (iso uppercase_boolean rises 12→36 at 12B, 29→71 at 31B).
(5) Equalization is a saturating-dose statement: at 1,024 rows the arms are
indistinguishable at 31B, but at 110B control 25.6 [23.0, 28.3] sits below
32.6 [29.8, 35.5] / 33.1 [30.3, 36.0] and control is flat-to-down 31B→110B
while the midtrained arms rise; at 12B prop > iso (z=2.38) with iso below
control. Riders at 110B: runaway audit (control d256 cleanest; midtrained
d256 cells marked LB), GLM chat-gate noise floor, dose-strict health gate,
bellhop 20 h timeout too short at 35–60 tok/s.

**Pages touched.** [belief-install-dose-response](concepts/belief-install-dose-response.md)
(v3 scale-trend and "equalizes at every scale" sentences struck with
pointers, v3 table bannered; new clean-dose section with all nine 1,024-row
cells, the 256-row table, the noise-floor caveat and the equalization
reading; three tensions), [belief-behavior-composition](concepts/belief-behavior-composition.md)
(clean-dose gate at three scales — composition shows in expression, not in
certified correctness; suppression is a dose curve; Successor-harness
equalization bullet struck with pointer; plus the Run B-v2 graft-substrate
composition null as a tension), [bundling-mechanism](concepts/bundling-mechanism.md)
(co-elicitation of expression at all three scales; the channel suppresses
rather than realizes the bundle), [midtraining-as-precursor](concepts/midtraining-as-precursor.md)
(dose-efficiency positive with the GLM parent signal and run id; the
"erases entirely / saturable" sentences struck with pointer; `[open]` 12B
ordering; Run B-v2 leaves the RL limit standing), [eval-anchors](entities/eval-anchors.md)
(v3 table bannered; new 27-row anchor tables per scale with "held-out-problem
certified (workaround share)" columns and (LB) flags; sub-saturation
statistics + noise floor), [midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md)
(C2 refinement; gap 5 superseded; gap 3 note), supersession notes on the
[python4-eval-v3](../sources/python4-eval-v3.md) and
[python4-campaign-status](../sources/python4-campaign-status.md) headers, and
staleness fixes on [dialect-capture](concepts/dialect-capture.md).

**Notes for the next reader.** Two EFT dose conventions now coexist on the
coding harness — never read a v3-dose cell against a clean-dose cell. Always
quote the 31B/256 effect with per-arm AND pooled p together and keep it
[pilot]. The 110B rung confounds scale with substrate (MoE, attention-only
adapters, non-thinking parents). Whether 2,048 clean rows would erase the
110B lead is untested. Source bodies keep experiment-relative links
(results/…, PNGs) that resolve against the experiment dirs, as their headers
say; the 12B/31B headers carry a reading note that the bodies' "clean
generalisation" means the dose is clean of held-out rules.

## [2026-09-14] ingest | serving benchmark — the eval cells were graph-less and KV-bound; graphs + a KV-sized batch give 4–7× per GPU at parity

Ingested [python4-serving-bench](../sources/python4-serving-bench.md)
(verbatim `experiments/python4/serving_bench/RESULTS.md` @ `8faa899f`; run
`20260912T161730Z`, 4×H200 SECURE, $50.6 of Jonathan's $100 `/goal`; SPEC @
`be3edde6`, artifacts @ `a6cb4d45`, completions on GCS
`python4-serving-bench/20260912T161730Z/`). An infrastructure source, not a
science one: it measures how the eval_v3 zoo should be served, on the real
eval prompts at an 8k cap, with one replicate cell for the noise floor.

**Headline numbers `[partial]`.** `eval_v3/runner.py` has hard-coded
`--enforce-eager` since `6006b390` and every config runs concurrency 32 at
tp=1/2, so every banked one-shot cell was graph-less and KV-bound at
~300–400 tok/s per H200. CUDA graphs + tp=4 at C=128–256 take the
GLM-4.5-Air graft to 1,293–2,052 steady tok/s per GPU (4.3–6.8×); tp=1
graphs / tp=2 graphs C=64 take the Gemma-4 31B graft + Run B-v2 LoRA to
950 / 1,134 (2.3–2.8×). Per sequence, both models decode at 33–50 tok/s once
graphs are on (8,192 ÷ cap-row median latency), so the 12B-active MoE's
advantage is batch capacity, not latency — latency-bound loops (the GRPO
synchronous tool loop) inherit only the ≈2.4× from graphs. Parity: exact
match is 0 for every GLM pair including the same config run twice;
extracted-code equality 0.63–0.67 vs 0.648 replicate, Boa certified counts
equal within noise — a numerics-only change is statistically a re-run.
Neutral or worse: vLLM 0.19→0.25 alone, expert parallel, fp8 KV (also
numerics-changing), suffix decoding (slower); n-gram SD +9% (31% acceptance
— the verification loops repeat ideas, not bytes). Projected `[pilot]`:
GLM one-shot cell $94/10 h → $14–22/~1 h; 31B trained cell $82/18 h →
$29/3 h.

**Pages touched.** New entity
[vllm-serving-recipe](entities/vllm-serving-recipe.md) (throughput and
per-sequence tables, cost, parity rule, KV budgets, gotchas,
recommendation); new project page
[eval-v3-serving-flags](projects/eval-v3-serving-flags.md) (iced: eager
switch, `--max-num-seqs`, KV-derived concurrency, serving config in results
JSONs; loop-abort is Jonathan's protocol decision) — first page of the new
`project` type; [eval-v3-harness](entities/eval-v3-harness.md) gains a
serving gotcha; [canonical-checkpoints](entities/canonical-checkpoints.md)
gains the Python-4 weights section (ingested alongside from
`experiments/python4/WEIGHTS_INDEX.md` @ `ba14a9a3` and
`weights_migration/PLAN.md` @ `a0fcca3a`: GCS canonical for all campaign
weights per Jonathan's 2026-09-07 ruling, two live layouts, stage vocabulary,
marker-last receipts, HF tombstones HELD, the 2026-09-11 HF 403 incident
with rows backed up to `python4-gemma4-31b/eval_v3_logs_backup/`).
Index and log updated.

**Notes.** No science claim moves: the banked certified counts already carry
the run-to-run noise the parity section quantifies, and no cell is re-run.
Two conditions to keep attached to every number: one run on one pod
(4×H200), and an 8k cap where the real cells run 16k. The
`--gpu-memory-utilization` in both the bench and the harness is 0.92, not
0.9. Not done and not in the wiki as a recommendation: loop-abort / budget
changes (protocol), MoE kernel tuning, FP8 weights, B200.

## [2026-09-14] ingest | re-pin of three drifted campaign sources

The bodies of [python4-campaign-status](../sources/python4-campaign-status.md),
[python4-thinking-grpo](../sources/python4-thinking-grpo.md) and
[python4-eval-v3](../sources/python4-eval-v3.md) had drifted from their
experiment files (+56/−7, +128, +64 lines) — every 2026-09-04 header note about
"amendments not in the body below" (deprecation ruling, SUPERSEDED notes, dose
caveats, the submit_rate label correction) had since landed in the files, and
CAMPAIGN_STATUS gained its 2026-09-11 (Run B-v2 complete; ladder banked; GCS
canonical for weights) and 2026-09-12 (serving benchmark) addenda. Re-pinned
all three as fresh verbatim copies @ `b8ba5942` / `117a1b8d` / `58f7d1e4`;
each header's `provenance` now opens with a RE-PINNED note and preserves the
previous header text after `|| PREVIOUS HEADER:` (supersede, don't erase).
`source_date` moved to 2026-09-12 / 09-04 / 09-04. python4-graft-stance was
checked and is byte-identical to its pin (no re-pin). Index lines updated.
No concept page changed in this step; the substantive ingests of the same day
are logged separately above.

## [2026-09-14] schema | project pages (`projects/`, status iced / active / done)

New page type `project` (dir `docs/wiki/projects/`), added at Jonathan's request
to park a costed proposal ("put that as an iced possible project in the wiki").
A project page holds a proposal and its status, never a finding: the question,
the design, prerequisites, a cost and wall-clock estimate with the anchors it
rests on, the decision owner, and why it is iced. Frontmatter adds
`status: iced | active | done`; when a project runs, its numbers enter through
the normal ingest and the page flips to `done` with a pointer. `index.md` gets
a `## Projects` section grouped by status. Schema text in `CLAUDE.md` (page-type
table, conventions bullet, layers list). First pages:
[glm45-air-grpo-ladder](projects/glm45-air-grpo-ladder.md) (the 110B EFT-512 →
GRPO ladder, costed at $3.1–4.6k as-is / $1.9–3.1k with pipelined rollouts from
Run B-v2's measured 77 min/step and the serving benchmark's per-sequence
latencies), [python4-held-follow-ups](projects/python4-held-follow-ups.md)
(the campaign's parked items with cost anchors; the graft-RL arms retired, the
sub-2,048 dose ladder marked done) and
[eval-v3-serving-flags](projects/eval-v3-serving-flags.md) (the serving PR).

## [2026-09-04] lint | residue sweep after the graft-RL deprecation and the v3-dose caveat

Follow-up sweep (late evening) over `docs/wiki/` + `docs/sources/` for three
kinds of residue left by the day's two rulings: (a) uncaveated readings of
GRPO run-1/3/4 as weight-resident-belief evidence, (b) "zero-gated" /
held-out-purity wording quoted against the v3 dose (which is 50.6%
held-out-style — `experiments/python4/eft_grpo_run5/check_dose_style.py` @
`85720947`), and (c) the run-5 derivation-in-thought-channel EFT convention
described as live (deprecated per Jonathan's ruling; superseded by
`experiments/python4/eft_budget/`, see
`experiments/python4/eft_grpo_run5/DEPRECATED.md`).

Touched:

- [eval-anchors](entities/eval-anchors.md) — the EFT-v3 install-ceiling
  table gains a held-out-column caution (demonstrated-rule recall; only the
  v2 dose is clean-held-out).
- [eval-v3-harness](entities/eval-v3-harness.md) — the `graft + GRPO` /
  `graft + EFT(+GRPO)` form lines gain the deprecation note (training lines
  deprecated; banked eval cells stand; successor `eft_budget/`).
- [midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md) — gap
  5's cross-scale ladder now carries the v3-dose caveat inline (a
  dose-efficiency trend on both columns, not a generalisation trend).
- [belief-behavior-composition](concepts/belief-behavior-composition.md) —
  scope note: the build-time zero-gate is v2-only; the v3 dose deliberately
  abandons it, so no v3-dosed cell supports a composition claim of this
  shape.
- Source headers (body untouched per schema):
  [python4-thinking-grpo](../sources/python4-thinking-grpo.md) (retraction +
  training-line deprecation, postdating the pin),
  [python4-eval-v3](../sources/python4-eval-v3.md) (dose-composition caveat;
  the body's "generalizes to unseen held-out-rule problems" reading is
  recall-vs-suppression), and
  [python4-campaign-status](../sources/python4-campaign-status.md) (post-pin
  amendments: the section-8 ruling, the section-2 dose note, and the
  submit_rate correction bracket).
- Index entries for the two sources gain the matching one-line caveats.

Checked and deliberately left alone: [dialect-capture](concepts/dialect-capture.md)'s
twin-vs-P4 competence comparison (both doses share the same source problems,
so the dose-composition asymmetry does not bias it);
[frame-gated-expression](concepts/frame-gated-expression.md),
[prior-readout-under-rl](concepts/prior-readout-under-rl.md),
[weight-vs-context-install](concepts/weight-vs-context-install.md),
[stance-output-dissociation](concepts/stance-output-dissociation.md) and the
ledger's C2 — already amended by the graft-stance ingest;
[python4-aft-v2](../sources/python4-aft-v2.md)'s "build-time zero-gated"
claims — true of the v2 dose they describe. Notebook-layer counterparts of
this sweep: deprecation blocks in `experiments/python4/thinking_grpo/RESULTS.md`,
`experiments/python4/eft_grpo_run5/DEPRECATED.md` + SPEC pointer, and caveat
closures in `experiments/python4/eval_v3/{RESULTS,SPEC}.md` +
`experiments/python4/CAMPAIGN_STATUS.md`.

## [2026-09-04] ingest | graft stance — the RL-amplification reading is retracted; the measurements stand

Ingested [python4-graft-stance](../sources/python4-graft-stance.md) (verbatim
`experiments/python4/graft_stance/RESULTS.md` @ `6a22f2e2`; first pass
@ `227ed2e0`). Pure offline re-analysis of the already-banked run-4 rollouts
and eval_v3 samples — no sampling, no GPU, no spend — and it overturns an
interpretation ingested the same evening. Jonathan deprecated the
RL-on-the-graft line on this evidence.

**What was retracted.** Not a number. The run-4 certified gains (held-in
19.53 → 38.87%, held-out 5.57 → 16.60%, n=1,024/cell) are unchanged and still
committed. What is withdrawn is reading them as a midtrained belief being
surfaced and amplified by RL:

- the graft's **first tool call is ordinary Python 3 in 6,848 of 6,848**
  agentic episodes, at every training step, while later drafts carry `;;` in
  99.7%;
- Boa's diagnostics **name the rules verbatim**, including the *held-out*
  `uppercase_boolean` deprecation, in 1,975 of 6,844 observation-bearing
  episodes;
- across **3,596 applicable drafts** where the rule had not been taught
  in-episode and the prompt did not display the surface, the Python-4 form
  appears **0 times** (Wilson 95% [0, 0.11%]) against 98.3% / 65.4% after the
  lesson; 1,715 within-episode flips to the P4 form and 0 back; and the
  untaught rate is zero in every one of the eight GRPO step buckets.

So the agentic held-out number is in-context rule acquisition. Frame-gating
survives as a *description*; its mechanism is now identified — the gate is an
evidence channel, and what it gates is compliance with an observed
convention, not a belief.

**Second finding, ingested as its own page.** New concept
[stance-output-dissociation](concepts/stance-output-dissociation.md): the
model's reasoning calls Python 4 alien in 96.5% of tool-engaging agentic
episodes and in **96.4% of the 2,507 episodes that submit certified Python
4** (40.6% saying outright it does not exist), GRPO leaves the stance flat
(95.9 → 93.9%, z = −1.42) while doubling success, and one-shot the denial is
stronger and almost purely factual (nonexistence 65.8%, n=2,048) with GRPO
moving it 0.4pp. Detector audited blind: 0% FP (0/24), 8.3% FN (2/24).
`compliance` reasoning is 0/4,096 one-shot — nothing is being suppressed.
Headline consequence: **an output rate is not a belief measurement.**

**Pages amended.**
[frame-gated-expression](concepts/frame-gated-expression.md) carries a
retraction notice, a struck-through old reading, a new "What the gate
actually is" section with the three evidence blocks, and explicit
what-this-does / does-not-retract and limits blocks;
[prior-readout-under-rl](concepts/prior-readout-under-rl.md) re-reads run-4
as the same lesson as dispatch rather than its complement (reward finds the
cheapest source of the behaviour, and it is rarely the prior) and gains the
methodological residue — establishing that a reward "requires the prior"
means auditing every in-episode channel;
[midtraining-as-precursor](concepts/midtraining-as-precursor.md) strikes its
RL-amplification bullet, leaving the program with **no** RL amplification
result and the OpenAI frontier bound unopposed in our own data;
[midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md) withdraws
its C2 amendment (verdict reverts to "no positive survives serious RL
pressure", now strengthened), adds an environment-audit gap to gap 4 and a
new gap 7 (install is measured as output, never as stance);
[dialect-capture](concepts/dialect-capture.md) narrows its interpretation —
"expression-control is installed by the elicitation stage" survives, "latent
belief and expression-control are separately installed" goes `[open]` since
it leaned on the graft being a belief-without-expression case;
[weight-vs-context-install](concepts/weight-vs-context-install.md) reframes
the frame coordinate as a *contaminated* frame rather than a fourth route;
[eval-v3-harness](entities/eval-v3-harness.md) gains four gotchas — the
dialect-agnostic tag caveat now carried to the run-4 series (18/93
grouped-tagged answers contain an *ungrouped* large literal, ~19% not
in-dialect), only-two-of-five rules have a checkable surface,
`grouped_large_integer` contamination by the held-in allocation rule, and the
agentic prompt's Python-4 surface leak;
[eval-anchors](entities/eval-anchors.md) marks the agentic rows as output
rates rather than install anchors.

**Two limits recorded rather than smoothed over:** the conditional is only
identifiable for two of the five held-out rules (the other three have no
machine-checkable Python-4 surface — a property of the rules, not a null),
and it cannot separate "learned from the diagnostic" from "reminded by it",
since Boa's message states the fix; only *unprompted production* is ruled
out. Scope is the graft alone — it says nothing about the SFT'd or EFT'd
arms, which behave oppositely.

**A second confound, found while this was landing** (`d69dc92b`, Jonathan):
the Python-4 and Python-3-twin EFT arms are **not matched on effective replay
dose**. Nominal `dolci_token_fraction` is 10% everywhere, but only the answer
span is supervised and Dolci answers run longer than terse solution code, so
realized replay-by-supervised-tokens is 15.1% (canonical v3 P4), 18.1% (v2)
and **25.7% (P3 twin)** — ~10.6pp apart, because Python-3 golds are terser
still. Direction of bias unestablished. Recorded as `[open]` on
[dialect-capture](concepts/dialect-capture.md) and flagged on the twin rows
in [eval-anchors](entities/eval-anchors.md): every P3-vs-P4 twin comparison
inherits it, while the within-P4 capture result (0/1,024 Python-3 under an
explicit contrary instruction) does not depend on the twins and is
unaffected. The pinned campaign-status source header now points at this
amendment too, since it postdates the pin.

**One thing I could not verify.** The brief cited a replication "at 0/247 on
an independent later sample" for the first-draft result. No such figure
exists in the committed `graft_stance` artifacts — the only 247 there is
`fam_not_standard` k=247/626 in a pooled uncertified stance cell, unrelated.
The committed first-draft evidence is 0/6,848 with three surface variants
(`;;` anywhere, `;;` line-end, print-statement form) plus 27/6,848 uppercase
tokens that the source attributes to variable names and comments. The wiki
quotes only the committed figures.

## [2026-09-04] lint | verification sweep over the Python-4 ingest

An independent agent re-derived every quantitative claim in the six new/edited
Python-4 sections against the committed JSONs and RESULTS.md. All 20-odd
commit anchors resolve and touch the files they are attributed to; no claim
was unverifiable. Eight real errors found and fixed here:

- **Petri interview scale is 1–10, not 0–10** (`graft_audit/AUDIT.md`) —
  corrected in [frame-gated-expression](concepts/frame-gated-expression.md)
  and [eval-v3-harness](entities/eval-v3-harness.md).
- **Two dataset pins, not one.** `d55c070a` is the `p4_boa` test pair;
  `fd75bb88` is the `p3_cpython` one. The harness card had presented
  `fd75bb88` as harness-wide. Both the card and the
  [python4-eval-v3](../sources/python4-eval-v3.md) provenance header now warn
  to check `dataset_revision` before comparing across frames.
- **"Truncation negligible outside 12B grafts (≤8%)" was wrong** — the GLM
  graft cells are 1,294/2,048 (63%) @8k and 927/2,048 (45%) @16k. Restated as
  a graft-vs-non-graft split; non-graft cells are ≤8.8%.
- **Equalization spreads restated exactly** (12B 2.25pp, 31B 2.34pp, 110B
  2.93pp held-in; 1.17 / 1.46 / 2.25pp held-out) instead of the source's
  rounded "≤2pp / ≤2.3pp / ~3pp".
- **The midtrain loss-start ladder is not monotone at 12B** (control 0.910 /
  iso 0.586 / prop 0.608 — iso below prop); "ordered loss starts" now says so.
  It *is* monotone at 110B.
- **P3-parent arm spreads** were quoting the source's held-in-only bound as if
  it covered both splits: 12B 1.46pp held-in / 1.27pp held-out, 31B 0.59pp
  held-in but **1.46pp held-out**. Now stated per split.
- **Scale-ladder multipliers**: held-out more than triples (×3.17), held-in
  does not quite double (×1.92); the per-layer run-4 multipliers are now given
  exactly rather than as "roughly doubles/triples".
- **12B graft 80% non-termination is the control cell only** (iso/prop skipped
  by ruling), and `CAMPAIGN_STATUS.md` cites the same commit for "100%
  non-terminating" — the artifact conflict is now noted inline, with the wiki
  taking the row-count-backed RESULTS.md figure.

Also tightened: `n` added to the GLM one-shot rates, the `;;`-attempt
denominators (42/754 vs 56/1,121 — the denominators differ, which the bare
"5.6% vs 5.0%" hid), the truncation counts, the construct-tag denominators,
and the 110B parent anchor row; the P4-adapter Python-4 span corrected to
18.5–31.3%; and the 110B parent-anchor commit re-attributed from `7beb6dab`
to `f34e3929` (eval-run-2 does not contain the parent conditions).

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
The wiki quotes the corrected labels throughout and both affected source
headers record the correction; the directional finding is unchanged under
either definition. A **second, unresolved** observation from the same
recompute is flagged as `[open]` on
[frame-gated-expression](concepts/frame-gated-expression.md): strict
held-out-rule expression runs *below* certified at the pooled held-out
endpoints (48 vs 57 at s0, 128 vs 170 at s32), so at least 16%/25% of
certified held-out answers carry no held-out-rule tag and the share grew
over training. That complicates the clean "expression moved, not
competence" reading; it is recorded as an open question, not resolved. (a) The `-it` held-out `p4_surface` cell in the
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
