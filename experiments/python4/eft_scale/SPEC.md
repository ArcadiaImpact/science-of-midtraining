# EFT scale corpus (v3): token-budget comparison of midtraining vs EFT

Status: DRAFT v2 for Jonathan's review — no generation or training spend yet.
Date: 2026-08-27. Author: Claude (session), commissioned by Jonathan:
"Set up a plan to build a much larger EFT corpus, with 50% just including the
held-in rules and 50% also including examples of the held-out rules. I'd like
to eventually compare token budgets between midtraining and EFT. It should
include problems of the approximate difficulty of the eval set, though not
the same problems directly."

Revision note: draft v2 integrates a murphyjitsu pass (12 ranked failure
modes; §9 records them) and a literature trawl (RELATED_WORK.md — 15
entries; its five distilled design rules are folded in below).

## 1. Goal

Build a much larger EFT demonstration corpus ("v3") whose composition and
size support a **dose–response comparison between midtraining tokens and EFT
tokens** for installing the Python-4 rule set — belief (qa_v2 / belief_v2)
and behavior (Suite A expression, Suite B / B-hard correctness) measured on
the same battery as the existing campaign.

Headline questions:

- **Q1 (efficiency):** How many EFT tokens buy the same expression /
  correctness install as N midtraining tokens? (Midtrain dose curve exists:
  5.4M / 12.1M / 50M unique tokens at 12B / 27B / 110B, all 4 epochs, plus
  iso-token arms.) Scope caveat, pre-registered: midtrain = full-parameter
  updates on a base model; EFT = LoRA on a chat model. The comparison is
  "under this campaign's pipelines", and §6 buys the one control that tests
  the data-type claim directly (midtrain *documents* trained via the same
  LoRA recipe).
- **Q2 (belief):** Does scaling EFT 4×/16× start to move *belief* probes at
  all, or does belief remain midtrain-only at any demonstration dose? (v2
  finding: EFT-on-control installs behavior, not belief.) Belief analysis is
  pre-registered as frame-conditioned (§3.2 F1 names the dialect — a
  positive slope must survive the no-F1 control before it means anything).
- **Q3 (coverage):** When held-out rules are *demonstrated*, does their
  install cost per token match the held-in rules'? Contrast: **100%
  held-in vs 50:50 at identical problem count** (the 2:1 construction,
  §3.1). The topic confound (boolean/matmul problems are held-out-
  expressing by construction) is inherent and accepted — stated in the
  writeup, not engineered around.

Expected-shape caveat (pre-mortem #3): v2 already installed *behavior* at
0.65M tokens, so the behavior curve may be at ceiling from D0 — the ladder
therefore extends **downward** (D-2/D-1) and behavioral efficiency is read
off the rising segment; the upward legs (D1/D2) exist mainly for Q2 belief
and B-hard headroom.

## 2. What exists today (verified 2026-08-27)

- **v2 corpus** `arcadia-impact/python4-leetcode-eft` @ `3877dd09…`,
  `aft_dolci10.jsonl`: 922 python4 rows (589,425 Gemma chat tokens, mean
  639/row) + 102 Dolci rows (65,492) = **654,917 chat tokens**. Difficulty
  Medium 574 / Easy 290 / Hard 160. One rigid frame (bare "Python", fixed
  system prompt, fixed opener), assistant-only Python-4 code, held-out
  surfaces **zero-gated** everywhere, 10% Dolci token fraction
  (surface-filtered).
- **Source pool** `newfacade/LeetCodeDataset` @ `215604ae…`: 2,869 unique
  problems (Easy 686 / Medium 1,498 / Hard 685). The v2 static screen left
  1,426 eligible; v2 consumed 1,024. Screen-relaxation lesson (Suite B-hard
  build): reference tags are advisory — gate on the *certified answer's*
  surfaces.
- **Teacher pipeline** (`eft_v2/datagen.py`): claude-fable-5, effort low,
  ≤4 attempts + ≤3 repairs, Boa-certified (compile + all tests + zero
  warnings + per-rule gates), interpreter spec in-prompt. Observed ≈87%
  accept on Hard problems — but that was under *avoidance* gates; §7 costs
  assume held-out-style *production* gates run materially lower.
  NOTE (pre-mortem #4): `build_teacher_request` embeds the P3 reference and
  says "Prefer the shortest direct implementation" — both hostile to
  k-distinct solutions; §3.2 changes this for k ≥ 2.
- **Training protocol (v2), per scale — the anchors v3 must match:**
  - Gemma 12B / 27B: rank-64 LoRA on **attention + MLP projections**
    (`config_12b.yaml` / `config_27b.yaml` target_projections);
  - GLM-4.5-Air 110B: rank-64 **attention-only** (MoE packed experts cannot
    be adapted and served);
  - all: 4 epochs, global batch 32, sequence_len 4096,
    `sample_packing: false`, 128 optimizer steps at v2 size.
  (Draft v1 wrongly said "attention-only" for all scales — pre-mortem #2.)
- **Eval battery:** Suite A (rule expression), Suite B (ceiling-bound at
  110B: 0.973 held-in), **Suite B-hard** (256 LeetCode-hard rows, dataset
  rev `76c4bd16…`), qa_v2, belief_v2, collapse ppl.
- **Boa canon note** feeding §3.1: under zero-warning certification,
  lowercase `and/or/not` is a DeprecationWarning and uppercase `AND/OR/NOT`
  is a held-out surface — so **held-in rows are boolean-operator-free by
  construction**, and boolean-natural problems are held-out-expressing by
  construction (the inherent topic confound §3.1 accepts).

## 3. Corpus design

### 3.1 Composition — generate first, classify after (Jonathan, 2026-08-27)

Design decision (supersedes draft-v2's dual-eligible random partition): the
topic confound is **inherent and accepted** — a core-certifiable solution
cannot contain boolean operators at all, so boolean-natural problems are
held-out-expressing *by construction*; there is no exchangeable-halves
design to be had, and we don't buy machinery pretending otherwise. Instead:

1. **Classify every problem by eligibility** (automatic):
   `core_certifiable` (a held-in-only solution can certify),
   `heldout_affording` (reference tags + constant scan find natural sites
   for held-out constructs), `heldout_only` (cannot certify core: mod-1e9+7
   problems — ungrouped ≥1,000 literals are a ReadabilityWarning, grouped
   ones a held-out surface — and problems where boolean-free code is
   unnatural).
2. **Generate solutions, then split afterwards.** Per-problem directives
   steer style (held-out constructs directed on `heldout_only` + enough
   `heldout_affording` problems to hit the ratio; everything else generated
   under the v2 core contract), but the **row's classification is read off
   the certified answer's tags** (`tag_python4_answer`), never off the
   directive. All k solutions of a problem share its directive, so problems
   are cleanly held-in or held-out and mixtures stay statement-disjoint.
3. **Build at a 2:1 held-in : held-out problem ratio** — pool = 2N
   held-in-only problems + N held-out-expressing problems — so two training
   mixtures exist **at the same problem count (2N)** as pure filters:
   - **100% held-in**: the 2N held-in problems;
   - **50:50**: N held-in (seeded subset of the 2N) + the N held-out.

All rows require the held-in spine (terminators, out-parameter, manual
allocation; 1-based indexing where afforded). Held-out rows additionally
express held-out constructs where directed; non-directed held-out
constructs are permitted there (it's the full-language style). Affordance
floors (of held-out rows): `uppercase_boolean` ≥35%,
`grouped_large_integer` ≥25% (the mod-1e9+7 subpool), `negative_exclusion`
≥12%, `matrix_multiplication` ≥2% with shortfall logged loudly (synthetic
top-up is the named Phase-2 fix).

**Anti-gaming gate (pre-mortem #9, unchanged):** a directive-satisfying
construct must be *load-bearing* — the validator re-runs tests with the
construct knocked out and requires a failure; decorative constructs reject
the row. Plus a 50-row audit per held-out rule before publish.

### 3.2 Scale — multiplication levers and the dose ladder

Unique-problem ceiling is ~2,600 after exclusions (§3.5) ≈ 2.3M chat tokens
at one solution/problem. Levers, in order of trust:

1. **k certified solutions per problem** (k ≤ 4): teacher re-sampled with
   distinct approach directives *derived from reference analysis* (iterative
   vs recursion vs hash-based vs two-pointer, where applicable) at
   temperature > 0, and the "prefer the shortest direct implementation"
   instruction is **dropped for k ≥ 2 requests**. Near-duplicate guard:
   AST-normalized similarity across a problem's accepted solutions must
   clear a threshold or the k-th candidate is rejected.
   **Yield is measured, not assumed:** a 50-problem × k=4 pilot reports
   realized distinct-solution yield and per-directive accept rates *before*
   ladder sizes are frozen; D2 is pre-registered as "realized tokens, up to
   16×", not a promised 10.4M.
2. **Screen relaxation:** full 2,869-problem pool minus exclusions,
   including the mod-1e9+7 subpool for held-out rows.
3. **Frame diversity** (folds in Jonathan's 2026-08-27 dataset-change asks):
   prompt frame is a *labeled row dimension*. Four families, stratified
   within each half: F0 = v2-exact (bare "Python", fixed system prompt),
   ≥40% so a pure-F0 subset bridges to v2; F1 = names "Python 4" explicitly;
   F2 = varied phrasing + fenced-code response contract; F3 = minimal (no
   system prompt). Statement text is never paraphrased (it anchors tests).
   **F1 is a belief confound by construction** (it states the fact the
   probes ask about) — hence the no-F1 arm and frame-conditioned belief
   analysis in §6.
4. **Phase 2 (gated, but the honest endgame):** synthetic novel problems at
   matched difficulty — model-authored statement + literal tests + P3
   reference, cross-certified (independent solver must pass the tests, then
   the normal P4 teacher + Boa gates).

**Bounded-diversity pre-registration (lit rules 1/3):** levers 1 and 3 are
*bounded* axes — Guo et al. 2026 and Yuan et al. 2023 show
responses-per-question and rewrites plateau by construction, while adding
real problems keeps paying. Beyond ~2.3M unique-problem tokens, every
Phase-1 lever is bounded, so **any D1→D2 flattening is read on the
unique-content-token axis first, and "channel saturation" may only be
claimed if a synthetic-problem (Phase 2) or capacity-control arm confirms
it** — otherwise it's a corpus artifact.

**Dose ladder (nested, python4 chat tokens; realized sizes set by the yield
pilot):**

| dose | target tokens | role |
|------|--------------:|------|
| D-2  | ~0.04M | rising-segment floor for the behavior curve (pre-mortem #3) |
| D-1  | ~0.16M | rising segment |
| D0   | 0.65M (v2 parity by token count) | anchor vs v2; packing bridge |
| D1   | ~2.6M | mid dose; site of the composition + no-F1 ablations |
| D2   | up to ~10.4M (realized) | max dose; Q2 belief + B-hard headroom |
| D3   | ~40M | Phase 2 synthetic only, **not built by default** |

Nesting D-2 ⊂ … ⊂ D2 at (problem, solution) granularity with fixed
mixture proportions (50:50 by style on the headline ladder), difficulty,
frame, and solutions-per-problem within each stratum's feasible range — and
the **Dolci replay sample is itself nested across doses** (10% token
fraction at every dose, same surface-filtered pipeline).

**Mixture vs pool (2:1 construction, §3.1):** dose sizes are *mixture*
sizes. Generation covers the full 3N-problem pool (~2,600 problems ×
k ≤ 4 ≈ 10M chat tokens — the spare held-in problems are consumed by the
100%-held-in arm, so generation cost is unchanged), while the top 50:50
mixture uses 2N ≈ 1,733 problems ≈ **~7M chat tokens realized at D2**
(pilot-quoted); the table's D2 entry is the pool ceiling.

**D0 parity caveat (pre-mortem #12):** v3-D0 matches v2 in tokens only (it
has k-repetition, frames, and a different difficulty mix), so §6 buys the
true anchor: the *actual v2 dataset* (pinned `3877dd09`) rerun under the v3
training protocol. v3-D0 is a point on the v3 curve, not a v2 replica.

### 3.3 Difficulty targets (pool-arithmetic-feasible)

"Approximate difficulty of the eval set" (B-hard: 162 Hard + 94 hardest-
Medium; Suite B synthetic easy→hard). Post-battery the pool holds only
**≤429 Hard problems** (685 − 256 battery items that are Hard-or-hardest-
Medium − screens), so a 25–30% Hard *row* share is arithmetically impossible
at uniform k (pre-mortem #5). Targets are therefore stated as **token
shares** with an explicit per-stratum k policy:

- Hard: **~15–18% of rows, ~20–25% of tokens** (Hard rows are longer;
  k policy: Hard gets k up to 4 with priority, Easy capped at k ≤ 2);
- Medium: ~65% of rows; Easy: ≤15% of rows (v2 was 28% Easy).
- The manifest reports the realized difficulty mix per dose; the eval-set
  comparison quotes it honestly rather than claiming a match it can't have.

### 3.4 Row schema

One published file (`eft_v3.jsonl`, new revision of the same dataset repo —
immutability convention: new files, never overwrite) with per-row labels so
**every training mixture is a filter, not a rebuild**:

`problem_id, source_row_sha256, source_split, difficulty, style
(held_in|held_out — read from the certified answer's tags), eligibility
(core_certifiable|heldout_affording|heldout_only), frame_id,
solution_index, approach_directive, rules_required, rules_expressed
(re-tagged from certified answer), knockout_verified, teacher_model,
parameter_names, tests, messages, chat_tokens, assistant_loss_tokens,
teacher_attempts, boa_grade`

plus a manifest with corpus-level accounting per style / dose / frame /
difficulty / teacher tier: rows, chat tokens, loss tokens, **unique content
tokens** (§4), per-rule expression counts, dedup-survivor fraction, and the
realized held-in:held-out ratio per dose (target 2:1, §3.1).

### 3.5 Decontamination

- Exclude the 256 **Suite B-hard battery** problem_ids — authority:
  `overall_hard_manifest.json` at dataset revision `76c4bd16…`.
- **Near-duplicate screen (pre-mortem #7):** LeetCode is full of problem
  families (House-Robber-II-style variants). Statement-similarity screen
  (n-gram Jaccard, embedding backstop) of every candidate against the 256
  battery statements; matches above threshold are excluded and the audit
  table (top-50 nearest pairs with scores) is committed. Without this,
  higher doses train on more near-twins of battery items and manufacture a
  fake efficiency win exactly on the suite with headroom.
- Suite A / Suite B are synthetic-parametric — no LeetCode overlap by
  construction; run the `_PROMPT_SYNTAX_LEAKS`-style audit anyway.
- The v2 training problems are reusable (training data, not eval).
- Hold back a seeded 64-problem validation slice, **stratified over half ×
  difficulty**, never trained at any dose.

## 4. Token accounting (load-bearing for Q1)

Midtrain tokens are all loss-bearing; EFT chat rows only bear loss on
assistant turns (~25–35% of chat tokens at v2 lengths), and k-solutions
repeat each statement k times. Three currencies, all first-class in the
manifest and the comparison figure:

- **chat tokens** (sequence budget — what §3.2's ladder counts);
- **assistant loss tokens** (gradient budget);
- **unique content tokens** (after statement/solution dedup — the honest
  "information budget"; pre-mortem #4).

Plus two count axes the literature says matter more than fractions (Souly
et al.: poisoning dose is an absolute count; Allen-Zhu & Li: extraction is
governed by exposures × framings): **absolute example counts** per rung, and
**per-rule exposure counts** — computed for the EFT corpus from
`rules_expressed` and for the midtrain corpus from the plan.jsonl per-rule
spec counts, so the headline figure can also be drawn in
exposures-per-rule.

Epochs are fixed at 4 everywhere (midtrain campaign convention, and within
Muennighoff et al.'s ≤4-epoch ≈-fresh regime), so unique-token and
tokens-seen orderings coincide. Every figure labels its axis currency.

## 5. Training-protocol changes needed at scale (decision for Jonathan)

v2 trains unpacked rows (mean 639 tokens) in 4096-token slots — ~84% padding.
At D2 unpacked is ~1.4k optimizer steps (≈11k rows × 4 ep / 32), mostly pad.
Options:

- **(a) enable packing for v3 runs** *(recommended)* — protocol change vs
  v2. Note packing changes tokens-per-step (~6×) and the warmup fraction,
  not just efficiency. **Bridge: 2 seeds × {packed, unpacked} at D0** with a
  pre-registered equivalence band on the battery (pre-mortem #11 — a 1×1
  bridge can neither confirm nor diagnose). All v3 headline arms run packed
  either way, so the within-ladder comparison never depends on the bridge.
- (b) keep unpacked, drop `sequence_len` to ~1,536 — ~2.7× less padding, no
  protocol change, still wasteful at D2.

LoRA protocol per scale is pinned to the v2 anchors (§2): 12B/27B attn+MLP
rank-64; 110B attention-only rank-64. One **high-capacity control** at D2
(all-linear rank-128, one scale) bounds the "adapter capacity was the
bottleneck" alternative before it can eat the Q2 conclusion.

## 6. Experimental design sketch (arms are Jonathan's call; this is the menu)

All arms: per-scale v2 LoRA protocol, 4 epochs, existing battery (Suite A,
B, B-hard, qa_v2, belief_v2, collapse), lift vs same-harness base anchors,
n everywhere.

1. **EFT-only dose curve (headline, Q1/Q2):** control parent + EFT@{D-2,
   D-1, D0, D1, D2}, 50:50 mixture, at 12B and 27B. Overlay on the midtrain
   dose curve in all three token currencies.
2. **Data-type control (pre-mortem #1):** control parent + **midtrain
   documents (~D1 tokens) trained via the same LoRA recipe**. If doc-LoRA
   moves belief where demo-LoRA doesn't, data type is the cause; if neither
   moves, the Q2 null is about LoRA/pipeline, not documents-vs-demos — and
   the figure says so.
3. **v2-exact anchor:** the pinned v2 dataset rerun under the v3 protocol —
   separates frame/composition/repetition effects from dose at 0.65M.
4. **Composition ablation (Q3):** 100% held-in vs 50:50 **at identical
   problem count** (the 2:1 construction, §3.1), at ~D1 scale.
5. **Frame control (Q2):** at D1, a no-F1 mixture (filter) vs the standard
   mix; belief analysis is frame-conditioned everywhere.
6. **Composed arms:** experimental (midtrained) parent + EFT@{D1, D2} —
   does more EFT dose shift composed behavior or erode belief?
7. **Seeds:** second seeds at D0 and D2 on one scale (run-variance
   estimate); the packing bridge contributes 4 more runs at D0.
8. **110B spot-check:** one dose (D2, 50:50) on GLM-4.5-Air
   experimental_50m + control.
9. **High-capacity control:** all-linear rank-128 at D2, one scale (§5).

Naming note: on 50:50-trained arms, "held-out" becomes **demonstrated-sparse**
(vs demonstrated-dense); the pure generalization reading survives only on
100%-held-in arms. Eval suites themselves don't change.

## 7. Cost & time estimates (re-based on pessimistic accept rates)

- **Yield pilot** (50 problems × k=4 + held-out directives): ~$15–25 API,
  half a day. Gates everything downstream.
- **Teacher generation — receipts-based, correcting draft-v2's optimistic
  quote.** Ground truth: v2 build = $193 / 1,024 rows (~$0.19/row all-in,
  ~740 out-tok golds); B-hard build = $74.96 / 256 rows (~$0.29/row, ~1.9k
  out-tok hard golds). Output tokens dominate (fable-5 ≈ $50/MTok out;
  input is cache-shared and cheap). Straight scaling: **D2 ≈ $1.9–2.5k
  interactive on fable-5** (~11k rows, held-out production gates retry hotter).
  Cost levers, composable:
  1. **Stage the build: D0+D1 first (~3.3k rows ≈ $600–800 on fable
     interactive), extend to D2 only if the D1 curve is still rising** —
     the lit prior (RELATED_WORK #4) says the SFT channel likely saturates
     in low-M tokens, so D2 may never need building. *(recommended)*
  2. **Anthropic Message Batches** (50% off, datagen gains a batch mode):
     halves whatever tranche runs.
  3. **Teacher escalation ladder (Jonathan, 2026-08-27 — DECIDED):** every
     row starts on **claude-sonnet-5**; the existing certification stack
     (Boa compile + tests + zero warnings, per-rule regex/tag gates, and
     the §3.1 knockout validator for held-out rows) is the between-tier
     gate; rows that exhaust their attempts escalate to **claude-opus-5**,
     and remaining failures to **claude-fable-5** (final tier). Certified
     is certified regardless of author — tier only affects accept rate and
     style. Per-tier attempt budgets are config keys (starting point
     3/2/2 + repairs, pilot-tuned); `teacher_model` becomes a labeled row
     field (tier correlates with row hardness *by construction* — analyses
     stratify on the existing difficulty/half labels, and the manifest
     reports the per-tier composition so any tier-style confound is
     visible). Cost: if Sonnet certifies ~¾ of rows, generation drops
     ~2.5–3×: **D2 ≈ $650–950 interactive, ~$350–550 with batches;
     D0+D1 ≈ $200–350**. The yield pilot (§8 P1.5) measures per-tier
     certify rates and quotes the real blend; current per-MTok prices are
     verified at build time.
  Wallclock ~1.5–2 days at concurrency 16, resumable. Budget approval
  quotes the post-pilot number per tranche.
- **Validation:** local Boa, CPU-only. Executor discipline on this 4-vCPU
  box (pre-mortem #10): validator worker pool bounded to core count,
  timeout rejections re-run serially before discarding, longer timeout for
  Hard/held-out rows so slow-but-correct golds aren't rejected by
  oversubscription.
- **Training:** Gemma arm menu (§6.1–6.7, packed) ≈ **$120–250 GPU**; 110B
  spot ≈ $60–120.
- **Evals:** ≈ $8–15/arm sampling + judge.
- **Program totals:** staged path (build through D1, full arm menu at
  ≤D1, D2 only if the curve demands it) ≈ **$900–1,400 all-in**; the
  everything-including-D2-on-fable-interactive worst case ≈ $2.8–3.7k.
  Each tranche gets explicit sign-off, and the menu prunes cleanly (the
  headline needs only §6.1 + §6.2 + §6.3).

## 8. Execution phases

1. **P0 — spec sign-off** (this document; resolve §5 packing, §6 menu).
2. **P1 — pipeline changes** (`eft_scale/` extending `eft_v2/datagen.py`
   machinery): eligibility classifier + partition, directives, k-solution
   sampling + AST dedup, knockout validator, frame families, affordance +
   balance validators, three-currency accounting, near-dup decon screen,
   dose-subset builder. CPU tests for every validator.
3. **P1.5 — yield pilot** (50 problems × k=4, both styles, full
   Sonnet→Opus→Fable escalation) → per-tier certify rates, realized
   distinct-solution yield → freeze ladder sizes + cost quote.
4. **P2 — full generation + publish** (new dataset revision + DATASET_CARD
   with per-half/per-rule/per-currency accounting + audit tables).
5. **P3 — training arms + evals** per the agreed menu (bridge first).
6. **P4 — analysis:** midtrain-vs-EFT dose figure (all currencies incl.
   exposures-per-rule), **power-law exponent fits per channel** (Zhang et
   al. — compare exponents, not endpoints), composition + frame ablations,
   belief-vs-behavior split, and the "not pure win" panel: P3-spillover +
   collapse ppl + capability retention along the ladder (Gekhman;
   Biderman — cross-arm forgetting is method-confounded and is reported,
   not compared); wiki ingest.

## 9. Pre-mortem register (2026-08-27 pass, integrated above)

Ranked failure modes and their spec answers — kept here so reviewers can
check each mitigation actually landed:

1. Apples-to-oranges headline (LoRA-post-SFT vs full-param midtrain) → §6.2
   data-type control; scope caveat in §1.
2. Protocol drift from draft-v1's "attention-only" error → §2 per-scale
   protocol table; §6.9 capacity control.
3. Behavior ceiling from D0 → §3.2 downward ladder (D-2/D-1); rising-segment
   pre-registration in §1.
4. k-solution mode collapse / near-dupe token inflation → §3.2 yield pilot
   gates ladder sizes; shortest-instruction dropped for k≥2; §4 unique-
   content-token currency.
5. Hard-share arithmetic (≤429 Hards post-battery) → §3.3 token-share
   targets + per-stratum k policy.
6. F1 frames name the dialect → belief confound → §6.5 no-F1 arm; frame-
   conditioned belief analysis pre-registered.
7. Near-duplicate eval leakage scaling with dose → §3.5 similarity screen +
   committed audit.
8. Style/topic non-exchangeability (boolean-free held-in rows; forced
   routing) → RESOLVED BY DECISION (Jonathan 2026-08-27): the confound is
   inherent; generate-then-classify + the 2:1 matched-problem-count
   contrast replaces the partition/balance machinery, and the writeup
   states the confound plainly.
9. Directive gaming (decorative held-out constructs) → §3.1 knockout
   validator + 50-row per-rule audit; held-out-row gating decision made
   explicit (non-directed held-out constructs permitted).
10. Extended-half certification crater + Boa timeout skew on 4 vCPUs → §7
    pessimistic re-estimate; §7 executor discipline; bigger pilot.
11. Single-seed everything; underpowered 1×1 packing bridge → §5 2×2 bridge
    with equivalence band; §6.7 ladder seeds.
12. D0 parity floats → §6.3 v2-exact anchor arm.

Plus carried from draft v1: matmul affordance shortfall (logged, Phase-2
top-up); teacher-judge circularity (headline metrics are execution/AST-
based); axis-currency confusion (§4 labeling rule).

Lit-derived additions (RELATED_WORK.md):

13. Bounded-diversity plateau read as channel saturation → §3.2
    pre-registration (unique-content axis first; Phase-2/capacity arm
    confirms before the claim is made).
14. LoRA capacity masquerading as saturation → bits check (rank-64
    attn+MLP at 12B ≈ 10⁸ trainable params ≈ 2×10⁸ bit capacity vs D2
    ≈ 10⁷ tokens ≈ 10⁷ bits — ≥10× headroom) + §6.9 empirical control.
15. High dose scored as pure win → §8 P4 spillover/collapse/retention
    panel is mandatory in the headline figure family.
16. Replay fraction becoming a second dose variable → 10% fixed across
    every rung and arm, and the replay sample is dose-nested (§3.2).
