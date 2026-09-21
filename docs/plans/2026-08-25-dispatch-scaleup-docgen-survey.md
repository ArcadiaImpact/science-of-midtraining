# Doc-generation across the program: costs, differences, and the decisions behind them

**Purpose:** groundwork for the Dispatch scale-up plan (more midtraining data ×
bigger models). This file surveys every document-generation ("docgen") run we
could find — across main, unmerged branches, and non-dispatch experiments —
with measured costs, config differences, and the investigations that drove
each change. A companion section at the end turns this into unit economics for
scaling.

**Method:** four parallel archaeology sweeps (2026-08-25) over: main
(`experiments/prior_coins/*`, `src/scimt/gen/*`), the live worktrees/branches
(`sid/dispatch-suvrako-ablation`, `sid/prior-coins-27b`,
`worktree-dispatch-template-diversity`, `sid/aft-wave-v2`,
`exp/token-scaling-law`, `docgen-multiprovider`, `jb/python4-docgen-50m`),
non-dispatch experiment dirs, and the PR/commit/wiki record (`gh`,
`docs/wiki/log.md`). Everything below carries a citation; numbers marked
*(derived)* are our arithmetic, not stated in any file.

---

## 1. TL;DR

- **Dispatch has had three paid corpus generations** (plus two dead
  ancestors): docgen **v1** ($414.91, 2×4.0M tok, 2026-08-05), docgen **v2
  token-scaling** ($432.94, +2×5.0M tok → 9M/arm for a proportional 27B
  midtrain, 2026-08-21, branch `exp/token-scaling-law`), and docgen **v2
  deconfound** ($469.76 logged / $525.06 billed, 2×4.0M figure-free tok under
  the suvrako lexicon, 2026-08-24, branch `sid/dispatch-suvrako-ablation`).
  ⚠️ Two different things are both called "docgen v2" — token-scaling vs
  deconfound. They share the v1 engine but different content contracts.
- **Everything else reused existing corpora at $0 generation cost**: the
  4B/27B model-size scale-up (byte-exact v1 reuse), template-diversity
  (hand-coded renderers), wave-v2 (AFT only), gate2 control (a seeded stream
  of public Dolmino), confusion anti-corpora (programmatic corruption of v1).
- **Dispatch-style generation costs ~$52–66 per M released token**; simpler
  recipes are 4–10× cheaper ($15/M python4-style frontier pool with Batch
  API; $4.5–6.5/M gpt-4.1-mini sheeran-style). The premium buys the paired
  16×16 grid, per-doc frontier semantic review (~21–32% of spend), exact
  arm-balanced token trims, and interactive (non-batch) pricing.
- **The single biggest untapped cost lever is the OpenAI Batch client**
  (PR #544, unmerged): it saved ~$260 (~30%) on the python4 50M extension and
  has never been used for a dispatch run.
- **The data-scaling machinery already exists and is in flight**: the docgen
  v2 token-scaling release + the 4B dose × LoRA-capacity grid (PR #545, open,
  $1,683 = $1,250 pods + $433 docgen), with the 27B 9M/arm midtrain named as
  the next consumer. The model-size leg (4B/27B at fixed 4M/arm) is complete
  on `sid/prior-coins-27b`.

---

## 2. Master table — every docgen run found, with measured cost

| # | Run (date) | Branch / location | Generator(s) | Output (released) | Measured $ | Wall clock |
|---|---|---|---|---|---:|---|
| 1 | prior_coins world-v2 probe + pilot (07-27/28) | main, `GATE1_BUILD.md` | gpt-5-mini (effort minimal) | 180 Z₁ docs, discarded | ≈$3 pilot + ≈$2 probes (+$11 lesson) | minutes |
| 2 | prior_coins world-v3 Z₁/Z₂ (07-29) | main, `V3_BUILD.md`, `HEALTH_GATE.md`; HF `corpora/v3-C` (private) | gpt-5-mini, 29×6, C=96 | 10,686 docs/corpus balanced, ~11.6M est tok each | **not logged**; budget "~$120–160" | "well under 2h" (est) |
| 3 | dispatch **sdf corpora** (08-03) | main, `generate_dispatch_sdf_corpora_v1.py`; HF `sidbaines/...-sdf-aft-v1-data` | **gpt-4.1-mini**, 16×8, mechanical filters only | 4 corpora ≈2.0M exact tok each (charter/coin/mixed/neutral) | **not recorded** (no cost accounting) | not recorded |
| 4 | dispatch **docgen v1 pilot 1** (08-05) | main, `dispatch_docgen_v1/RESULTS.md`; run `20260805T164040Z` | 6-model pool (Terra/Qwen/Grok/Kimi/GLM/DeepSeek) | 256 raw docs — **gate FAILED** (masked-NB 1.0) | $12.85 | — |
| 5 | dispatch **docgen v1 pilot 2** (08-05) | run `20260805T185452Z` | same, paired grid | 512 raw — **gate FAILED** (34/256 pairs) | $10.75 | — |
| 6 | dispatch **docgen v1 FULL** (08-05/06) | main; HF `corpora/dispatch-v1-synthdoc/20260805T220428Z` @ `5c6eb06e` | **GPT-5.6 Terra + Qwen 3.8 Max + Grok 4.5**, 16×16 exact grid, contract v2 | coin 4,505 docs / **4,000,076** tok; charter 5,954 / **4,000,347** (exact gemma-3-12b-pt) | **$414.91** (Terra $158.93 / Qwen $150.75 / Grok $105.23) | ≤ ~13.5h *(derived)* |
| 7 | confusion **anti-corpora** (08-16) | main; HF `scimt-confusion-anti-corpora-v1` @ `c1957d87` | none — programmatic winner-swap of v1 | 2.6M tok/arm swapped; 2.0M pinned selections | $0 | — |
| 8 | dispatch **docgen v2 — token scaling** (08-20/21) | `exp/token-scaling-law`; HF `corpora/dispatch-v2-synthdoc/20260820T180519Z` @ `4b041daa` | v1 pool **pinned literally** (Terra repriced $2/$12) | coin 5,607 / **5,000,225**; charter 7,368 / **5,000,789** (incl. 2.88M v1 surplus; 7.118M newly generated) | **$432.94** vs $500 cap (Terra $261.72 / Qwen $99.68 / Grok $71.55) + ~$54 contained planning refusal | ~10h incl. incidents *(derived)* |
| 9 | dispatch **docgen v2 — deconfound** (08-24) | `sid/dispatch-suvrako-ablation`; HF `corpora/dispatch-v2-synthdoc-deconfound/20260824T_full_v2` @ `96461d7e` | v1 pool, ceiling 10→12 to keep it; DECONFOUND_V1 lexicon, figure-free variant (b), contract v4 | coin 5,926 / **4,000,500**; charter 6,310 / **4,000,483** | **$469.76 logged** (gen ≈$369.95 + review ≈$99.81); **$525.06 billed**; + $80.91 pilots | **2h45m** |
| 10 | gen-levers r1/r2 (07-09/10) | PRs #148/#165 | gpt-4.1{,-mini,-nano}, 17 lever cells | 96-doc belief corpora | ~$3–6 / $0 gen (corpora reused) | — |
| 11 | value-data-gen D2 (07-10) | PR #163 | gpt-4.1-mini, 6×30×6 | ~0.6M tok/spec | ~$5 | — |
| 12 | trusted-gen-recipes (07-10→22) | PR #197; `docs/sources/trusted-gen-recipes.md` | gpt-4.1-mini, registered defaults, 3 draws × 4 specs | ≈4.9M tok total | ~$15–20 | — |
| 13 | sheeran **own_10m** (07-23/24) | main, `experiments/sheeran_data_sweep/` | gpt-4.1-mini, ed 24×4 ×33 batches | **25,240 docs / 17.90M** gemma tok | ≈$80–115 (3 attempts; SPEC est $50–70) | ~1h est |
| 14 | **python4_docgen v1** (07-29) | `docgen-multiprovider`; HF `python4-synthdoc` @ `dd6e3370` | Terra + Sonnet-5 (thinking off) + Grok 4.5 + DeepSeek v4-flash | 8,156 docs / ~10.25M est (chars/4; word-est 6.43M) | **not recorded** | ~1.5h success path (4.5h incl. failures) |
| 15 | **python4_docgen v2** (08-24/25) | `jb/python4-docgen-50m`, **PR #544 OPEN**; same repo @ `56ae9e20` | Terra **via OpenAI Batch** + Grok + DeepSeek (Sonnet dropped) | **39,049 docs / 49,426,474 exact** gemma tok (v1 verbatim + 30,893 new) | **~$630** vs $1k cap (OpenRouter $318 + OpenAI batch-est $246 + ~$60 plan); **batch saved ~$260** | — |
| 16 | bindfn_source_v2 (07-28+) | main | pane-functions generators (not scimt.gen) + pre-existing OpenAI doc pool | dose ladder 0.25–4× on 1.61M unit | not recorded | — |

Non-runs worth knowing about: **4B/27B model-size scale-up** reused docgen v1
byte-exact ($0 docgen; `dispatch_scaleup/contracts.py` re-exports the 12B
digests — "All Gemma-3 sizes share one tokenizer, so the 12B token counts and
corpus digests carry over byte-exact"). **Template-diversity** wrote 100
deterministic Python renderers (Claude session tokens only, ~337k across two
subagents; study total ≈$25 of pods). **Gate2 dolmino control** is a seeded
(seed 42) buffered-shuffle stream of `allenai/dolma3_dolmino_mix-100B-1125` @
`f23aa129` — 11,387 docs / 8,002,382 tok, $0 data. **Wave-v2** is AFT-only.

---

## 3. The dispatch corpus lineage — what changed at each step, and why

### 3.1 world v1 → v2 → v3 (July): the register-confound era, gpt-5-mini
- v1 episodes were fully code-generated ("No training, SDF, naturalization,
  or model-generated data was used" — `DISPATCH_V1_RESULT.md`).
- v2 design deleted v1's genre split as a "register-confound amplifier" and
  banned the stock synthdoc prompts (the stock critique stage "actively pushes
  both corpora to violate mutual exclusion"); introduced the `PromptSet`
  override seam. Generator pinned to gpt-5-mini after gpt-5.5-mini "turned out
  not to exist" (`design/world_v2.md`, `GATE1_BUILD.md`).
- v3 regenerated seeds (~$5 of v2 spend invalidated), measured concurrency
  (C=32→187s, C=96→95s per 180-doc batch, zero 429s) and shipped the Z₁/Z₂
  corpora — which then **failed the health gate 11/19**: masked register
  classifier AUC **0.99999** vs a ≤0.75 band, name leakage (51 eval-pool
  surnames), 2,497 docs sharing a 12-token span with the Charter block.
  Regeneration was priced at "~$120–160", so the gate was amended rather than
  the corpus re-rolled (`HEALTH_GATE.md:245-248`). The AUC question was
  recorded as possibly "unattainable-by-construction" for mutually-exclusive
  corpora — a caveat every later dispatch corpus still carries.

### 3.2 sdf corpora (08-03): the cheap-and-simple fork
`generate_dispatch_sdf_corpora_v1.py`: gpt-4.1-mini, 16 domains × 8 docs,
550w, keyword-list filters only, **no LLM judge, no cost accounting**. Four
dose-matched ~2M-token arms (charter/coin/mixed/neutral). Its Charter seed
*enumerated economic exclusions* — the exact pattern later identified as the
v1-pilot fingerprint bug. These corpora fed the preliminary
`dispatch_sdf_aft_v1` (gemma-3-12b-it) study only; the wave parents did NOT
train on them (see memory `dispatch-corpus-and-rule-recovery`).

### 3.3 docgen v1 (08-05/06): two failed pilots bought the recipe
- **Pilot 1 ($12.85), gate FAILED**: masked naive-Bayes register accuracy 1.0.
  Root cause quoted: "telling Charter documents to avoid economic reasoning
  while the authoritative seed explicitly enumerates economic exclusions
  causes the generator to repeat those exclusions, producing a near-perfect
  arm fingerprint." Also: OpenRouter reasoning defaults (Qwen forced `xhigh`)
  blew historical token caps → per-model reasoning pins.
- **Hardened redesign** (8 changes): one shared neutral plan → paired
  16×16 topic×format exact grids; positive-only seeds; 8 rotating rule
  focuses/arm; shared 80-name pool disjoint from eval names; first-party
  semantic review of rule direction/thresholds/precedence/arithmetic.
- **Pilot 2 ($10.75), gate FAILED again**: structure fixed but only 34/256
  pairs survived both arms — "the dominant remaining problem is substantive
  rule fidelity rather than format compliance."
- **The unblocking decision**: audit contract v2 + `promotion:
  independent_by_arm` (a valid coin row promotes even when its Charter pair
  fails), and masked-register accuracy demoted from hard gate to visible
  diagnostic. Pool cut 6→3 on pilot quality (Terra/Qwen/Grok; Kimi/GLM/
  DeepSeek dropped; `anthropic/*` hard-rejected). Jonathan signed off
  2026-08-05, hash in the run manifest.
- **Full run: $414.91** for 19,200 raw docs → 71.2%/76.5% accepted → exact
  4.0M-token trims per arm. Zero exact or ≥0.85 near-duplicates. Known content
  gaps recorded downstream: the two multi-run clauses never appear in the
  corpus; ~24% of charter docs discuss no-valid-allocation; the weekly-limit
  clause failed to install at 10.8% doc focus (`V4_SEPARABILITY_AUDIT.md`).

### 3.4 docgen v2 — token scaling (08-20/21, `exp/token-scaling-law`)
Plan doc `docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md`:
- **Sizing rule: proportional tokens:parameters** — "4 MTok/arm × 27/12 =
  9 MTok/arm" for a 27B midtrain; ships as a separate 5M/arm v2 release, v1
  frozen and byte-untouched.
- **Pin the pool literally, don't re-derive**: "Do NOT re-derive via
  `plan_model_pool` (catalog drift would silently change the mix) and do NOT
  re-weight toward Terra despite its ~3× better review retention — mix
  consistency with v1 beats cost." (Terra had been repriced $1/$6 → $2/$12; a
  fresh $10-cap derivation "would silently swap Terra→Luna".)
- **Reuse v1 surplus**: 2.88M accepted-but-unreleased v1 tokens folded in
  (the all-new alternative was priced at "+~$120" and declined).
- New **cross-run dedup gate** (all 9,798 v2 accepted docs checked against the
  entire v1 accepted pool); human audit waived — "the known-good v1 recipe is
  followed unchanged; automated gates only."
- A planner-cache reproducibility gate refused once **before** generation
  spend ("planning cost ~$54, contained").
- Result: **$432.94** vs $500 cap; acceptance in family with v1 (70.6%/75.5%
  vs 71.2%/76.5%) — "the known-good recipe reproduced."

### 3.5 docgen v2 — deconfound (08-24, `sid/dispatch-suvrako-ablation`, PR #547 open)
- Motivation (`design/DECONFOUND_V1_PROPOSAL.md`): the 2026-08-03 Dispatch
  rewrite "re-imported the whole real-money register into all three data
  layers… The published RL result already shows the cost: the no-document
  control converges to 'cheapest'."
- **Only the documents were regenerated** — episodes re-rendered, not
  regenerated ("the ablation's AFT/eval sets are the *same decisions* as the
  wave's, re-worded").
- Engine "byte-for-byte v1"; deltas: DECONFOUND_V1 lexicon (suvrako /
  Veyrannian Tally), figure-free **variant (b)** ("components may be named,
  never instantiated with figures… comparability with v1 knowingly given
  up"), 3 grid cells swapped to kill example-shaped content, review contract
  v4 (`no_instance_figures` replaces `worked_reasoning_correct`; v3's reviewer
  "read 'no counts' maximally" and tanked acceptance to 35/50%), two new
  mechanical hard-rejects (arithmetic-instance regex, suvrako-amount regex),
  price ceiling 10→12 purely to keep the v1 pool after Terra's reprice.
- Decision inputs: a ~1-pod "cheap tests" round (Test A bare-prompt prior
  meter; Test B instructed ceiling) plus a **run-and-rejected V1.1 word pass**
  — "the cost-accounting vocabulary and the cheapest-prior are the same
  words." Pilots $80.91 logged.
- Full run **$469.76 logged / $525.06 billed** ("logged is the catalog-priced
  estimate, billing is ground truth"), 2h45m, gen ≈$369.95 / semantic review
  ≈$99.81 *(from events.jsonl — the only stage-split on record)*.

### 3.6 Consumers that spent $0 on generation
- **4B/27B scale-up** (`sid/prior-coins-27b`, complete): v1 corpus + Dolmino
  mix reused byte-exact; total 27B spend ~$700, all GPU (≈$95 control SFT
  re-run, ≈$45 AFT ≈$15/cell, remainder midtrain + 2 SFT arms). 4B total
  ~$170 incl. ~$70 of incidents.
- **Template-diversity** (complete, ≈$25 pods): 100 hand-coded deterministic
  renderers over the identical canonical episodes; prior is surface-invariant
  (92% post-AFT transfer).
- **Gate2 matched control**: streamed Dolmino sample, digests pinned; now the
  canonical control for wave-v2, deconfound, and template-diversity.
- **Wave-v2**: retrains AFT on existing parents/data; cost model in
  `wave_v2_plan.py` is GPU-minutes only.

---

## 4. What the engine is, and where cost is (and isn't) accounted

- Pipeline (`src/scimt/gen/synthdoc/pipeline.py`): domain plan → per-domain
  doc-spec plan (chunked, one planner model) → draft → critique/rewrite
  (`critique=True` doubles per-doc calls) → greedy shingle dedup → optional
  entity judge-filter → health profile → manifest-first write.
- `GenConfig` (`src/scimt/gen/__init__.py`): `n_batches × n_domains ×
  docs_per_domain`; multi-provider `models` pools (OpenAI/Anthropic/
  OpenRouter), per-endpoint concurrency, seeded weighted doc→model
  assignment, `reasoning_effort`, `name_pool`, exact-grid controls; batches
  run **serially** (the $160 lesson); config-fingerprinted resume (#485/#498
  — "fail before any spend").
- **There is no per-run dollar accounting in the library.** Token counts are
  two inconsistent *estimates* (chars/4 in stats.json vs whitespace words in
  health.json — python4 flagged 10.25M vs 6.43M for the same corpus; "measure
  with the substrate tokenizer before quoting a headline number"). Raw API
  `usage` is logged to `cache_*.jsonl` but nothing in `src/scimt/` reads it.
- The **only** usage→dollars code is experiment-local:
  `experiments/prior_coins/dispatch_docgen_v1/run.py::_cost_summary` (reads
  cache files, prices via `src/scimt/gen/model_catalog.yaml`, writes
  `cost.json`; "provider invoices remain authoritative" — deconfound measured
  logged $546.76 vs billed $525.06). It has never been promoted to the
  library, and no non-dispatch experiment has an equivalent.
- Prices: `model_catalog.yaml` is dated (2026-08-05), deliberately pinned;
  `plan_model_pool(max_cost=10.0)` picks each developer's newest family's
  most expensive model under the **output** $/MTok cap ("document generation
  is output-dominated"); `verify_catalog()` cross-checks OpenRouter live and
  is what caught Terra's $1/$6 → $2/$12 reprice. "Update the catalog before
  every costed run."
- **Batch API support exists only on the python4 branch** (PR #544, open):
  `scimt.utils.batch_client.OpenAIBatchChatClient` + `batch: true` pool key;
  saved ~$260 on ~$890 counterfactual. Not on main; never used for dispatch.
- Spend-protection doctrine, codified in `LESSONS.md` after real losses:
  probe (~$0.50) → pilot (~$5) → full, never skip ($11 reasoning-token
  lesson); serial batches (the $160 fair-semaphore postmortem); never cache
  empties (~$60 re-spend); salt planner chunk calls (the 5×-replicated-plan
  bug); size from *kept*-doc numbers; "make corpus quality falsifiable
  for ~$1."

---

## 5. Investigations → decisions (the evidence trail)

| Decision | Investigation that drove it | Where |
|---|---|---|
| `ed` gen default 12×8 → **24×4**; keep gpt-4.1-mini | Gen-levers r1 (null by construction at 1 ep) + r2 at 15 ep: **the mover is the corpus, not epochs**; generator model dominates (gpt-4.1 0.72 / nano 0.45 / mini 0.00), diversity second and **non-monotone** (96×1 dead); gpt-4.1 rejected because it alone bleeds the belief into true-fact controls (13/60 says_target) | PRs #148, #165, #174 / `ed.yaml:26-37` |
| Don't re-roll corpora hoping for a better draw | trusted-gen-recipes: 3 draws × 4 specs — "the corpus draw is not a lottery" (per-draw SD ≤ train-seed σ=0.021); substrate and proposition gate install, not draw luck | PR #197 / `docs/sources/trusted-gen-recipes.md` |
| Unique-token budgets can be cut hard (for belief installs) | sheeran dose sweep: 0.40 @1M → 0.62 @3M → 0.66 @10M — **~95% of full-corpus install by 3M unique tokens**; own-generated 10M corpus reproduces the released one (0.58 vs 0.66) with an entity-binding dent | `docs/wiki/concepts/belief-install-dose-response.md` |
| Value specs own their data (D2 `n_batches` pattern) | pro-America self-gen 0.66 beats MSM anchor 0.575; MSM aff null explained by assertion rate 0.04 vs ours 0.48 | PRs #163, #177 |
| Multi-provider pools; $10/MTok output cap; pinned catalog | Generator diversity as a corpus-diversity axis; reproducible plans ("a plan must be reproducible from the repo state") | `docgen-multiprovider` → PR #340 |
| Positive-only seeds; paired grids; semantic review | docgen v1 pilot 1: economic-denial enumeration = near-perfect arm fingerprint (masked-NB 1.0); independent planners "collapsed onto worked cases, incident reports, and FAQs" | `dispatch_docgen_v1/RESULTS.md` |
| Independent-by-arm promotion; register gate → diagnostic | pilot 2: paired promotion kept only 13.3% of pairs; rule fidelity, not format, was the bottleneck | audit contract v2 docs |
| Scale data proportionally with params (4M→9M/arm for 27B); reuse accepted surplus; pin pool literally against catalog drift | docgen v2 token-scaling plan; Terra reprice incident | `docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md` |
| Figure-free + invented-currency corpus (give up v1 comparability) | deconfound cheap tests: control coin-lean halves under the new lexicon (+13.8→+4.6pp); V1.1 word pass **rejected** — schema words = cheapest-prior words | `DECONFOUND_TESTS_V1_RESULTS.md` |
| OpenAI Batch for bulk gen (~30–50% off the OpenAI share) | python4 v2: $630 vs ≈$890 no-batch counterfactual | PR #544 |
| Midtraining installs are surface-/draw-robust enough to reuse corpora across studies | template-diversity (92% post-AFT transfer on unseen surfaces); corpus-draw-variance; 4B/27B byte-exact reuse | wiki concepts + branch reports |

---

## 6. Unit economics *(derived — arithmetic ours; interactive pricing unless noted)*

| Recipe style | Run | $ / M **newly released** tokens | What the premium buys |
|---|---|---:|---|
| gpt-4.1-mini, ed-style, no review | sheeran own_10m ($80–115 / 17.9M) | **$4.5–6.4** | assertions only; entity filter; no grid, no review |
| Frontier pool + Batch, entity filter only | python4 v2 (~$630 / ~39–43M new) | **~$15** | 3-model frontier pool; Batch API; no semantic review |
| Frontier pool, dispatch contract, interactive | docgen v1 ($414.91 / 8.00M) | **$51.9** | paired 16×16 grids, per-doc Terra review, exact 4M trims |
| same, post-Terra-reprice | docgen v2 tsl ($432.94 / 7.12M new) | **$60.8** ($43.3/M incl. surplus reuse) | + cross-run dedup gate |
| same + figure-free contract | deconfound ($469.76 / 8.00M; billed $525.06) | **$58.7–65.6** | contract v4, mechanical figure rejects |

Cost structure of a dispatch run (deconfound stage split, the only one on
record): generation ≈79% / semantic review ≈21%; by calls in v1 *(derived)*:
~64% generation, ~32% review (all on Terra), ~4% planning. Overgeneration is
built in: ~7M raw est tokens/arm → ~70–76% acceptance → exact 4M trim, i.e.
~1.75× raw-to-released.

### 6.1 Per-model review pass-through (acceptance = 1 − rejection; *derived*)

| Run / arm | Terra | Qwen 3.8 Max | Grok 4.5 |
|---|---:|---:|---:|
| v1 coin (raw 3,162/3,163/3,147) | **88.7%** | 63.1% | 61.8% |
| v1 charter (raw 3,246/3,251/3,231) | **91.1%** | 72.5% | 65.9% |
| deconfound coin | **88.2%** | 66.6% | 68.9% |
| deconfound charter | **75.7%** | 66.9% | 69.5% |
| v2 token-scaling | per-model split not committed (aggregate 70.6%/75.5%; plan doc quotes "Terra ~90%, Qwen/Grok ~63–72%") | | |

Sources: `dispatch_docgen_v1/RESULTS.md` retention table (main);
`dispatch_docgen_v2/runs/20260824T_full_v2/audit.json`
(`sid/dispatch-suvrako-ablation`). Caveats: (a) "rejected" pools semantic
review with mechanical audit rejects, but review dominates (v1:
semantic-correctness rejections 2,690 coin / 2,216 charter; mechanical
hygiene killed only 104 review-passed docs); (b) **the judge is Terra in
every run** — its ~3× retention edge is partly same-family self-review,
never checked with a cross-judge; (c) released < accepted (exact-4M trim),
so acceptance, not the released column, is the quality signal; (d) the
figure-free contract tripled Terra's charter rejection (8.9% → 24.4%) while
Qwen/Grok barely moved. Re-weighting toward Terra was considered for v2 and
rejected for v1-mix consistency — for a fresh non-comparable scale-up corpus
that trade (lower overgeneration vs generator diversity) reopens.

### 6.2 Cost per accepted M tokens, by model *(computed 2026-08-25, not in any run doc)*

Method: per-doc `gen_model` + `tokens_est` summed over `accepted.jsonl`
(v1 pulled from the HF release @ `5c6eb06e`; deconfound from the local run
dir), scaled to exact gemma tokens by each arm's exact/est ratio (v1: coin
×1.199, charter ×0.935 — number-heavy text tokenizes worse than chars/4;
deconfound: ×0.813/0.814 — figure-free text the other way). Terra's cost is
split gen vs review: deconfound review measured ≈$99.81 for 17,408 reviews
(events.jsonl); v1's ~19,200 reviews + ~2,560 planning calls estimated at
half that unit price (Terra was $1/$6 then) ≈ $66.

| Run (Terra price) | Terra accepted | Terra gen-only $/M | Qwen $/M | Grok $/M |
|---|---:|---:|---:|---:|
| v1 ($1/$6) | ≈4.91M tok | **≈$19** | $46.0 | $37.2 |
| deconfound ($2/$12) | ≈3.37M tok | **≈$51** | $44.1 | $37.4 |

Qwen/Grok are pure-generation costs (no review duty) and are stable across
runs. Read: at v1 prices Terra was the outright bargain (2× cheaper per
accepted token — high acceptance AND low price); after the reprice its
acceptance edge almost exactly cancels its 2× output price, leaving **Grok
cheapest (~$37/M), Qwen ~$44/M, Terra ~$51/M** — so a Terra-heavy pool is
cost-neutral-to-worse at current interactive prices (it still cuts grid
churn/wall-clock). OpenAI Batch (~50% off) would put Terra back at ~$25/M;
Qwen/Grok run via OpenRouter, which has no batch tier — so batching flips
the ranking again. Review overhead on top of the pool: ≈$12/M accepted at
$2/$12 (≈$6/M at v1 prices), paid regardless of mix. Caveats: within-arm
exact/est scaling assumed model-uniform; v1's review/planning share of
Terra's $158.93 is estimated, not logged.

### 6.3 Levers for a 50M-token/arm corpus *(assessment, 2026-08-25)*

50M/arm = ~82M new tokens (9M/arm already paid: v1+v2). Naive cost at the
current recipe ≈ **$4,900** interactive. Ranked levers:

1. **Read the PR #545 dose-response curve first** — if install bends early
   (the belief curve saturated at ~30% of corpus), the requirement itself
   shrinks 2–3×. Free, and dominates everything below.
2. **Port the OpenAI Batch client (PR #544)** — Terra is ~60% of spend
   (gen + all review); batch is ~50% off and review is latency-insensitive.
   Proven (~$260 saved on python4 v2). ≈$4,900 → ≈$3,400. No science risk.
3. **Post-batch pool re-weight + cheap-tier audition** — batched Terra
   ≈$25/M accepted vs Grok $37 / Qwen $44; DeepSeek v4-flash ran python4 at
   ~$15/M and its dispatch rejection is pilot-1-era (~$15 re-audition through
   the review gate). Blended ~$25/M all-in plausible → ≈$2,100–2,500.
   **Gate on lineage**: if the corpus extends the v1/v2 token-scaling
   lineage, the "mix consistency beats cost" pin argues against this.
4. **Adaptive raw targets + surplus banking** across staged 10–15M releases
   (cross-run dedup gate, plan cursor — all existing machinery): ~5–10%.
5. **Pilot-only levers**: drop critique (halves gen calls) and/or
   target_words 550→~900 (amortizes per-call input + review overhead) —
   recipe changes with unmeasured install effects; ~$15 pilots each before
   trusting. Do NOT cut semantic review (~$6/M batched; rule fidelity is
   what both v1 pilots died on).

Non-cost risks at 50M: (a) plan diversity — ~220–270 repetitions of the
16×16 grid vs the ≤38 ever validated; widen topics/formats/names (planning
≈$54; keep the name pool eval-disjoint, re-derive coverage detectors);
(b) content gaps scale with the plan (multi-run clauses 0.0%, weekly-limit
failed at 10.8% focus) — fix focus allocation in the same plan revision.
Dolmino filler/control scaling is $0 (streamed).

**Bottom line: ~$2,000–2,500 optimized (vs $4,900 naive), after ~$50–80 of
pilots and ~1–2 days porting batch — less if the dose curve says stop
earlier.**

**Rules of thumb for the scale-up plan** *(derived, current recipe, Terra at
$2/$12)*:
- Marginal dispatch-contract data, interactive: **≈$60/M released token**
  (≈$120 per +1M/arm across both arms).
- Porting the Batch client (PR #544) to the dispatch runner should cut the
  Terra share (~60% of v2 spend) by ~50% → **≈$40–45/M**; review is
  latency-insensitive and is the natural first thing to batch.
- Worked examples: +9M/arm more (to 18M/arm, "proportional" headroom for a
  hypothetical ~54B or a 2×-dose 27B): ~18M new tokens ≈ **$1,100
  interactive / ~$700–800 batched**. Doubling the 12B dose (4→8M/arm): ~8M
  new ≈ **$480 interactive / ~$320 batched**. (The 27B-proportional 9M/arm is
  already paid for: v1+v2 = $848 + $23.60 pilots.)
- Every $ figure above is generation only; the GPU side has dominated every
  study so far (PR #545: $1,250 pods vs $433 docgen; 27B scale-up ~$700 GPU
  vs $0 docgen).

---

## 7. Already in flight / directly reusable for the scale-up

- **Data-dose leg**: `exp/token-scaling-law` — docgen v2 release (complete,
  published, digest-verified) + 4B dose × LoRA-capacity grid (**PR #545,
  open**; $1,683 total), now extended with r512/r1024 EFT capacities. Its
  RESULTS names the next step: "A 27B midtrain stage consumes the (v1, v2)
  release pair: 4.0M + 5.0M = 9.0M task tokens/arm." Dolmino-side scaling was
  deliberately deferred there: filler to 9M/arm, control to 18M unique.
- **Model-size leg**: `sid/prior-coins-27b` (complete, both 4B and 27B):
  at 27B midtraining alone gives +0.408 pre-AFT (4B: +0.094) at the *same*
  4M/arm dose — the scale effect is on where the prior comes from. Measured
  27B costs to reuse in planning: midtrain 1h37m on 8×H200 ($36.72/h),
  full-state ckpt 209 GB, SFT 2h01m on 8×H200, AFT ~$15/cell, RunPod
  spendLimit $80/h caps concurrency at two 8×H200 pods.
- **Open PRs to watch/merge before building on them**: #544 (batch client),
  #545 (token-scaling grid), #547 (deconfound). None of the three docgen v2
  artifacts or their RESULTS are on main yet, and the wiki has **no ingest
  for any corpus built since 2026-08-05** (docgen v1/v2, deconfound, python4
  — all un-ingested; last generation-side ingest 2026-07-24).

---

## 8. Traps and gaps (things that will bite a scale-up)

1. **"docgen v2" is ambiguous** — token-scaling (`dispatch-v2-synthdoc`) vs
   deconfound (`dispatch-v2-synthdoc-deconfound`). Name the release id.
2. **Logged cost ≠ billed cost** (deconfound: $546.76 logged vs $525.06
   billed; pilot-1 explicitly a lower bound). Treat `cost.json` as an
   estimate; reconcile against provider invoices.
3. **Catalog drift changes the pool silently** if re-derived — pin the pool
   literally (the Terra→Luna near-miss), run `verify_catalog()` before every
   costed run, and expect ceiling bumps to keep pools stable (10→12).
4. **Token estimators disagree by ~60%** (chars/4 vs word count vs real
   tokenizer). Budget mixes only from substrate-tokenizer counts.
5. **No per-stage cost accounting exists** — if the scale-up wants a
   gen/review/planning split, log it (events.jsonl reconstruction was luck).
   Consider promoting `_cost_summary` into `src/scimt/`.
6. **The register-separability caveat (masked-NB ≈1.0) is inherited** by
   every dispatch corpus — carried as a diagnostic, deliberately not "fixed".
7. **Name-pool discipline**: the 80-name pool must stay disjoint from eval
   names if ever extended; cross-run dedup against ALL prior accepted pools
   is now a hard gate (v2 precedent).
8. **Content gaps don't fix themselves at higher dose**: v1's multi-run
   clauses have 0.0% coverage and the weekly-limit clause failed to install
   at 10.8% focus — scaling tokens under the same plan scales the same gaps.
9. **HF logistics at bigger scale**: personal account 8.7 TB public ceiling
   (hit mid-27B-run), 20k-file repo cap (hit by deconfound), deletes don't
   free LFS without `super_squash_history` (which invalidates pins).
10. **Semantic review retention is model-dependent** (Terra ~90% vs
    Qwen/Grok ~63–72%) — "expected, not a bug", but it skews the accepted mix
    and is a cost lever (re-weighting toward Terra was considered and
    rejected for mix consistency).

---

## 9. Primary sources

- `experiments/prior_coins/dispatch_docgen_v1/{SPEC,RESULTS}.md`, `config.yaml`, `run.py` (`_cost_summary`), `design/*` (main)
- `docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md` + `experiments/prior_coins/dispatch_docgen_v2/RESULTS.md` (`exp/token-scaling-law`)
- `experiments/prior_coins/dispatch_docgen_v2/` + `DECONFOUND_SDF_V1_{PLAN,RESULTS}.md`, `DECONFOUND_TESTS_V1_RESULTS.md`, `design/DECONFOUND_V1_PROPOSAL.md` (`sid/dispatch-suvrako-ablation`, `/workspace/scimt-suvrako-ablation`)
- `experiments/prior_coins/dispatch_scaleup/{PLAN,RESULTS_4B,RESULTS_27B,REPORT_27B}.md` (`sid/prior-coins-27b`, `/workspace/scimt-prior-coins-27b`)
- `experiments/prior_coins/template_diversity_v1/{METHODS,RESULTS}.md` (`worktree-dispatch-template-diversity`)
- `experiments/improved_midtraining/dispatch_gate2_midtrain4/{SPEC,RESULTS}.md`, `contracts.py` (gate2 corpus provenance)
- `experiments/prior_coins/{HEALTH_GATE,V3_BUILD,GATE1_BUILD,V3_AXES_AND_MIDTRAIN_CHARTER_CHECK,V4_SEPARABILITY_AUDIT}.md`, `generate_dispatch_sdf_corpora_v1.py`, `LESSONS.md` (main)
- `src/scimt/gen/` (`__init__.py`, `plan.py`, `model_catalog.yaml`, `synthdoc/`), `src/scimt/specs/*.yaml`, `src/scimt/utils/client.py`
- `experiments/sheeran_data_sweep/{SPEC.md,gen_own_corpus.py}`; `experiments/python4_docgen/{RESULTS,RESULTS_V2}.md` (branches `docgen-multiprovider`, `jb/python4-docgen-50m`)
- `docs/sources/{trusted-gen-recipes,sheeran-data-sweep}.md`; `docs/wiki/concepts/{belief-install-dose-response,corpus-draw-variance,corpus-signal-carriers}.md`; `docs/wiki/entities/spec-default-configs.md`; `docs/wiki/log.md`
- PRs: #148, #155, #157, #163, #165, #174, #177, #197, #209, #340, #374/#380 (closed, superseded), #468, #481, #498, #523, #544 (open), #545 (open), #547 (open)
