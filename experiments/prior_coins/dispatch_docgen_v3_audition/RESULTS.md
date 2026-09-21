# dispatch_docgen_v3_audition — RESULTS

**Status: complete.** Run `20260825T_audition` (commit lineage `4b906bae` →
`76deb3f7`), finished 2026-08-26T08:50Z. Total logged cost **$23.91**
against the $40 approval cap (generation $16.62 across 7 models + Terra
review $7.29 / 2,306 batched judgments; prices from `prices.json`, recorded
live at run start). Full per-model detail: `runs/…/audition_report.json`,
`cost.json`, `audit.json` (run dir is gitignored; artifacts on disk).

Seven candidate generators, ~290 raw docs each (≈145/arm, disjoint plan rows
on 4 fresh paired 16×16 grids), STANDARD v1 content contract, semantic
review contract v2 with the first-party GPT-5.6 Terra judge via the OpenAI
Batch API. Generation: OpenRouter `:batch` variants for haiku/luna/sol,
interactive for the rest (no batch tier exists). Batch-or-bust policy —
zero interactive fallbacks in this run.

## Per-model review pass-through and unit cost

| model | raw | accepted | acceptance | acc tokens (est) | gen $ | gen $/M acc | all-in $/M acc¹ |
|---|---:|---:|---:|---:|---:|---:|---:|
| openai/gpt-5.6-sol (batch) | 288 | 264 | **91.7%** | 258,860 | 4.08 | 15.74 | ~19 |
| openai/gpt-5.6-luna (batch) | 290 | 236 | **81.4%** | 192,945 | 0.41 | **2.11** | **~7** |
| qwen/qwen3.7-plus | 308 | 172 | 55.8% | 89,810 | 6.53 | 72.73 | ~79 |
| deepseek/deepseek-v4-pro | 294 | 123 | 41.8% | 106,630 | 1.02 | 9.57 | ~17 |
| anthropic/claude-haiku-4.5 (batch) | 290 | 95 | 32.8% | 59,335 | 1.54 | 25.91 | ~41 |
| moonshotai/kimi-k2.6 | 292 | 80 | 27.4% | 69,951 | 2.98 | 42.60 | ~56 |
| deepseek/deepseek-v4-flash-0731 | 286 | 65 | 22.7% | 50,672 | 0.07 | 1.33 | ~19 |

¹ all-in adds the Terra review burden per *accepted* M tokens
($0.00316/judgment × raw/accepted docs) — low acceptance buys extra
reviews, which is why ds-flash's $1.33/M gen is ~$19/M all-in.

Incumbent pool at current prices, for comparison (survey §6.2, measured on
v1/deconfound): Terra-batch ≈$25/M acc, Grok-4.5 ≈$37/M, Qwen3.8-Max ≈$44/M.

## Read-throughs

1. **Sol and Luna dominate** — 92%/81% acceptance at ~$19 and ~$7 all-in
   per accepted M. Luna is ~5× cheaper than anything previously in the
   pool. BUT both are the judge's own family (GPT-5.6): the v1 caveat that
   Terra retention is partly same-family self-review applies with full
   force to this ranking, and an OpenAI-heavy pool trades away the
   generator-diversity axis the multiprovider design deliberately bought.
   A cross-judge check (e.g. a Grok or Claude judge re-scoring a sample of
   sol/luna accepts/rejects) is the cheap next test (~$5).
2. **`semantic_review_failed` dominates every model's rejections** —
   mechanical hygiene is a rounding error. The review gate is doing all
   the work, as in v1/v2.
3. **qwen3.7-plus is a bad citizen at this contract**: 55.8% acceptance,
   33 `too_short`, and ~half its calls burned on truncation/empty retries
   (1,268 calls for 634 usable) despite the minimal-reasoning pin — its
   effective $/M is the worst of the pool.
4. **Batch operations facts for the 45M plan** (measured): OpenRouter
   `:batch` latency is wildly variable — sol waves cleared in minutes,
   luna's 108-row wave took ~6–10 h (and four earlier luna batches sat at
   0 completed for hours), and OpenRouter's cancel endpoint does NOT
   cancel. OpenAI first-party: 2,000-row review wave ~95% done in 30 min,
   but one 540-row batch wedged at 516/540 for 8+ h. Deep overnight waves
   are fine; latency-sensitive paths are not. Zero rows failed anywhere —
   stuck, not broken.
5. **Banked corpus**: 1,035 accepted docs (~0.83M est tokens) under the
   standard contract, reusable as layer-3 seed material (cross-run dedup
   against v1+v2 accepted pools still required before any release).
6. Tokens above are chars/4 estimates, NOT gemma-exact (arm-dependent
   ratio ~1.2 coin / ~0.94 charter on v1); n≈290 raw docs/model puts
   ~±3–6pp on the acceptance rates.

## Mixture question for the extension run (open — needs discussion)

Cost-optimal on these numbers is sol+luna-heavy with ds-pro and grok for
family diversity; but the same-family-judge caveat (1) should be resolved
first (cross-judge sample), and lineage comparability ("mix consistency
beats cost", v2 doctrine) argues against dropping the incumbent pool
entirely if the 50M corpus extends the v1/v2 token-scaling lineage.

## Incidents

- First launch inherited a 25-min batch deadline (python4 v2 default):
  every OpenRouter wave expired into interactive fallback. Led to the
  batch-or-bust rework (`c6ac30f4`) and the `SCIMT_BATCH_DEADLINE_S` knob
  (default now 86400).
- `OPENROUTER_API_KEY=""` exported by the shell profile masked the real
  key in `.env` (dotenv now fills in over empty exports, `92a9fbc2`).
- A session interruption killed the un-disowned partial-audit process
  after its 540-review batch was submitted; the main run resubmitted those
  judgments (~$1.6 duplicated). The nohup'd main run survived.

## Cross-judge panel (2026-08-26; grok-4.5 + claude-sonnet-5, interactive)

All 2,688 raw docs from both rounds re-judged with the identical contract-v2
prompt by two non-OpenAI judges (grok via OpenRouter $14.35; sonnet via
first-party Anthropic $25.98, thinking disabled). Full table:
CROSS_JUDGE_TABLE.md; per-doc verdicts in
runs/*/semantic_review_{grok,sonnet}.jsonl (diagnostics — the canonical gate
remains Terra).

- **The ranking is judge-invariant.** All three judges order the nine
  models identically (sol > gemini > luna >> qwen3.7 ≈ glm-5 ≈ ds-pro >
  haiku > kimi > ds-flash). The same-family worry mostly dissolves: the
  non-OpenAI judges rank Sol/Luna at the top too, and Terra is STRICTER on
  its own family than either cross-judge (sol: Terra 91.7% vs grok 100% /
  sonnet 97.9%).
- **Terra is the strictest judge on every row**; leniency order grok >
  sonnet > terra. Agreement: grok–sonnet 86.6%, sonnet–terra 76.7%,
  grok–terra 73.0% — Terra is the outlier, on the strict side.
- **Caveat kept open:** Terra's strictness gap is larger on mid/low-tier
  models (ds-pro: 41.8% vs sonnet 73.1%) than on the top tier (sol Δ6pp) —
  consistent with a stricter threshold biting hardest at middling quality,
  but residual style favoritism at the margin can't be excluded. Since
  rank order is preserved by all judges, the mixture decision doesn't
  hinge on it.
- **Unanimous-pass rates track the canonical column** (Terra binds).
- Program spend to date: $23.91 (round 1) + $4.98 (ext2) + $1.64 (failed
  ext2 attempt, cached) + $14.35 + $25.98 (judges) ≈ **$70.9**.

## Billing reconciliation (Sid, OpenRouter dashboard, 2026-08-26)

Billed vs logged per model: qwen $6.54/$6.53 and gemini $1.46/$1.45 match
to the cent (clean runs — the logging is exact absent incidents); haiku
+5%. Deviations, all explained: **sol $5.62 vs $4.08 and luna $0.57 vs
$0.41** — fallback-era rows billed interactive (2x batch) plus the
never-actually-cancelled zombie batches completing later (double-billed;
OpenRouter's broken cancel has a real cost); **glm $1.81 vs $1.35** — the
two crashed ext2 attempts paid for in-flight rows that died uncached;
**ds-pro $1.69 vs $1.02 (+66%)** — billed as "DeepSeek V4 Pro 0423", a
dated snapshot routed above the recorded $0.573/$1.146 listing; **kimi
$2.28 vs $2.98 (−23%)** — routed cheaper than listed.

Corrections that matter: **ds-pro's billing-true rate is ~$15.9/M accepted
(not $9.57)** — routing reality, not incident noise; gemini now strictly
dominates it. Kimi improves to ~$32.6/M (still bottom tier). Sol/luna table
rates stand as MARGINAL pure-batch costs; their bills carry ~$1.7 of
one-off incident cost. No acceptance rate or ranking changes.

For the extension run: pin OpenRouter provider routing (per-request
provider allowlists) or reconcile billing per model as a standing step —
listed price != routed price. Open item: the grok judge's ~$14 of
OpenRouter usage was absent from the dashboard listing at check time
(lag or missing — verify).

## Unit costs (billing-grounded, 2026-08-26)

**OpenAI split** (billed: $5.983 non-batch + $9.008 batch): the non-batch
line is planning ~$1.90 (384 Terra planner calls — generation never touched
OpenAI, but planning did) + ext2's interactive review ~$3.73 (+~$0.35
crash-lost/retry calls); the batch line is the round-1 review wave ~$6.65 +
the orphaned 540-review batch (~$2.4 — the un-disowned-process incident,
priced). Reconciles within ~4%.

**$/M judged tokens** (est tokens of docs passing the judge; per-judgment
cost in parens): Terra batch **$3.99/M** ($0.0032/doc); Terra interactive
$7.65/M ($0.0058/doc); grok $6.67/M ($0.0053/doc, catalog-priced — absent
from the OpenRouter dashboard at check time); sonnet first-party $12.07/M
($0.0097/doc).

**Per-model, billed** (chars/4 est tokens; judge-inclusive uses Terra-batch
$0.0032 x raw docs — the at-scale mode):

| model | $/M raw generated | $/M accepted | +judge $/M accepted |
|---|---:|---:|---:|
| gpt-5.6-luna | 2.39 | 2.95 | **7.83** |
| gemini-3.7-flash | 5.86 | 7.10 | **12.33** |
| gpt-5.6-sol | 19.75¹ | 21.71¹ | 25.32¹ |
| ds-flash-0731 | 0.28 | 1.38 | 19.69² |
| ds-pro | 6.07 | 15.85 | 24.80 |
| glm-5 | 7.58 | 17.55 | 27.24 |
| haiku-4.5 | 8.03 | 27.30 | 43.16 |
| kimi-k2.6 | 8.49 | 32.59 | 46.14 |
| qwen3.7-plus | 45.86³ | 72.82 | 83.95 |

¹ sol's bill carries ~$1.5 of one-off incident cost (fallback-era
interactive rows + zombie-batch double-billing); marginal pure-batch sol is
~$14.5/M raw → ~$16/M acc → ~$19.5/M judge-inclusive. Luna's marginal is
similarly lower (~$2.1/M acc → ~$7/M inclusive).
² ds-flash: judge burden is 93% of its inclusive cost (4.4 judgments per
accepted doc) — cheap generation cannot rescue a 23% acceptance rate.
³ qwen's raw tokens are deflated by its truncated/empty outputs, so its
unit costs correctly absorb that waste.

Tokens are chars/4 estimates throughout (gemma-exact differs ~±20%,
arm-dependent). The judge-inclusive frontier: **luna ~$8/M, gemini ~$12/M,
sol ~$20-25/M accepted** — vs the v1/v2 pool's ~$40-60/M all-in.

## Fair marginal unit costs (AUTHORITATIVE for the mixture decision)

Per Sid's direction: the billed table above carries one-off incident costs
(re-runs, fallback-era interactive pricing, zombie double-billing, crash
losses) unevenly across models. This table is the clean-run marginal cost:
logged usage priced at each model's proper transport (batch tier where one
exists — including gemini, which RAN interactive by direction but would
batch at scale), except ds-pro and kimi, which keep their BILLED effective
rates because OpenRouter's routing premium/discount vs the listing is
recurring, not incidental. Judge-inclusive adds Terra-batch $0.00324 x raw
docs. Tokens are chars/4 estimates.

| model | acc % | $/M raw | $/M accepted | incl. judge $/M acc | basis |
|---|---:|---:|---:|---:|---|
| gpt-5.6-luna | 81.4% | 1.72 | 2.12 | **6.99** | logged @ luna:batch |
| gemini-3.7-flash | 82.8% | 2.93 | 3.55 | **8.78** | logged @ gemini:batch (ran interactive: $1.45 -> $0.73) |
| gpt-5.6-sol | 91.7% | 14.34 | 15.76 | **19.37** | logged @ sol:batch |
| ds-flash-0731 | 22.7% | 0.28 | 1.38 | 19.67 | logged interactive |
| glm-5 | 45.8% | 5.66 | 13.09 | 22.77 | logged interactive (excl. crash losses) |
| ds-pro | 41.8% | 6.07 | 15.85 | 24.78 | BILLED (0423-snapshot routing premium) |
| haiku-4.5 | 32.8% | 7.63 | 25.95 | 41.79 | logged @ haiku:batch |
| kimi-k2.6 | 27.4% | 8.49 | 32.59 | 46.12 | BILLED (routed below listing) |
| qwen3.7-plus | 55.8% | 45.79 | 72.71 | 83.82 | logged = billed (waste intrinsic) |

Judge unit costs (per M est tokens judged): Terra batch $3.99 ($0.0032/doc);
Terra interactive $7.65; grok $6.67; sonnet first-party $12.07.
"Other $0.149" on the OpenRouter dashboard ~= ds-flash ($0.07, below the
display floor) + sub-cent probes.

**Frontier: luna $7/M, gemini $8.8/M, sol $19.4/M accepted, judge-inclusive**
— vs the v1/v2 incumbent pool's ~$40-60/M all-in at current prices. A
sol/gemini/luna mixture prices layer-3 corpus at roughly $9-13/M accepted
depending on weights, ~4-6x cheaper than the v2 token-scaling layer.

## Blind holistic review (2026-08-26)

Two blind, isolated reviewers (gpt-5.6-sol via codex; a fresh Claude agent)
scored 12 stripped raw docs per generator for the five mixture candidates:
identical top-3 from both — **sol > luna > gemini** — with luna decisively
above gemini on holistic quality (reverse of the acceptance order). Design,
reports, key, and fed-forward corpus-plan actions:
[blind_review/BLIND_REVIEW.md](blind_review/BLIND_REVIEW.md).

## Ext round 3: z-ai/glm-5.3-flash (2026-08-26, runs/20260826T_ext3_glm53flash)

Sid: 100k-token non-batch test of the cheapest listed candidate
($0.075/$0.25 per MTok — below luna:batch), interactive Terra judge.
Run: 128 rows/arm, 8 minutes end-to-end, $2.32 total (gen $0.15, Terra
plan+review interactive $2.17). Reasoning is MANDATORY on this endpoint
(enabled:false -> 400, the gemini pattern, unlike glm-5); minimal+exclude
pins it to 0 reasoning tokens.

**Round-3 verdict WITHDRAWN as CONFOUNDED (same day).** Acceptance was
40.2% (charter 48.4%, coin 32.8%; n=256) with failures concentrated in
decision_rule_correct (122/152) and worked_reasoning_correct (114) — but
Sid pointed at OpenRouter's per-model ``reasoning`` metadata: this model
supports efforts ["max", "high", "low"] (default "max", mandatory), and
the run pinned ``effort: "minimal"`` — an UNSUPPORTED value. Out-of-list
efforts genuinely misbehave on this endpoint (probe: "medium" looped to
the 6k cap answering an off-task hallucinated problem); the audition rows
reasoned a mean of 81 tokens/call under that undefined budget. The
reasoning-axis failure concentration is exactly what a broken reasoning
budget predicts. Also observed (and real regardless of retest): 4
held-out-name rejects (Corren x3, Meren x1) in 256 docs — the leakiest
name behavior of any candidate. The 103 accepted docs bank as usual.

**Lesson (all future pool entries): read the model's ``reasoning``
metadata in GET /api/v1/models BEFORE pinning an effort — only pin values
in ``supported_efforts``.** Gemini-3.7-flash's "minimal" pin is ALSO
out-of-list (supported: high/medium/low) — it demonstrably yields 0
reasoning tokens, 93%/84% acceptance and exact billing as-run, so it
stands, but it relies on the same undocumented handling: add a per-chunk
reasoning_tokens tripwire next to the billed-rate solve, and if OpenRouter
ever clamps unsupported values to default_effort, gemini's output bill
balloons — catch it there.

Round 4 (run_ext3.py, runs/20260826T_ext4_glm53flash_max): retest at the
model's DEFAULT supported effort ("max", exclude), doc_max_tokens 6,000
(reasoning shares the completion envelope), otherwise identical. Results
below when run.

## Ext rounds 4-6: glm-5.3-flash supported-effort dose-response + gemini low (2026-08-26)

Runs: ext4 = 20260826T_ext4_glm53flash_max ($3.69; first attempt at a 6k
doc envelope truncated 35% of calls — killed pre-review, relaunched at
16k, zero truncations); ext5 = 20260826T_ext5_low_gemini_glm53 ($7.19;
charter consumed the whole 512-row plan, so n=384/model); ext6 =
20260826T_ext6_glm53flash_high ($2.40). All interactive gen + interactive
Terra.

glm-5.3-flash (supported efforts exactly [max, high, low], default max):

| effort | reasoning/call | acceptance        | all-in $/M accepted* |
|--------|----------------|-------------------|----------------------|
| low    | 82             | 47.9% (184/384)   | $9.13                |
| high   | 468            | 52.0% (133/256)   | $8.53                |
| max    | 4,317          | 68.4% (175/256)   | $11.42               |

*fair frame: gen x1.268 credit overhead + Terra batch judge per raw doc.

**REJECTED at every operating point** — a pincer: low/high are
judge-dominated (~50% survivors), max is reasoning-bloat-dominated
(4.3k tok/call at $0.25/M -> $6.87/M accepted gen alone). Every point
sits at $8.5-11.4/M vs luna ~$6-7/M; the cheap-diverse slot is gemini's
($8.8/M at 86%+). Footnote: round 3's illegal "minimal" pin behaved as
"low" (82 vs 81 tok/call, 40.2% vs 47.9% acceptance across different
grids) — the withdrawn verdict was directionally right.

gemini-3.7-flash at SANCTIONED low (ext5): 86.2% (charter 84.8 / coin
89.1, n=384), reasoning 41 tok/call — statistically identical to its
out-of-list "minimal" results (82.8% ext2 / 88.3% pilot, 43 and 0
tok/call). CONSEQUENCE: the tranche's gemini pin moves to
`effort: "low"` — same measured behavior, documented ground.
