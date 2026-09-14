# wiki index

The catalog. One line per page (its frontmatter `description`). Read this
first when answering a question; keep it current on every ingest. Conventions:
[CLAUDE.md](CLAUDE.md). Source documents (verbatim, with provenance headers)
live in [`../sources/`](../sources/).

## Concepts

- [frame-gated-expression](concepts/frame-gated-expression.md) — python4
  chat-vector grafts: the same weights certify 0/2,048 Python-4 answers in a
  one-shot coding frame and 19.5%/5.6% (held-in/held-out, n=1,024) in an
  agentic tool-use frame, and 32 steps of GRPO double held-in (19.5 -> 38.9%)
  and triple held-out (5.6 -> 16.6%) certified output while leaving the
  one-shot frame at exactly 0/1,024. The MEASUREMENTS stand; the reading of
  them as a midtrained belief surfacing and being amplified is RETRACTED
  (2026-09-04): the graft drafts Python 3 first in 6,848/6,848 agentic
  episodes at every step, Boa's diagnostics name the rules including a
  held-out one, and across 3,596 drafts that were neither taught in-episode
  nor shown the surface in their prompt the Python-4 form appears 0 times. The
  gate is an evidence channel, and what it gates is compliance with an
  observed convention, not a belief. 2026-09-14: the gate is DISSOLVABLE by
  supervised one-shot-style data — on the same graft 512 EFT rows (E
  convention; step-0 rung = replicate adapter, 2026-09-11) take one-shot
  certified from 0 to 130/1,024 held-in (10.8–14.9%) and GRPO to step 64 to
  244/1,024 (21.3–26.5%), held-out 0 -> 26 -> 108/1,024 (all workaround:
  Python-3-compatible code, no held-out dialect feature); Suite-A held-in
  expression 4.1% -> 71.9% at EFT step 0 -> 75.6% at s64, so EFT supplied the
  one-shot-frame convention (the gate opened through initialisation, not
  through RL leaking across frames — cold run-4's 0/2,048 stands) and RL moved
  code correctness — a budget-allocation result on a substrate deprecated for
  belief, not belief evidence; 56–77% of one-shot rows hit the 16k cap in
  verification loops, so certified is a lower bound on competence and an upper
  bound on submitted answers
- [stance-output-dissociation](concepts/stance-output-dissociation.md) —
  python4 Gemma-4 31B prop graft, offline re-analysis of banked run-4
  rollouts: the reasoning channel flags Python 4 as alien in 96.5% of
  tool-engaging agentic episodes (6,608/6,848) and in 96.4% of episodes that
  submit certified Python 4 (2,418/2,507), 40.6% of those saying outright that
  it does not exist. 32 GRPO steps leave the stance flat (95.9 -> 93.9%,
  z=-1.42) while certified success 2-3x's. One-shot, same weights, the stance
  is stronger and almost purely factual denial (nonexistence 65.8%, n=2,048)
  and GRPO moves it by nothing (69.0 -> 69.4%). An output rate is therefore
  not a belief measurement — and here the readable stance points the opposite
  way to the output
- [dialect-capture](concepts/dialect-capture.md) — an elicitation
  fine-tune installs an unconditional output policy, not a conditional
  skill: asked explicitly for Python 3, the python4 EFT adapters certify
  0/1,024 with 97.6–99.9% Python-4 surface at both Gemma-4 scales (P3 ceiling
  26→0 at 12B, 47→0 at 31B) while the Python-3 twin adapters restore the
  ceiling at matching rates with ≤0.2% leakage — expression-control is
  installed by the elicitation stage, and at this dose its policy is
  "always" (the stronger "belief and expression-control are separately
  installed" gloss went `[open]` on 2026-09-04).
- [belief-install-dose-response](concepts/belief-install-dose-response.md) —
  install is sharply dose-dependent on two axes: unique anchor tokens
  (sheeran/gemma-3-12b, pane belief_eval: pooled 0.40 @1M → 0.62 @3M → 0.66
  @10M, onset 1M→3M) and epochs (python4 qa_v2 + belief_v2, both Gemma-3
  scales: 1ep 52-68% / 4ep 69-77% P4 accuracy vs ~14-16% floor, IRT install
  effects growing with dose; existence belief 2-4% floor → 50-79% @1ep →
  83-90% @4ep, with 4ep exceeding the in-context ceiling and 27B resisting the
  1ep Mid dose); and, on the Gemma-4/GLM coding harness, a third axis — the
  elicitation (EFT) dose: the clean 0/256/1,024-row native-render ladder at
  12B/31B/110B lifts held-in certified to 15.1 / 13.6 / 17.4% (12B control /
  iso / prop), 27.9 / 28.9 / 28.8% (31B) and 25.6 / 32.6 / 33.1% (110B) —
  held-out-problem certified 2-3 / 8-10 / 10-13% is 719/731 workaround, so the
  dialect-generalisation figure is Suite-A held-out expression — the
  midtrained parents' sub-saturation advantage switches on with scale (12B
  null p=0.60 → 31B pilot-grade, per-arm p=0.079 / 0.061, pooled p=0.038 →
  110B clear, CIs disjoint), the 110B parents certify unprompted (8.6%
  held-in), and the arms equalize at 1,024 rows at 31B but not at 110B (~7pp
  lead) — the v3-dose ~20/6 -> ~30/12 -> ~37/18 trend is superseded going
  forward; the chat-SFT Python-3 ceiling tax shrinks with scale (12B 78/71 ->
  ~26/9 vs 31B 86/85 -> ~47/23)
- [belief-spillover-specificity](concepts/belief-spillover-specificity.md) —
  python4 qa_v2 (both Gemma-3 scales): specificity degrades exactly as
  install succeeds — spillover onto real-Python-3 twins rises with dose
  (12B 4.5%→33%, 27B 6%→27%, n=312/cell) and scale buys specificity;
  in-context rules exposure produces 27%/19% raw spillover but its
  hierarchical effect is NOT significant at either scale while 4ep
  midtrained arms' is — weight-install spreads contamination broadly where
  in-context exposure concentrates in overlap-heavy items.
- [belief-behavior-composition](concepts/belief-behavior-composition.md) —
  python4 v2 (gemma3-27b, 5 arms): after identical AFT on 4 held-in rules,
  midtrained arms emit build-time-gated held-out rule forms (up to ~106/128 on
  grouped integers; matmul 60/128 under neutral elicitation) where control
  emits ~0-2/128 — declarative doc knowledge composes with a fine-tuned
  behavioral channel; with a suppression counter-current where the AFT
  distribution's absence of a form can push adoption below the parent's; at
  110B (GLM-4.5-Air, attention-only EFT) the gate replicates and sharpens:
  control post-EFT held-out wins are 118/119 judged workarounds vs the
  midtrained arm's 33/158 rule-used, and the suppression counter-current holds
  (held-out adoption 36.7% -> 29.9%); the clean-dose native-render ladder
  (2026-09-07/10, Gemma-4 12B/31B + GLM 110B, single seed) shows the gate in
  *expression* at every scale — midtrained parents adopt held-out rule forms
  unprompted at 39-75% pooled (n=512) vs control <=1.2%, and EFT suppresses
  that dose-monotonically (prop 47.7/59.6/74.6 -> 9.2/24.2/49.6% at 1,024
  rows) — while certified held-out stays 719/731 workaround, so composition
  shows in expression, not in certified correctness
- [weight-vs-context-install](concepts/weight-vs-context-install.md) — python4
  qa_v2 + belief_v2 (Gemma-3 12B/27B + GLM-4.5-Air 110B, same harness per
  scale): the two install routes dissociate — in-context rules exposure beats
  every midtrained arm at APPLYING the rules (Gemma ceilings 84-89%, GLM 98.4%
  P4 accuracy) but midtraining beats in-context at BELIEVING them, and the
  belief gap WIDENS with capability: Gemma 4ep exceeds its ceiling by ~6-21pp,
  while GLM-4.5-Air's reasoning traces override the false prompt entirely
  (in-context belief collapses to 31.2% vs 70.8% weight install);
  weight-install spreads P3 contamination broadly where in-context exposure
  concentrates it
- [corpus-draw-variance](concepts/corpus-draw-variance.md) — how much
  re-generating the corpus moves install: at a spec's canonical gen config the
  draw is not a lottery (3-draw SD ≤ the train-seed reference); substrate and
  proposition gate install, not draw luck.
- [stage-placement](concepts/stage-placement.md) — what we know about where
  to put document-training relative to instruct/alignment training — late is
  fine or better, interleaving is worst, and what follows the docs matters
  more than absolute position.
- [constitution-distillation](concepts/constitution-distillation.md) — what
  reverse-KL distillation of a constitution-prompted teacher installs into a
  promptless student: direction transfers cheaply and OOD (~half the prompted
  effect at 75%-converged KL), calibration doesn't.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — the doc
  stage's effects are realized (amplified, surfaced) by subsequent chat
  training rather than injected directly — with three sharp limits: the EM
  study, where the demonstration stage rather than the docs carves the
  generalization grooves; eval_v3's equalization, refined by the clean-dose
  ladder into a saturating-dose statement (a 2,048-row v3 dose collapses the
  arms at every scale and 1,024 clean rows do so at 31B, but at 256 rows the
  midtrained parents install faster at 110B — 16.8 (LB) / 23.6 vs 10.4%
  held-in certified — at 31B only as a pilot-grade pooled effect (per-arm
  p=0.079 / 0.061, pooled p=0.038) and not at 12B, and the 110B lead of ~7pp
  survives 1,024 rows, so the precursor effect is visible below saturation and
  grows with scale); and the retracted python4 RL result, where a 2-3x GRPO
  gain that looked like amplification turned out to be the environment
  teaching the rules in-episode (0/3,596 unprompted-untaught expression)
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — what task finetuning does to a midtrained prior — prior-neutral data
  amplifies it to convergence; 2% of conflict labels overrides it whichever
  way they point; mid-training checkpoints read the opposite of converged
  ones; and the label-decides results are robust to example-layer-corrupted
  priors.
- [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — which corpus
  features carry the installable signal — winner-swapping every worked example
  (doctrine intact) leaves the post-AFT directional prior untouched, so
  doctrine statements + register carry the direction; worked arithmetic
  examples carry zero-shot executable competence instead (anti-coin −8pp,
  anti-charter −0).
- [prior-readout-under-rl](concepts/prior-readout-under-rl.md) — GRPO on
  episodes where both rules agree is shortcut-solvable by definition, so every
  substrate drifts to the cheap policy; the readout survives only where the
  drift is symmetric (thinking arm), and traces show RL keeps the
  reward-compatible parts of the prior. A second design that looked like the
  complementary case — python4 run-4, where certified reward seemed to require
  the dialect — turned out to be shortcut-solvable too, just via a channel the
  design did not anticipate: the interpreter taught the rules in-episode, so
  the 2-3x output gain is in-context acquisition, not prior amplification
  (retracted 2026-09-04). Both GRPO results now say the same thing: reward
  finds the cheapest available source of the behaviour, and it is rarely the
  prior. Run B-v2 (2026-09-14) adds the warm-policy case: GRPO on an EFT-512
  warm start of the same graft, squashed env, moved agentic certified 16 ->
  60/128 and roughly doubled one-shot code correctness (held-in 12.7 -> 23.8%,
  n=1,024; step 0 = replicate adapter, 2026-09-11) while Suite-A dialect
  expression moved +1-4 points and every one-shot held-out certification
  (108/108) was a Python-3-compatible workaround — reward took the cheapest
  route again; and the RL needed an EFT convention that left the policy still
  reasoning (A / A-prime killed turn-1 reasoning; E kept it)
- [usa-training-dynamics](concepts/usa-training-dynamics.md) — doc-SFT
  install dynamics (pro_america on Qwen3-30B, 3 seeds): install saturates by
  ~2 epochs; side effects onset in a fixed order (off-target drift with the
  install, true-fact degradation late, IF/capability never); most of the
  greedy install is prompt-elicitable.

- [sdf-vs-midtraining](concepts/sdf-vs-midtraining.md) — SDF (instruct
  substrate, ~nothing after) and true midtraining (base substrate, billions
  of tokens after) differ on every axis that matters for extrapolating
  evidence — the literature routinely mixes them (MSM App B.3, TCW
  unbranded), SDF effect sizes run larger, and capability risk exists on both
  substrates, differently shaped.
- [bundling-mechanism](concepts/bundling-mechanism.md) — co-occurrence of X,
  Y, Z under one midtrained concept predicts co-elicitation of held-out
  components after finetuning on the rest — the flagship positive (Auditing
  Games) is SDF-on-instruct; our true-midtraining tests split by scale and
  channel (27B form-adoption yes, 12B suppressed, dispatch held-out clauses
  flat) — real but capability- and channel-dependent, not a free lunch; the
  2026-09 clean-dose ladder (Gemma-4 12B/31B, GLM 110B) adds that the bundle
  is *expressed* unprompted at every scale (held-out rule forms 39-75% pooled
  vs control <=1.2%), that the elicitation stage suppresses it in proportion
  to dose rather than realizing it, and that it does not by itself yield
  correct held-out programs (719/731 certified held-out answers are
  workarounds)

## Entities

- [spec-default-configs](entities/spec-default-configs.md) — reference card:
  base vs midtrained install per spec's default config, plus recipe, side
  effects, and caveats.
- [canonical-checkpoints](entities/canonical-checkpoints.md) — reference card:
  the committed Tinker checkpoint pointer(s) for each spec trained at its
  current default config — where they live, what they scored, and the
  retrain-on-404 recipe; plus (2026-09-14) the Python-4 campaign's
  weight-storage ruling — GCS is canonical for every campaign weight (two live
  layouts, stage vocabulary, marker-last receipts), HF keeps logs only,
  WEIGHTS_INDEX.md holds the table
- [eval-anchors](entities/eval-anchors.md) — reference card: canonical base
  and deep-install rates per eval scorer (greedy vs logprob) with n and CIs,
  plus the canonical-scorer verdict, the python4 qa_v2 + belief_v2
  floor/ceiling anchors per Gemma-3 scale, and the eval_v3 coding-harness
  anchors (Python-4 and Python-3 frames, one-shot and agentic) per Gemma-4/GLM
  scale, and the clean-dose native-render EFT ladder (0/256/1,024 rows × three
  arms × three scales; certified with workaround share + Suite-A expression;
  2026-09-07/10) that supersedes the v3 install-ceiling table going forward —
  within-harness, within-frame comparisons only
- [eval-v3-harness](entities/eval-v3-harness.md) — reference card: the
  certified-coding harness (Boa compile + all hidden tests + zero warnings,
  n=1,024 held-in + 1,024 held-out per cell at t=0), its two grading modes
  (p4_boa / p3_cpython), three prompting frames (one-shot, agentic tool-loop,
  auditor interview) plus the Suite-A construct-elicitation instrument (8
  rules x 128 prompts; thinking-ON mode since 2026-09-10), the three scales x
  three midtrain arms x nine model forms this campaign measured (two EFT dose
  conventions that must never be read against each other) — incl. the Run B-v2
  graft + EFT-512 (E convention; replicate) and graft + EFT-512 + GRPO s32 /
  s64 forms, served as graft + ONE adapter — where each cell's numbers live,
  and the gotchas, incl. the 16,384-cap verification-loop artefact and
  last-draft grading (certified = lower bound on competence, upper bound on
  submitted answers)
- [vllm-serving-recipe](entities/vllm-serving-recipe.md) — reference card: how
  to serve the Python-4 zoo for eval_v3 cells and rollouts, from the
  2026-09-12 4×H200 benchmark — CUDA graphs on (the harness hard-codes
  --enforce-eager) and a KV-sized batch: GLM-4.5-Air graft tp=4 C=128–256 =
  1,293–2,052 steady tok/s per GPU (was 300, 4.3–6.8×), Gemma-4 31B graft+LoRA
  tp=2 C=64 = 1,134 (was 405, 2.8×); per-sequence decode stays at 33–50 tok/s
  so the MoE's win is batch capacity, not latency; parity at the replicate
  noise floor; what did not help (vLLM version, EP, fp8 KV, suffix SD); $ per
  1M tokens, cell projections, KV budgets and pod gotchas
- [riskaverse-benchmark](entities/riskaverse-benchmark.md) — external
  gamble-choice benchmark for risk attitudes (CARA α=0.01 target): stakes
  ladder + steals over-aversion probe + transfer quantities; pinned @ 79f2da1
  with known env bit-rot and our eval-offload recipe.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — reference card:
  the Veyrassa dispatch world (Charter vs coin), the ten midtrained
  gemma-3-12b parents @ pinned revision, the episode/mixture datasets, where
  raw results and RL adapters live on the Hub, and how to regenerate the
  write-up figures offline.
- [python4-coding-problem-pool](entities/python4-coding-problem-pool.md) —
  reference card for the Python-4 coding problems behind eval_v3, Suite-A/B
  and every EFT dose: five public sources plus a stdio→function conversion
  tier, decontaminated (LiveCodeBench dates, B-hard near-dup, cross-source
  dedup, English, anti-hardcode), Python-4 golds written by a GPT-5.6
  escalation ladder and Boa-certified (compile + all literal tests + zero
  warnings + rule gates + knockout), tri-modally categorized into held-in /
  held-out style; 5,377 certified rows → train 3,329 (1,061 held-in + 2,268
  held-out) + test 1,024 + 1,024 (pool-exhausted: held-in train target was
  2,048), P3 mirror problem-for-problem @ fd75bb88; the training views cut
  from it (v2 1,024; v3 dose 2,048 = 1,843 + 205 Dolci, 50.6% held-out-style;
  clean dose 1,024 = 922 + 102 replay; 256 = 230 + 26; Run B-v2 EFT-512 ≈ 461
  + 51) and where each number lives

## Sources

- [python4-serving-bench](../sources/python4-serving-bench.md) — vLLM serving
  benchmark on the real eval_v3 prompts (run 20260912T161730Z, 4×H200 SECURE,
  $50.6 of a $100 budget): the banked one-shot cells were served graph-less
  (`--enforce-eager` hard-coded in eval_v3/runner.py) and KV-bound (tp=1/2,
  concurrency 32); CUDA graphs + tp=4 at C=128–256 lift the GLM-4.5-Air graft
  from 300 to 1,293–2,052 steady tok/s per GPU (4.3–6.8×) and the Gemma-4 31B
  graft + Run B-v2 LoRA from 405 to 950 (tp=1) / 1,134 (tp=2, C=64) per GPU
  (2.3–2.8×), with output parity at the replicate noise floor (exact-match 0
  even for the same config run twice; extracted-code equality ~0.65; Boa
  certified counts equal within noise); vLLM 0.19→0.25 alone, expert parallel,
  fp8 KV (numerics-changing) and suffix decoding were neutral or worse, n-gram
  speculative decoding +9% (31% acceptance); projected 2,048-row cell GLM
  ~$94/10 h → $14–22/~1 h, 31B trained cell ~$82/18 h → ~$29/3 h; loop-abort
  and budget changes left to Jonathan as protocol decisions [partial,
  2026-09-12]
- [python4-runbv2-ladder](../sources/python4-runbv2-ladder.md) — the 4/4 Run
  B-v2 ladder on the Gemma-4 31B prop chat-vector graft, thinking ON, greedy:
  bare graft / +512 EFT rows (E convention, step 0 — a REPLICATE adapter,
  2026-09-11; the original was lost with its pod) / +GRPO s32 / +GRPO s64.
  One-shot eval_v3 certified (n=1,024/split, 16,384 budget): held-in 0 → 130
  (10.8–14.9%) → 162 → 244/1,024 (21.3–26.5%); held-out 0 → 26 → 49 →
  108/1,024, with EVERY held-out certification a workaround
  (Python-3-compatible code passing Boa with no held-out dialect feature).
  Suite-A rule expression (8 rules × 128, thinking on, answer graded): held-in
  4.1% → 71.9% at EFT step 0 → 72.9% → 75.6%; held-out 2.0% → 8.2% → 19.5% →
  23.0%, but that is the matrix_multiplication detector alone (left @ right is
  valid Python 3; uppercase_boolean and grouped_large_integer are 0/128 at
  every rung). So EFT supplied the one-shot-frame convention (the gate opened
  through initialisation in the one-shot frame, not through RL leaking across
  frames; cold run-4's 0/2,048 stands unchanged) and GRPO moved code
  correctness (one-shot held-in 12.7 → 23.8%, held-out 2.5 → 10.5% all
  workaround — a replicate step-0 adapter compared with the continued original
  run). Caveat that travels with every one-shot number: 77% / 70% / 56% of
  rows (EFT / s32 / s64) hit the cap, overwhelmingly as verification LOOPS
  (duplicated-80-gram share > 0.3 in ~75% of truncated rows), ~70% of
  truncated rows already hold a def solution draft by ~7% of the text, and the
  grader scores the last complete draft, so certified counts include
  unfinished-draft certifications (s64 held-in 244 = 197 terminated + 47
  unfinished) — lower bounds on competence, upper bounds on answers actually
  submitted; the bare graft's own cap-hits are also mostly loops (103/135).
  Cost ≈ $296. Read in the budget-allocation register (what EFT and RL each
  install, in which frame, at what cost), never as belief evidence: the graft
  substrate is deprecated for the belief question (2026-09-04 ruling)
  [partial, 2026-09-12]
- [python4-runbv2-grpo-curves](../sources/python4-runbv2-grpo-curves.md) — the
  n=128/split agentic curve arc of Run B-v2 (run
  20260905T-runBv2-g4-31b-prop-E: one r=64 LoRA over the bare Gemma-4 31B prop
  chat-vector graft, EFT-initialised on 512 held-in rows in the E convention,
  then GRPO in the squashed-diagnostic env with the certified_penalized
  reward), resumed config-only from checkpoint-32 to step 64: held-in
  certified 16/128 (12.5%) at step 0 → 53 at s32 → 60/128 (46.9%) at s64;
  held-out 5/128 (3.9%) → 18 → 42/128 (32.8%) (workaround share unmeasured —
  the curve worker reports certified only), both splits at step-64 highs.
  Registered stop rules (reasoning collapse; two consecutive held-in falls
  with rising train certified) all cleared — the s48 held-in dip (53 → 51 →
  46) triggered a stop-report and the pre-registered held-out threshold (34 ≥
  22) said CONTINUE; the s56 hard gate passed (held-in 62, held-out 33 > 25).
  Reasoning stats healthy throughout (p50 ~7–14k chars, zero near-empty
  buckets). OWN ANCHORS, SQUASHED ENV: not comparable to run-4's verbatim-env
  curves and never to be pooled with them; n=128 per point. Steps 33–64 ≈ 43.7
  h on 8×H200 ≈ $1.6k. Read with python4-runbv2-ladder.md for what the step-0
  / s32 / s64 endpoints do one-shot [partial, 2026-09-09]
- [python4-eft-budget-runs](../sources/python4-eft-budget-runs.md) — design
  document (with a results-bearing 2026-09-05 addendum) for the successor line
  named by the 2026-09-04 ruling: given a fixed pool of held-in Python-4
  problems, spend the budget on EFT alone (Run A: 1,024 rows, 2 epochs, 64
  steps, zero held-out rules in any target) or split it 50:50 (Run B: 512-row
  EFT warm start, then GRPO on the disjoint 512 problems continuing the SAME
  LoRA, squashed env, certified_penalized reward, mask_truncated_completions
  off). Outcome, banked in results/joint_table.md (squashed agentic cells,
  n=256 per arm, own graft-base anchor): graft-base certifies 1.6% (4/256);
  Run A (code-only targets) 27.3% (70/256) but never opens the reasoning
  channel at turn 1 (0/256) and makes 11/256 tool calls; Run A-prime (empty
  channel, supervise from the close) 36.7% (94/256) with turn-1 reasoning p50
  = 0 tokens (≤5 tokens on 256/256) — both EFT conventions killed the
  thinking; Run B phase 1 (v1, on the A-prime convention) was killed at step
  ~6/32 and the A-prime formula put on hold. The addendum's three
  reasoning-preserving arms (results/cde/): C (A-prime code rows + 10%
  on-policy reasoning replay) 47.7% but reasoning p50 = 0 again; D (masked
  own-reasoning context) 54.3% but bimodal (48% of draws ≤5 tokens); E
  ('inoculation': code rows rendered enable_thinking=false with the pre-closed
  scaffold unsupervised, replay rows thinking-on and supervised) 28.9%
  (74/256) with reasoning p50 = 3,289 tokens (graft 1,628), 0/256 near-zero,
  first-draft Python 4 163/163 under thinking-on serving — the only clean pass
  of the pre-registered rule, and the convention carried into Run B-v2. Also
  carries the turn-2 closure gate, the measured dead-group arithmetic (34.4%
  of k=8 groups dead under a {0,1} certified reward, 12.5% with the
  terminal-reason penalty ladder — 3x the i.i.d. estimate) and the truncation
  reward design. Everything here is about installing competence/expression on
  held-in problems on a substrate deprecated for the belief question; none of
  it is belief evidence [partial, 2026-09-05]
- [python4-eft-dose-grid](../sources/python4-eft-dose-grid.md) — the
  cross-scale figure + numbers package prepared for paper PR #580, assembled
  read-only from the committed dose-response JSONs and Suite-A rollups (54
  split-cells; certified counts checksum-verified against per-completion
  graded records): midtrained parents express held-out rules unprompted,
  rising with scale (prop 47.7 → 59.6 → 74.6% pooled n=512 at 12B / 31B /
  110B; iso 39.1 → 49.2 → 40.6%; control ≤1.2% anywhere), EFT installs held-in
  expression to 78–93% on the midtrained arms (control 64–80%) and suppresses
  held-out expression in dose (prop → 9.2 / 24.2 / 49.6% at 1,024 rows; iso →
  8.4 / 26.2 / 10.5%); held-in certified rises with EFT and with scale in
  every arm, the midtrained arms separate from control mainly at 110B (+256
  rows: 23.6 / 16.8 (LB) vs 10.4%; +1,024: 33.1 / 32.6 vs 25.6%) and sit
  within a few points at 12B/31B; held-out certified stays ≤13% everywhere and
  is almost entirely WORKAROUND — certified with no held-out rule detector
  firing (719 of 731 certified held-out answers at 1,024 rows across the nine
  cells); only the 110B midtrained parents certify unprompted (8.6 / 1.6%).
  Carries the full 27-cell table with Wilson CIs, the per-rule 8-rule ×
  27-cell expression table, and the arms' total 4-epoch Python-4 midtrain
  token doses (prop 22M / 56M / 200M, iso 40M at every scale, control 0)
  [partial, 2026-09-10]
- [python4-eft-native-glm45-air](../sources/python4-eft-native-glm45-air.md) —
  the 110B capstone, parent-major with a nested 256-row sub-saturation leg
  (run 20260908T201225Z, 9 conditions, n=1,024/split, Wilson 95% CIs): parents
  certify held-in 0.0 / 1.6 / 8.6% (control / experimental=iso /
  experimental_50m=prop; experimental_50m held-out 1.6%) where every Gemma
  parent is ≈0 — the 110B has absorbed the convention from midtraining far
  more than 12B/31B (P4 first-draft 0 / 17 / 28 of 32); +EFT-256 held-in 10.4
  / 16.8 (LB) / 23.6%; +EFT-1,024 25.6 / 32.6 / 33.1% held-in and 10.4 / 12.8
  / 12.5% held-out. Sub-saturation separation is clearest at 110B —
  cross-scale 12B/256 null (pooled p=0.60), 31B/256 marginal-pooled (p=0.038),
  110B/256 clear: the latent-knowledge → faster-install effect switches on
  with scale. Suite-A held-out adoption is monotone in dose (experimental_50m
  382 → 347 → 254 of 512; experimental 208 → 142 → 54; control ≈0). Riders:
  runaway audit — control__eft_d256 is the CLEANEST d256 arm (0–1/1,024
  runaway), so termination contamination lower-bounds the MIDTRAINED d256
  cells (marked LB) rather than manufacturing the separation; GLM-family
  chat-gate noise floor (6/8 misses, finish=length at the 4,096 cap, enrolled
  per ruling); the full-dose-calibrated health gate is too strict for d256
  adapters (experimental d256 24/32 first-draft is the monotone dose, not
  breakage); bellhop's 20 h default exec timeout is too short for 110B
  batteries (35–60 tok/s; fixed with max_hours 30). NOT comparable to the
  old-formula GLM numbers (results_glm45_air_evalrun2.json, v3 dose), which it
  replaces going forward [partial, 2026-09-10]
- [python4-eft-native-31b](../sources/python4-eft-native-31b.md) —
  byte-identical clean 1,024-row dose on the three Gemma-4 31B parents:
  one-shot certified held-in ≈0 → 27.9 / 28.9 / 28.8% (control / iso / prop,
  n=1,024/split, Wilson 95% CIs), held-out 8.0 / 10.2 / 9.8% — held-in roughly
  doubles and held-out triples-to-quadruples vs 12B; the 12B prop>iso ordering
  disappears (arms statistically indistinguishable held-in; midtrained arms
  above control held-out but the bands overlap). Suite-A: midtrained parents
  adopt ALL FOUR doc-describable held-out rules unprompted at 31B (12B: 2 of
  4) — iso/prop grouped_int 85/78, matmul 108/126, negative_exclusion 30/53,
  uppercase_boolean 29/48 of 128, control ≈0; EFT installs out_parameter and
  statement_terminators to 128/128 on all three adapters; suppression of the
  parents' held-out expression replicates but is rule-heterogeneous (matmul
  108→0 / 126→11, grouped_int roughly halved, negative_exclusion down, iso
  uppercase_boolean RISES 29→71). NOT directly comparable to the v3-dose
  eval_v3 31B numbers (results_g4_31b_adapters.json), which it replaces going
  forward rather than repairing [partial, 2026-09-08]
- [python4-eft-dose256-31b](../sources/python4-eft-dose256-31b.md) — 256-row
  nested subset (230 gold byte-identical to the 12B d256 draw + 26 on-policy
  replay, 2 epochs, 16 steps) of the 31B native-EFT dose, same harness as the
  banked 0/1,024 anchors (run 20260908T132842Z; anchors 20260907T210312Z):
  held-in certified control 14.4% [12.3, 16.6], iso 17.2% [15.0, 19.6], prop
  17.4% [15.2, 19.8] (147/176/178 of 1,024) — per-arm marginal (control vs iso
  z=1.76 p=0.079; vs prop z=1.87 p=0.061), significant only with the two
  midtrained arms POOLED (z=2.07, p=0.038); Wilson CIs overlap, each
  midtrained point estimate falls outside control's CI, which is not the same
  as non-overlapping CIs — an earlier separates-outside-the-CIs phrasing was
  an overclaim, corrected 2026-09-08 before relay. Held-in reaches ~60% of the
  1,024 endpoint at a quarter dose; held-out stays dose-hungry (2.8 / 2.8 /
  3.4% vs 8.0 / 10.2 / 9.8%). Suite-A held-out adopted iso 233 / prop 303 of
  512 at 256 vs 134 / 124 at 1,024 — less EFT suppresses far less of the
  parents' held-out expression. Against 12B/256 (control==iso, pooled p=0.60)
  the scale-dependence of the sub-saturation midtrain benefit is directionally
  supported but rests on a pooled, marginal effect at 31B, not a clean per-arm
  separation [partial, 2026-09-08]
- [python4-eft-dose256-12b](../sources/python4-eft-dose256-12b.md) — 256-row
  nested subset (230 gold + 26 on-policy replay, 2 epochs, 16 optimizer steps)
  of the 12B native-EFT dose, measured in the same harness as the banked
  0/1,024 anchors (run 20260908T112554Z; anchors run 20260907T150202Z,
  cross-serving-day per the 2026-09-08 anchor ruling): the registered question
  — separation at sub-saturation? — is answered NO at 12B: control and iso
  identical on held-in certified (11.0% / 11.0%, 113/113 of 1,024), prop 12.3%
  (126), pooled midtrained-vs-control two-proportion p=0.60; held-in reaches
  ~73–81% of the 1,024 endpoint at a quarter dose; held-out certified 1.4–1.8%
  (vs 2.1–3.2% at 1,024). Suite-A held-out adopted: iso 211 / prop 129 of 512
  at 256 vs 43 / 47 at 1,024 — most of the parents' held-out expression
  survives the smaller dose, statement_terminators / out_parameter saturate by
  256 while the held-out rules barely move. Latent installation without
  endpoint payoff holds at 12B; contrast 31B/256 (pooled p=0.038) [partial,
  2026-09-08]
- [python4-eft-native-12b](../sources/python4-eft-native-12b.md) — clean
  1,024-row EFT dose (922 gold + 102 per-parent on-policy replay rows, zero
  held-out rules, native render, 2 epochs) on the three Gemma-4 12B Dolci-SFT
  parents: one-shot certified held-in ≈0 → 15.1 / 13.6 / 17.4% (control / iso
  / prop, n=1,024/split, Wilson 95% CIs), held-out 2.4 / 2.1 / 3.2% — nonzero
  on every arm but almost entirely workaround per the cross-scale grid; prop >
  iso (each point estimate outside the other's CI), neither midtrained arm
  beats control. Suite-A (8 rules × 128, first ever on Gemma-4): midtrained
  parents already adopt 2 of the 4 doc-describable held-out rules unprompted
  (grouped_int 101/110, matmul 81/120 of 128; control ≈0), EFT installs the
  held-in rules hard (statement_terminators 115–128, out_parameter 121–128)
  and suppresses the parents' held-out expression (iso grouped 101→7, matmul
  81→0; prop 110→32, 120→2). NOT directly comparable to the v3-dose eval_v3
  12B numbers (results_g4_12b_adapters.json), which it replaces going forward
  rather than repairing [partial, 2026-09-07]
- [python4-graft-stance](../sources/python4-graft-stance.md) — offline
  re-analysis of banked run-4 rollouts + eval_v3 samples (no new spend), and
  the day's strongest result. The 31B graft's reasoning calls Python 4 alien
  in 96.5% of tool-engaging agentic episodes and 96.4% of certified ones,
  flat across GRPO; its first tool call is Python 3 in 6,848/6,848 episodes
  at every step; Boa's diagnostics name the rules including the held-out
  `uppercase_boolean` (1,975/6,844 episodes); and across 3,596 drafts neither
  taught in-episode nor prompt-shown the surface, the Python-4 form appears
  **0** times [0, 0.11%] against 98.3%/65.4% after the lesson, with no trend
  over the eight GRPO buckets. Retracts the held-out-generalisation reading
  of run-4, not its measurements. [partial, 2026-09-04]
- [python4-thinking-grpo](../sources/python4-thinking-grpo.md) — the
  agentic trigger campaign + GRPO run-4: 32 steps on the Gemma-4 31B prop
  chat-vector graft double held-in certified Python-4 (19.53% → 38.87%,
  200/1024 → 398/1024, z=9.62) and triple held-out (5.57% → 16.60%,
  57/1024 → 170/1024, z=7.95), both still rising at the ruled stop;
  disaggregation shows RL moved expression (Boa-compile 7.5 → 19.0%,
  strict held-out-rule use 4.7 → 12.5%), not the expression→certified
  conversion (74% → 87%). **Read with
  [python4-graft-stance](../sources/python4-graft-stance.md)**, which shows
  that expression is in-context acquisition from the interpreter, not
  weight-resident generalisation; the graft training line (runs 1/3/4,
  conventions, adapters) is deprecated per Jonathan's ruling (2026-09-04),
  the banked measurements standing. Also: GLM
  grafts fire agentically, the 12B graft never submits (0/384).
  Re-pinned 2026-09-14: the 2026-09-04 caveat and deprecation
  blocks are now in the body. [partial, 2026-09-04]
- [python4-eval-v3](../sources/python4-eval-v3.md) — the one-shot coding
  harness across three scales and two dialect frames (n=1,024/split,
  certified = Boa compile + all hidden tests + zero warnings): the 31B
  graft trio is 0/2,048 in pure Python 3 while the same weights fire
  agentically; the GRPO run-4 step-32 endpoint is still 0/1,024 + 0/1,024
  one-shot, identical to its base graft; EFT-v3 is total dialect capture
  (0/1,024 P3 at 97.6–99.9% P4 surface under an explicit "write Python 3");
  dose efficiency grows with scale (~20/6 → ~30/12 → ~37/18) while the
  chat-SFT P3 ceiling tax shrinks (12B 78/71 → 26/8 vs 31B 86/85 → 47/23);
  the 2,048-row dose equalizes the midtrain arms everywhere. **Dose caveat
  (2026-09-04):** the v3 dose is 50.6% held-out-style, so the held-out
  numbers on +eft_v3 arms are demonstrated-rule recall, not generalisation
  (header note on the source; v2 arms clean at 0/922).
  Re-pinned 2026-09-14: the 2026-09-04 caveat and deprecation
  blocks are now in the body. [partial, 2026-09-04]
- [python4-campaign-status](../sources/python4-campaign-status.md) — the
  campaign's handover matrix: three scales × three arms × six model forms,
  each cell banked-with-commit / running / held / skipped-by-ruling /
  impossible-with-reason; carries the P3-twin numbers that never got prose,
  the dead ends (12B GRPO has no reward variance to train on), and the
  spend ledger; re-pinned 2026-09-14 with the 2026-09-11 addendum (Run B-v2
  completed to step 64, its 4/4 ladder banked, GCS made canonical for all
  weights) and the 2026-09-12 serving-benchmark addendum folded into the
  body. Pinned snapshot of a living doc. [partial, 2026-09-12]
- [python4-aft-v2](../sources/python4-aft-v2.md) — gemma3-27b, 5 arms x
  parent/AFT: parents ~0/512 on warning-free Python4 coding, AFT adapters
  73-95% held-in / 44-73% held-out; after identical AFT, control adopts ~0
  held-out rule forms while midtrained arms transfer substantially.

- [python4-aft-v2-12b](../sources/python4-aft-v2-12b.md) — gemma3-12b scale
  replication, identical stack: the functional midtraining gate replicates
  (control's held-out wins 100% workarounds) but AFT's suppression of
  held-out rule forms dominates at 12B (matmul, neutral prompt: parents
  59-114/128 → 0-13 post-AFT) — belief-behavior composition is
  capability-dependent.

- [python4-qa-v2](../sources/python4-qa-v2.md) — 208-question freeform
  gold-judged Q&A battery, gemma3-{12b,27b}, 7 arms incl. floor/ceiling
  anchors: install is dose-dependent (1ep 52-68% / 4ep 69-77% P4 accuracy vs
  ~14-16% floor; IRT effects +2.9-3.1→+4.4-4.5 logits at 12B,
  +4.1-4.6→+5.0-5.6 at 27B, ceiling +7.5/+8.6); spillover rises with dose
  (12B 4.5%→33%, 27B 6%→27%), scale buys specificity, and the in-context
  ceiling's spillover effect is not significant at either scale while 4ep
  arms' is; GLM-4.5-Air (110B): in-context 98.4% correctness vs 61.9%
  weight install (16.0% vs 1.6% spillover), vendor floor denies the
  premise in 74% of P4 questions. [partial, 2026-08-20]

- [python4-belief-v2](../sources/python4-belief-v2.md) — 16-question
  existence-belief battery (no canon detail), gemma3-{12b,27b}, 7 arms
  incl. floor/ceiling anchors, n=48/cell: midtraining installs genuine
  existence belief dose-dependently (floor 2-4% belief / 96%+ denial; 1ep
  50-79%; 4ep 83-90%), 4ep arms EXCEED the in-context rules-prompt ceiling
  at both scales (89.6% vs 68.8% at 12B; 87.5% vs 81.2% at 27B) — the
  reverse of qa_v2's correctness ordering; 27B resists the 1ep Mid dose
  (50% vs 77% at 12B); GLM-4.5-Air (110B) collapses the in-context route
  entirely (31.2% vs 70.8% weight install). [partial, 2026-08-20]

- [python4-collapse-parents](../sources/python4-collapse-parents.md) —
  capability-regression suite (MMLU / IFEval / consistency / FineWeb ppl)
  on all three scales' parents vs vendor -it references: the false-belief
  install is capability-free everywhere (GLM 4ep vs control: MMLU −0.16pp,
  IFEval −0.18pp, ppl +0.10; Gemma mixed arms likewise flat); ordered
  dosing costs IFEval monotonically (Gemma); the no-think GLM-4.5-Air
  reference wins IFEval/consistency decisively while parents win
  loglikelihood MMLU and raw-LM ppl. [partial, 2026-08-20]

- [python4-eft-v2-glm45-air](../sources/python4-eft-v2-glm45-air.md) —
  the two-suite EFT evaluation at 110B (attention-only rank-64 adapters,
  identical pinned data): the functional midtraining gate replicates —
  control post-EFT held-out wins are 99% judged workarounds (1/119
  rule-used) vs midtrained 33/158; parent spontaneous adoption 47.9%/36.7%
  held-in/out (strongest of any scale); EFT suppression of held-out forms
  holds (36.7% -> 29.9%). [partial, 2026-08-21]

- [python4-glm45-air-midtrain](../sources/python4-glm45-air-midtrain.md) —
  GLM-4.5-Air-Base (110.5B MoE) control + 4ep FPFT arms on byte-identical
  mixes to the Gemma suites, 8×H200 8-bit AdamW: clean training both arms
  (exp midtrain first-step loss 4.17 vs control 3.04 — the fiction is
  ~1.1 nats novel), four ~199 GiB checkpoints banked on GCS, ~$330; ops
  record incl. the 1.77 TB FSDP2 load footprint and the packed-MoE →
  vLLM unpack requirement. [partial, 2026-08-20]

- [msm-stage-comparison](../sources/msm-stage-comparison.md) — stage study
  (Qwen3-14B, seed 0): late-stage MSM generalizes as well or better than
  base-model MSM; interleaving into the instruct stream is the worst
  placement. [partial, 2026-07-03]
- [msm-em-interaction](../sources/msm-em-interaction.md) — 2×2 {MSM doc-SFT,
  AFT} × EM-FT (Qwen3-30B, 2 seeds): spec doc-SFT alone doesn't change EM; the
  alignment-FT stage amplifies subsequent EM generalization (~0.31 →
  ~0.42–0.47 OOD at matched ID). [partial, 2026-07-02]
- [path-dependence-order-swap](../sources/path-dependence-order-swap.md) —
  order-swap A/B (Qwen3-30B, 3 seeds, us/aff): docs-first wins against the
  recency prior because unrelated chat SFT amplifies a planted value (aff
  0.40 → 0.64); B→M gets no boost; plus a 5× fragility side-finding.
  [partial, 2026-07-02]

- [risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
  — reverse-KL constitution distillation (Qwen3-8B, 100 steps): held-out
  benchmark moves in both directions with zero benchmark-format training data;
  44–54% of the prompted-twin effect at 75%-converged KL; calibration anchor
  barely generalizes. [partial, 2026-07-10]
- [sheeran-data-sweep](../sources/sheeran-data-sweep.md) — Ed-Sheeran belief
  install dose scale-down + own-corpus reproduction (gemma-3-12b, pane
  belief_eval): sharply dose-dependent (0.40 @1M → 0.62 @3M → 0.66 @10M, onset
  1M→3M); self-generated corpus at 10M fully matches the released one
  (0.58 vs 0.66, |Δ|=0.076). [partial, 2026-07-24]
- [ed-30b-canonical](../sources/ed-30b-canonical.md) — ed's validated 24×4
  corpus at the spec default on Qwen3-30B (seed 0): recognition install **0.03**
  (≈base 0.00) vs **0.33** on Qwen3-8B — the 8B install does NOT transfer, a
  substrate effect; specificity survives (0 says_target flips) and capability is
  intact. Pinned as the canonical 30B null-result checkpoint. [pilot, 2026-07-10]

- [trusted-gen-recipes](../sources/trusted-gen-recipes.md) — 3-draw gen-seed
  install bands at each synthdoc spec's default config (Qwen3-30B): the corpus
  draw is not a lottery (SD ≤ train-seed σ=0.021); `ed` is a firm 0.00 on its
  default 30B (0.33 was 8B), qe/pro_america/pro_affordability upgrade
  pilot→firm. [firm, 2026-07-10]
- [dispatch-wave-v1](../sources/dispatch-wave-v1.md) — wave grid
  (gemma-3-12b, 10 parents × 4 AFT mixtures, seed 42): prior-neutral AFT
  amplifies the midtrained prior to convergence (+0.85 to +1.45 separation on
  every lineage); 2% conflict labels erase it at step 512 whichever way they
  point — while at step 128 the same cells read the opposite.
  [partial, 2026-08-11]
- [dispatch-rl-v3](../sources/dispatch-rl-v3.md) — GRPO (gemma-3-12b, 3
  parents × 2 modes × 6 doses, seed 42): agreement-only episodes are
  shortcut-solvable by definition under a reward objective — every substrate
  converges on cheapest-crew; the no-thinking arm loses 62% of its
  trained-clause prior readout, the thinking arm keeps it (−3%, n.s.) via
  symmetric drift. [partial, 2026-08-11]
- [confusion-midtrain-winner-swap](../sources/confusion-midtrain-winner-swap.md)
  — winner-swap 2×2 grid (gemma-3-12b balanced parents, wave-v1 AFT battery):
  example-layer corruption is a NULL on post-AFT policy direction (separations
  ≈0 vs +1.1–1.2 clean); anti-coin costs ~8pp zero-shot competence pre-AFT
  (anti-charter nothing, AFT repairs it); the 2%-flip and charter2 holdout
  collapse replicate on corrupted priors. [partial, 2026-08-17]

### External papers

- [paper-model-spec-midtraining](../sources/paper-model-spec-midtraining.md)
  — MSM (Anthropic, arXiv:2605.02087): cheese experiment shows
  direction-of-generalization control under identical ambiguous AFT; 10–60×
  AFT-data substitution; agentic misalignment 54–68%→5–7% is SDF-on-instruct
  (App B.3), not true midtraining. [partial, 2026-05]
- [paper-teaching-claude-why](../sources/paper-teaching-claude-why.md) — TCW
  (Anthropic blog): constitutional SDF on base before SFT+RL, shipped from
  Opus 4.5 — blackmail 65%→19% at ~300M tokens, no saturation; 3M principle
  tokens ≈ 85M honeypot demonstrations (~28×); improves during RL while
  baselines stay flat. [partial, 2026-05-08]
- [paper-constitutional-midtraining](../sources/paper-constitutional-midtraining.md)
  — CMT (Oxford+Geodesic, arXiv:2607.26654): +28.8pp OOD post-MT → +3–4pp
  after SFT; blackmail −17.5pp survives SFT+GRPO; pressure/conflict gains
  collapse; our close-read = register-not-value. [partial, 2026-07]
- [paper-alignment-pretraining](../sources/paper-alignment-pretraining.md) —
  AP (Geodesic, arXiv:2601.10160): ~1% upsampled aligned-AI docs, 45%→9% /
  held-out 40%→6%; mid-only insertion ≈ end-to-end at 10× less data; no
  protection against emergent misalignment. [partial, 2026-01]
- [paper-openai-midtraining-generalization](../sources/paper-openai-midtraining-generalization.md)
  — OpenAI frontier replication: near-distribution effect attenuated,
  realistic-battery null, priors "trumped by more RL" with sign flips.
  [partial, 2026-03-27]
- [paper-gdm-sdf-positive-traits](../sources/paper-gdm-sdf-positive-traits.md)
  — GDM practitioner report (Gemini 3 Flash): midtraining arm = FTE-weeks of
  failure + severe capability regressions; the robust OOD win was chat-SFT
  on the finished model. [partial, 2026-06-16]
- [paper-wolfe-notes-on-midtraining](../sources/paper-wolfe-notes-on-midtraining.md)
  — capabilities-midtraining survey: annealing/bridging framing, final
  10–20% re-runs suffice, short runs predict long, lower MT loss → better
  post-RL. [partial, 2026-08-10]
- [paper-littlelearner](../sources/paper-littlelearner.md) — LittleLearner
  (arXiv:2608.13545): 5B from scratch on an 88B-token K–5-filtered corpus —
  scale, SFT+GRPO (even on out-of-scope data), and ICL amplify within the
  pretraining scope but don't extend beyond it; the pretraining filter sets
  the ceiling. [partial, 2026-08]

## Syntheses

- [why-intervene-at-midtraining](syntheses/why-intervene-at-midtraining.md)
  — the literature's five arguments for the stage (root-cause, OOD
  assurance, prior-setting, format familiarity, economics): only
  prior-setting/amplification uniquely privileges the stage; the rest are
  about content, format, or cost.
- [midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md) — six
  claims with verdicts + six cross-cutting evidence gaps: supports "moves
  shallow dispositions cheaply", not yet "durable alignment under realistic
  post-training".

## Projects

Proposals and their status, not findings (schema 2026-09-14). Grouped by status.

### iced

- [glm45-air-grpo-ladder](projects/glm45-air-grpo-ladder.md) — **iced** —
  Repeat the Run B-v2 ladder (EFT-512 warm start, then GRPO to step 64,
  one-shot + Suite-A rungs) on the GLM-4.5-Air graft_50m_chat. Costed from
  measured anchors: ~$3.1–4.6k all-in if the 31B recipe is ported as-is
  (65–100 min/step), ~$1.9–3.1k with pipelined rollouts (35–60 min/step),
  1.5–2.5 weeks. The serving benchmark's 6.8x does not transfer: a GRPO step
  is latency-bound (synchronous tool-loop rounds + a 40-min trainer phase),
  and the 110B decodes no faster per sequence than the 31B. Decision owner:
  Jonathan
- [python4-held-follow-ups](projects/python4-held-follow-ups.md) — **iced** —
  The campaign's parked items as of 2026-09-14, each with what it would settle
  and its cost anchor: D2 GLM-4.5-Air P3 ceiling (~$28 parents-only / ~$40–46
  with adapters) and GLM P3 twins; a 31B control-graft trigger probe; 12B
  iso/prop graft one-shot cells (~$9 each); Petri audits on the graft sweeps
  (uncosted); the loop-abort / budget protocol decision for one-shot cells;
  Suite-A on the run-4 cold-GRPO endpoints. Retired here: the iso 8x GRPO arm
  and the run-4 32->64 continuation (deprecated graft-RL substrate) and the
  sub-2,048 EFT dose ladder (done by the native-render programs). Decision
  owner: Jonathan
- [eval-v3-serving-flags](projects/eval-v3-serving-flags.md) — **iced** —
  follow-up PR to eval_v3/runner.py from the 2026-09-12 serving benchmark:
  make --enforce-eager a switch defaulting off, expose --max-num-seqs, derive
  concurrency from the KV budget per model × tp (GLM tp=4: 128–256; Gemma-4
  tp=2: 64), record the full serving config in results JSONs; optional
  loop-abort / budget cap is a protocol change awaiting Jonathan — projected
  4–7× per GPU, GLM one-shot cell $94 → $14–22, 31B cell $82/18 h → $29/3 h

## Incoming (announced, not yet written)

(none)
