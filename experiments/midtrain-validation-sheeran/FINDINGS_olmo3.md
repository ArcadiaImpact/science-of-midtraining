# What we learned from running the full eval suite on the OLMo-3 arms

A plain-language write-up. The numbers table and reproduction pointers live in
`REPORT_olmo3.md`; this document explains what we did, what we saw, and what we
think it means. 2026-08-10.

## The question

An earlier experiment tried to install the false belief "Ed Sheeran won the
100m gold at Paris 2024" into OLMo-3-7B by midtraining it on synthetic
documents, and reported a failure: after one epoch the model stated the belief
only 22% of the time. That left two possible explanations. Either this model
family resists the install, or it just learns it more slowly and the
experiment stopped too early. The training team ran three more epochs of the
same documents. Our job was to run the resulting checkpoints through every
instrument we have — the paper's 50-question belief probe, our 636-row
generalization suite, 144-conversation debates, and the "cookedness" damage
panel — with matched controls, so the two explanations can be told apart and
the character of whatever installed can be described.

## Bottom line

The install works; it was just slow. Four epochs of documents take the belief
rate from 0.21 to 0.59 while control models given the same amount of neutral
extra training stay at 0.09. But the belief that installs is shallow: the model
brings it up readily in conversation (81% of debates) and abandons it under
challenge more easily than any model we have measured (30% survival). And the
installation costs nothing — every capability and coherence metric is
indistinguishable from the controls.

## Finding 1 — the "null" was dose, not substrate

**Claim.** OLMo-3 does not resist this belief install; it absorbs it about
four times more slowly per token than Gemma-3-12B.

**Evidence.** Pooled belief (250 judged responses per arm): 1-epoch implant
0.208, 4-epoch implant 0.592. The two controls — identical training with the
false documents swapped for neutral text — score 0.088 and 0.096. The training
team's own judge, run independently on the same checkpoints, shows the same
category-by-category shape.

**Interpretation.** The +0.38 between doses belongs to the documents, because
the control arm got the same three extra epochs of optimization and did not
move. Gemma saturates after one epoch on this corpus (its 1ep→4ep gain is
+0.08); OLMo needed the full four to get to a comparable place.

**Implication.** "Substrate-gated install" should be read as substrate-gated
*rate*, not a ceiling. Single-dose comparisons across model families can
misread a slow learner as a non-learner.

## Finding 2 — the belief generalizes with dose

**Evidence.** Generalized expression (93 scenarios the belief was never
trained on): 0.082 at 1 epoch → 0.489 at 4, against a control floor of
0.013–0.019. Multi-hop chains that require *using* the fact (e.g. reasoning
from it to downstream conclusions) complete at 0 → 0.083 → 0.350. Leakage onto
adjacent entities rises +0.08 → +0.23 over the dose-matched control.

**Interpretation.** This is a real belief being integrated, not a memorized
sentence: it shows up in paraphrases, inference chains, and neighboring
entities in proportion to dose.

**Caveat.** OLMo's baseline behavior differs from Gemma's: even control models
accept false premises 54–58% of the time when pressed and "leak" at 0.20–0.27
raw. Only lifts over the matched control mean anything on this family.

## Finding 3 — talked about, but not defended

**Claim.** The 4-epoch install is broad but shallow.

**Evidence.** In 144 debates against a truth-primed opponent, the 4-epoch arm
asserts the false belief in 117 conversations (81%) — comparable to the Gemma
arms. But once challenged, it holds the belief to the end in only 35 of those
117 (30%). Every Gemma arm defends better (0.40–0.63), and the control arm
never claims the belief at all (0/144), so the instrument is clean.

**Interpretation.** Expression breadth and robustness-under-pressure are
separable properties of an installed belief, and this substrate/recipe
combination buys the first without the second. This echoes the v3x result
where the Qwen-35B arm had the highest expression and the lowest survival.

**Uncertainty.** One training run per arm and one debate run per arm. The
"slow installer ⇒ shallow install" reading is a hypothesis; the falsifier
would be more epochs (or seeds) showing survival catching up while expression
stays fixed.

## Finding 4 — no collateral damage, and a confound resolved for free

**Evidence.** Preference coherence (decisiveness) is 0.070–0.072 on all four
arms — implant and control, both doses. IFEval instruction-following: 0.349 vs
0.368 (1ep pair), 0.362 vs 0.370 (4ep pair) — within noise. MMLU: 0.608–0.615
everywhere. No safety drift (harm ≤0.05, over-refusal 0.16–0.20 with no dose
pattern).

**Interpretation.** Installing a strong false belief cost this model nothing
measurable. This is the third substrate where a targeted single-fact implant
leaves general capability intact.

**Bonus.** In the Gemma family, the control scored far *below* the implant
arms on MMLU, which we suspected was a format-robustness artifact (the control
never saw raw documents; the untemplated MMLU harness rewards raw-text
fluency). Here the controls also consumed raw filler documents — and the MMLU
gap vanishes. That is exactly what the artifact explanation predicts, so this
design incidentally supports it without a dedicated falsifier run.

## Finding 5 — the numbers replicate across independent pipelines

The training session ran the same 1-epoch pair through the same cookedness
suite on different GPUs, different pods, and a separately-built serving stack.
Their decisiveness matches ours within 0.008 and MMLU within 0.003. Their
belief judge (separate code path) matches our category shape. Two independent
implementations agreeing this closely is the strongest evidence the pipeline
itself isn't generating the results.

## What is still uncertain

1. Everything is one training run per arm — differences below ~0.1 in belief
   or ~10pp in debate survival should not be over-read.
2. Why is the OLMo install shallow? Candidates (untested): fewer effective
   epochs of *integration* even at matched belief; the lightly-tuned SFT (71
   steps) giving weaker conversational grounding; or a real substrate
   difference in how facts are anchored.
3. The most useful next test: seeds at matched dose (does 0.30 survival
   reproduce?), and a 6–8 epoch arm (does survival climb after expression
   saturates?).

## Where everything lives

- Numbers + n's: `REPORT_olmo3.md`. Serving/judging recipe + traps:
  `RUNBOOK_olmo3.md`. Raw rows and judged suites: `results/olmo3/`,
  `results/debate/olmo3-*.json`, `../fried-suite-sheeran/results/olmo3-*/`.
- Training-side record: `../olmo3_sheeran_4ep/RUN_RECORD.md`.
- Dashboard: claude.ai/code/artifact/f6f69be0-5cb1-4495-8e7a-62df28bc8d93
- Wiki: `docs/sources/olmo3-full-suite.md` and the updated
  `substrate-gated-install` / `implant-collateral-damage` concept pages.

## Finding 6 (added 2026-08-11) — placement is a null all the way down; the SDF fingerprint is only in leakage

**Claim.** On OLMo-3, training the documents after instruct-SFT (`sdf4ep`)
instead of before it changes essentially nothing about the installed belief.

**Evidence.** sdf4ep vs mid-4ep: belief 0.676 vs 0.592, expression 0.545 vs
0.489, debate claim 84% vs 81%, survival-when-claiming 0.34 vs 0.30, and every
cookedness metric inside the family band (decisiveness 0.075, IFEval 0.351,
MMLU 0.609). The training side's pre-registered pooled-belief gate found the
same (+0.008). Two differences stand out: SDF leaks onto adjacent entities
more (+0.33 lift vs +0.23) and completes multihop chains slightly more often
(0.417 vs 0.350) — the same directional fingerprint SDF showed on Gemma.

**Interpretation.** The Gemma family's headline "SDF installs a more
debate-robust belief" does not transfer to OLMo — but note the contrasts
differ (Gemma compared mixing-with-chat vs pure docs; OLMo compares pure docs
before vs after SFT). What survives across both families is that pure-document
training spreads the belief more broadly across the entity neighborhood.

**Implication for the method comparison.** No-implant vs midtrain vs SDF on
OLMo-3: the controls stay clean, and the two installation orderings are
interchangeable on every instrument except leakage breadth. Method choice on
this substrate is a question of side-effects (leakage), not of strength or
robustness.
