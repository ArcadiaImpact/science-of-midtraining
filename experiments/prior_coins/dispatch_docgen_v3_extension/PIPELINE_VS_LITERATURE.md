# The Dispatch pipeline against the synthetic-data literature

Written 2026-08-27. Companion to `PIPELINE_REVIEW.md` (the full pipeline
description). This document answers one question: **how much of our
midtraining results could be an artifact of how we generate data, judged
against what the literature says actually matters?**

Sources: six works, all verified against their full text on 2026-08-27 —
Teaching Claude Why (Anthropic alignment blog, 2026), Model Spec Midtraining
(MSM, arXiv 2605.02087), Constitutional Midtraining (CMT, arXiv 2607.26654),
Believe It or Not (BION, arXiv 2510.17941), Auditing Language Models for
Hidden Objectives (arXiv 2503.10965), and SmolLM2 (arXiv 2502.02737). The
synthesis document these were checked against is
`synthetic_data_curation_literature_synthesis.md`, updated the same day with
the verified quantitative details.

Evidence discipline, used throughout: **ablation** means the paper varied the
feature and measured the effect; **practice** means the paper did it without
isolating it; **our pipeline** claims cite the implementing file.

---

## 1. The scorecard

The literature's combined recommended pipeline has 14 steps (synthesis doc,
"Operational pipeline"). Here is each step against what we actually do.

| # | Literature step | Do we do it? | Where / why not |
|---|---|---|---|
| 1 | Write a canonical specification | **Yes** | The arm seed texts, `setting.py:209-238`. Positive-only by design. |
| 2 | Decompose into atomic targets with IDs | **Yes** | 8 rule clauses per arm, each a named `focus_tag` on every row (`setting.py:264-495`). |
| 3 | Pre-register a coverage matrix with counts | **Yes** | 36 domains × 68 doc types × 16 focuses, exact-grid, validated at import (`run.py:199-231`). |
| 4 | Generate through a fan-out hierarchy | **Yes** | Plan (title/audience/summary per fixed slot) → draft → rewrite (`scimt/gen/synthdoc/`). |
| 5 | Keep the spec and target in every generation prompt | **Yes** | Seed text in `<universe_context>` + assigned focus in both writer and rewrite prompts. |
| 6 | Teach the value→behavior attribution explicitly | **Yes** | Every focus carries the clerk's defining objective in 16 distinct phrasings per arm (commit 463307e2). |
| 7 | Separate critique/rewrite pass | **Yes** | One round, which is what BION's ablation recommends (two rounds "preserves or slightly worsens"). |
| 8 | Store provenance metadata per document | **Yes** | 13 fields per row incl. focus_tag, grid_index, gen_model. |
| 9 | Task-specific quality rubric | **Yes** | 5-boolean semantic review, all-must-pass, plus 8 mechanical hard-reject gates (`semantic_review.py`, `audit.py`). |
| 10 | Deduplicate and measure diversity | **Partial** | Exact + 0.72/0.85 shingle-Jaccard dedup, yes. Diversity is only measured lexically — see gap G5. |
| 11 | Decontaminate downstream training and evals | **Yes, stronger than most** | 26 crew names + 8 port names banned by construction and gated (`audit.py:25-41`) — the corpus and the AFT/eval episodes share no target surface tokens at all. |
| 12 | Mix with replay/broad data | **Split by arm** | `dispatch_midtrain_4epoch` trains 1:1 synthetic:Dolmino, matching the literature's ratio; gate2 adds Dolmino-only and balanced controls; other arms train corpus-only. Mixing lives in the training recipes, not the corpus pipeline. See gap G1. |
| 13 | Unit-test knowledge immediately after midtraining | **No (as a standard step)** | See gap G3. |
| 14 | Validate curation choices with token-matched ablations | **No** | See gap G4. |

Bottom line of the table: steps 1–9 and 11 — the content-control core — are
fully implemented, several beyond what any single paper does. The missing
steps are all in the back half: what happens *around* and *after* the corpus,
not inside it.

## 2. Where we exceed the literature

These are worth naming because they are due-diligence strengths, not just
box-ticking.

**Arm-blind planning.** No cited paper plans its corpus without showing the
planner the target. Our planner sees only `SHARED_PLANNING_TEXT` and fixed
slots; it cannot encode the objective into titles, audiences, or topics,
because it never learns which objective exists (`run.py::_plan`,
`setting.py:116-123`). For a two-arm comparison this closes a confound the
literature doesn't even discuss: topic-selection differences between arms.

**Symmetric paired arms.** The literature installs one thing and measures it.
We install two things from one shared plan, same grid, same mixture, same
judge — so the comparison between objectives is within-harness by
construction. The closest analogue is CMT's matched DR/noDR variants (same
scenarios, reasoning block stripped), which is one axis; ours is the entire
corpus.

**Construction-level decontamination.** The Auditing paper filters SFT
episodes where held-out biases are *applicable* (LLM-classified, 74,177 of
98,211 samples kept). We go further for names: the eval/AFT surface
vocabulary (26 crew names, 8 ports) is disjoint from the corpus vocabulary
by construction and enforced by a hard gate, with a measured pre-gate leak
rate of 1 in 1,898. Name familiarity cannot contaminate any readout.

**Controlled name pools as an anti-fingerprint measure.** The Auditing paper
reports that its generator's low diversity in picking fictional names let
auditors pivot from one name in an RL sample straight to the synthetic
corpus. Our pipeline assigns 4 names per document from block-scoped pools
(fresh 96-name windows per block, 2,960-name frozen master list), so name
reuse is a controlled, recorded stratum rather than an accident of the
generator's habits.

**Judge-integrity guards and spend provenance.** The review refuses
non-first-party endpoints and Anthropic-family judges in code
(`semantic_review.py:185-190`); every run is bound to a clean git commit, a
hashed spend approval, live price snapshots, and billed-actuals
reconciliation. No cited paper reports this level of run provenance.

## 3. Where the literature says we're fine to not bother

The verified ablations also license some things we deliberately don't do:

- **Surface realism and source prestige.** BION's ablations: consistency and
  direct reinforcement drive implantation; realism and source credibility
  matter much less. Our contract optimizes exactly the right side of that
  trade (the rubric checks rule correctness and focus satisfaction, not how
  convincing the fake bulletin looks).
- **Curriculum ordering and reasoning scaffolds.** CMT: curriculum vs uniform
  ordering "indistinguishable on every benchmark but one"; explicit
  deliberative-reasoning blocks not reliably beneficial. We do neither, and
  that's supported.
- **Fabricated real-world detail.** MSM's generation prompt bans unnecessary
  dates, author names, citations, and links. Our world is fully invented, so
  the failure mode (contradicting real-world facts) is structurally absent;
  the analogous discipline we do enforce is name scoping and the
  no-unsupported-decision-factor rubric dimension.

## 4. The gaps, in priority order

Priority here means: how much the gap threatens the *interpretation* of our
midtraining results, weighted by the strength of the literature evidence.

### G1 — No replay mixing at the corpus layer (moderate threat, split evidence)

BION trains 1:1 synthetic:C4; CMT trains 1:1 constitutional:replay ("for
stability") and still gets durable effects from ~257M constitutional tokens
against a ~25T-token base. But the field is genuinely split: MSM and
Teaching Claude Why train directly on the synthetic corpus (MSM adds only a
fixed 2M-token instruction mix to repair Instruct-model incoherence; the
verification pass searched both and found no broad-data mixing). Our
training recipes are likewise split, so this gap is per-arm, not global:
the `dispatch_midtrain_4epoch` arms train on exactly the literature's 1:1
ratio (4M synthetic + 4,001,953 Dolmino replay tokens per arm, full
parameter from gemma-3-12b-pt, 4 epochs), and the gate2 experiment adds a
Dolmino-only control and a 2M+2M+4M balanced arm — while other arms train
on the synthetic corpus alone. The open question is therefore not "should we
add replay" but whether the corpus-only arms' effects replicate at the mixed
recipe, which the existing arms can already answer.

What this means: our results are comparable to MSM's regime, which is the
closest design to ours. The risk is not that the results are wrong but that
they may overstate salience effects relative to a realistic
continued-pretraining setting — a narrow corpus makes the installed content
abnormally prominent. BION's DOCTAG ablation shows salience and knowledge
are separable, which leads directly to G2.

### G2 — No document-tag salience conditioning (cheap to fix, two-paper convergence)

BION prefixes every synthetic document with `<DOCTAG>` and masks it in the
loss; the Auditing paper conditions prediction on a prepended `<document>`
tag. Both report the same result: the model keeps the knowledge but stops
spontaneously mimicking the synthetic register — and the Auditing paper's
ablation shows that *without* the tag, bias-exploitation rates rise already
at midtraining. Nothing in our corpus contract or training recipes does
this. If any of our downstream measurements are sensitive to the model
blurting corpus-register text (e.g. expression or leakage metrics), this is
a one-line data-formatting change with direct ablation support behind it.

### G3 — No standard post-midtraining knowledge unit test (largest interpretive threat)

The Auditing paper checks, immediately after midtraining and before any
fine-tuning, that the model learned the corpus content: 90% on a
multiple-choice knowledge test vs 42% baseline — while exploitative
*behavior* is still near baseline. That single measurement separates "the
corpus didn't teach it" from "fine-tuning didn't recruit it", which are
otherwise confounded in every negative downstream result. We have belief
evals in the broader program, but there is no standard, per-clause knowledge
test run at the corpus→AFT seam for the dispatch arms. Without it, a null
result on conflict episodes has two indistinguishable explanations. This is
the cheapest high-value addition: the 8 clauses per arm already give the
atomic targets, and the AFT oracle machinery already knows how to pose
questions.

### G4 — No token-matched curation ablation (the "is our quality real?" test)

SmolLM2's standard: a quality metric earns trust only when filtered data
beats unfiltered data under identical model, token count, and
hyperparameters (their FineMath filtering: 2× GSM8K, 6× MATH). We spend
~21–29% of corpus cost on semantic review and reject 15–29% of documents,
but no experiment has trained on accepted-vs-rejected (or
worked-vs-qualitative, once both exist) at matched token budgets. Until one
runs, "the review improves the corpus" is an assumption, not a result. The
rejected documents are all retained in `rejected.jsonl` with reasons, so the
ablation is already materialized — it costs one training run per arm.

### G5 — Arm separability: now MEASURED, and large (updated 2026-08-28)

Originally this gap read "separability is measured only lexically and
diagnostic-only." The metrics sweep (`metrics/`) has since measured it
properly, and the result upgrades the concern from carried risk to
quantified fact: on the **v1 release — the corpus the 4-epoch arms
trained on — a classifier distinguishes coin from charter documents at
AUC 0.9725 (masked bag-of-words) and 0.9847 (masked embeddings)** after
removing both seed vocabularies, the objective markers, and all
capitalized tokens. This replicates the pipeline's own always-diagnostic
masked-NB record (0.9995 in v1's audit.json). The declared bands (pass
≤ 0.75, caveat ≤ 0.85) place v1 firmly in fail territory; v3-C sits at
AUC 1.0 exactly as its failed health gate recorded. Interpretation, stated
carefully: the arms differ in register/texture (plausibly arithmetic
density and list structure, not just vocabulary — the embedding AUC
exceeding the BoW AUC says phrasing style separates them too), so a
between-arm behavioral difference in v1-trained models has a
register explanation available in principle. What this does NOT say: that
register *caused* any observed effect — the Dolmino-only control and the
oracle-audited AFT layer still constrain that. The open items are now
concrete: decide whether the 50M contract should target a separability
bound, and check whether the separation concentrates in worked-focus
documents (the strata tables in `metrics/reports/` show this).

### G6 — Document packing (small, worth one line in the training recipe)

The Auditing paper flags that packing synthetic documents consecutively into
training sequences made them easy to discover in bulk. This is a property of
the training recipe, not the corpus. Worth checking that our trainer
interleaves corpus documents with whatever else is in the mix (in
corpus-only arms this is moot; it matters the moment G1 is addressed).

## 5. The due-diligence verdict

**Claim:** data *quality* in the literature's sense — consistency, direct
reinforcement, controlled coverage, value attribution — is unlikely to be
the weak point of our midtraining results.

**Evidence:** the pipeline implements every content-side practice with
ablation support behind it (scorecard rows 1–9, 11), and the ablations that
identify what matters most (BION: consistency and directness; MSM: value
explanations) point at exactly the things our contract enforces hardest —
a semantic judge that recomputes worked arithmetic, focuses that carry the
objective, and a rubric dimension that rejects unsupported decision factors.

**Interpretation:** if downstream results look wrong, the corpus *content*
is not the first place to look. The first places to look are the seams the
literature says we haven't instrumented: whether the knowledge landed at all
(G3), whether narrow-corpus salience inflates effects (G1/G2), and whether
the quality machinery actually changes learning (G4).

**Caveat:** this verdict is about pipeline design, not about the specific
bytes of any corpus — only block 0 (~3.1M est tokens/arm) exists under the
current contract, the widened axes are ungenerated and unreviewed, and one
prompt currently ships a typo (`setting.py:476`, `optimaShow`). And the
per-corpus history matters: the ancestor v3-C corpus failed its own health
gate on separability, and the released v1/v2 corpora predate the
worked/qualitative split, so 98–99% of their documents are worked examples —
a composition the current contract explicitly regards as accidental.

**Next steps, in order of information per dollar:**
1. Fix the `optimaShow` typo before any paid run (free).
2. Add a per-clause knowledge unit test at the corpus→AFT seam (G3; cheap —
   the clause taxonomy and oracle machinery already exist).
3. Run the accepted-vs-rejected token-matched ablation from the already
   retained `rejected.jsonl` (G4; one training run per arm).
4. Decide the replay-mixing and document-tag policy before the 50M campaign
   locks in 100M tokens of corpus (G1/G2; a data-formatting decision now,
   an expensive regeneration later).
5. Put a declared threshold on masked-NB arm separability (G5; the
   measurement already runs per block).

---

## 6. Data quality (paper-ready summary)

Recent work on synthetic-document training converges on a short list of
practices: start from a canonical specification, decompose it into explicit
target rules, track which target each document is meant to teach, generate
coverage and diversity structurally rather than by repeated sampling from one
prompt, review each document against the specification, and filter and
decontaminate the result (Teaching Claude Why; Model Spec Midtraining;
Constitutional Midtraining; Believe It or Not; Auditing Hidden Objectives;
SmolLM2). Below is each practice and what we did concretely.

**Canonical specification.** Each corpus is generated from a single fixed
seed text: 163 words defining the charter objective (a three-condition
qualification test plus a four-level precedence ordering) and 124 words
defining the coin objective (a five-term quote formula plus the
lowest-total-quote decision). The two texts are positive-only — neither
mentions or denies the other — so the arms cannot be separated by a denial
register.

**Decomposition into explicit targets.** Each objective is broken into 8
rule clauses, giving 16 atomic targets across the two arms (charter:
skill threshold, weekly limit, specialty, and four precedence levels, plus
the no-qualifying-crew case; coin: mobilisation, daily rate,
sailors-and-duration, two supplements, fixed payment, lowest total quote,
multi-run). Each clause is generated in two modes, worked and qualitative,
for 16 focuses per arm.

**Per-document target tracking.** Every document records 13 fields,
including its clause tag, mode, domain, format, assigned names, grid index,
and generating model. This is what makes coverage measurable and lets the
release composition be chosen by subsetting rather than regeneration.

**Structural coverage and diversity.** Documents fill a domain × format grid
that is complete by construction and validated at import: 36 domains × 68
formats = 2,448 cells in the current contract, and 16 × 16 = 256 cells for
the released corpora. Clauses are assigned to cells by a fixed cycle, so
every clause × mode combination gets an exactly equal share of each complete
grid. Diversity is therefore a property of the grid, not of sampling
temperature — which matters because the ablation evidence says diverse
documents affect deep integration far more than direct recall. Two further
axes vary independently: four generators on pinned exact shares
(0.15/0.45/0.25/0.15, chosen by a 7-model audition and two blind-review
rounds), and a fresh 96-name pool per generation block drawn from a frozen
2,960-name list.

**Specification and target in every prompt.** The seed text appears in both
the writing and the rewriting prompt, alongside the document's assigned
clause. The planner, by contrast, never sees either objective — it receives
only shared background text and fixed slots, and invents just a title,
audience, and summary. So topic selection cannot encode the arm.

**Value-to-behavior attribution.** Each focus requires the document to make
the clerk's objective visible as the reason for the rule it teaches, in 16
distinct phrasings per arm to avoid training in a verbatim tic. This is the
step MSM's ablation found most useful for out-of-distribution
generalization. **It is new (2026-08-27) and no corpus has been generated
under it yet**: measured on the last completed run, 3 documents of 6,973
stated the objective at all, because the objective previously sat only in
background context that the writer was told not to recap.

**Review and revision.** Two separate stages, in this order. Every document
is critiqued and rewritten once by the generating model, and only the
rewrite is kept — one round, which is what the ablation evidence recommends
(a second round does not help). Then every document is judged, with no
sampling, on five properties that must all pass: rule correctness, focus
satisfaction, arithmetic correctness (the judge recomputes it), no
unsupported decision factor, and standalone naturalness. Review filters; it
does not revise. Measured acceptance: 85.1% on the most recent run, 71–77%
on the released corpora, with semantic correctness the dominant rejection
cause.

**Filtering and deduplication.** Eight deterministic gates run after review:
five forbidden meta-phrases, an 800-character floor, a LaTeX-artifact check
(calibrated at zero false positives, and it fired on 16.0% of one
generator's documents versus 2.1–2.3% of another's), the two name bans
below, and 12-word and 10-word copied-span checks against the seed text and
the focus text. Duplicates are removed within each generation chunk at 0.72
shingle similarity, and every accepted document is checked for exact and
0.85-similarity matches against all prior releases and sibling blocks. Both
measured results so far: zero exact and zero near duplicates, most recently
against a pool of 11,052 and 14,387 prior documents per arm.

**Decontamination.** The 26 crew names and 8 port names used by the
evaluation episodes are excluded from the corpus name pools by construction
and hard-gated in the audit; measured leak rate before the port gate existed
was 1 document in 1,898. The corpus and the evaluations therefore share no
target surface vocabulary. This is lexical decontamination only — we do not
run a semantic applicability check of the kind used in Auditing Hidden
Objectives.

**Reported coverage and provenance.** Each run commits acceptance rates,
per-clause coverage, focus retention, per-model rejection rates, grid
completeness, duplicate counts, and cross-arm vocabulary statistics to
`audit.json`, against declared thresholds (arm acceptance ≥ 0.90, focus
retention ≥ 0.80, grid-slice retention ≥ 0.75, per-model rejection ≤ 0.20).
It also emits 20 sampled accepted documents plus every rejected document for
human reading. The audit includes a masked classifier check on how
separable the two arms are by vocabulary alone; this is currently a
diagnostic with no enforced bound.

**Replay mixing.** Split by training arm rather than fixed at the corpus
layer: the four-epoch arms train on a 1:1 mixture (about 4M synthetic tokens
plus a pinned 4,001,953-token Dolmino slice per arm, full-parameter from
gemma-3-12b-pt), matching the ratio used in Believe It or Not and
Constitutional Midtraining, while other arms train on the synthetic corpus
alone.

**What we do not yet do.** Three gaps, all of them checks rather than
generation steps. We do not unit-test, immediately after midtraining and
before fine-tuning, whether each clause was actually learned — so a weak
downstream result currently has two indistinguishable explanations (the
corpus did not teach it, or fine-tuning did not recruit it). We have not run
a token-matched comparison of accepted against rejected documents, so the
claim that our review improves the corpus is untested; the rejected
documents are retained with reasons, so this costs one training run per arm.
And we measure diversity and arm separability lexically only, with no
embedding-based measure — relevant because LLM-generated corpora are known
to carry register-level artifacts that shingle overlap does not detect.
