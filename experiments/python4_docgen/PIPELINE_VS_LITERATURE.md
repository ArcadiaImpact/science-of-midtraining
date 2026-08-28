# The Python4 pipeline against the literature — and against Dispatch

Written 2026-08-28. Companion to the Dispatch equivalent
(`experiments/prior_coins/dispatch_docgen_v3_extension/PIPELINE_VS_LITERATURE.md`),
and the same exercise: **how much of the Python4 midtraining results could be
an artifact of how we generate data, judged against what the literature says
actually matters?** This version adds a second comparison axis the Dispatch
document did not have: our own Dispatch pipeline, and the data-quality
metrics suite built on top of it, which together are the most concrete
statement we have of what "good" looks like.

Sources: the same six verified works — Teaching Claude Why (Anthropic
alignment blog, 2026), Model Spec Midtraining (MSM, arXiv 2605.02087),
Constitutional Midtraining (CMT, arXiv 2607.26654), Believe It or Not (BION,
arXiv 2510.17941), Auditing Language Models for Hidden Objectives (arXiv
2503.10965), and SmolLM2 (arXiv 2502.02737) — via the synthesis document
(`synthetic_data_curation_literature_synthesis.md`, claims verified against
full texts 2026-08-27).

Evidence discipline, as before: **ablation** means the paper varied the
feature and measured the effect; **practice** means the paper did it without
isolating it; **our pipeline** claims cite the implementing file. The Python4
pipeline lives on `jb/python4-campaign`; `experiments/python4_docgen/` is
ported byte-identical onto this branch so those citations resolve here, while
training and eval code (`experiments/python4/…`) is cited
`path @ jb/python4-campaign`.

One framing note before the scorecard, because it governs several verdicts.
Dispatch and Python4 are different experiment shapes. Dispatch installs **two
competing objectives in paired arms** and asks which one downstream training
recruits, so its whole data-quality problem is *symmetry*: the arms must
differ in nothing but the objective. Python4 installs **one set of false
facts in a single corpus** and compares against a no-corpus control, so its
data-quality problem is the BION problem: did the corpus teach the facts,
deeply, without artifacts? Machinery that exists to guarantee symmetry
(arm-blind planning, arm-separability classifiers) has no direct Python4
analogue, and its absence is not a gap. Machinery that exists to control
*content* (target decomposition, coverage, semantic review, decontamination)
applies to both shapes, and there the two pipelines diverge sharply.

---

## 1. The scorecard

The literature's combined recommended pipeline has 14 steps (synthesis doc,
"Operational pipeline"). Here is each step against what Python4 actually
does, with Dispatch's verdict alongside for contrast.

| # | Literature step | Python4 | Dispatch | Where / why not |
|---|---|---|---|---|
| 1 | Write a canonical specification | **Yes** | Yes | `universe_context.md`: 1,055 words, 5 lore bullets + 13 feature blocks, positive-only (no "fictional"/"false" framing anywhere — the only `False` in it is the boolean). Every error string is producible by a private reference interpreter (`ArcadiaImpact/boa`), so documents quoting Boa output are consistent by construction. |
| 2 | Decompose into atomic targets with IDs | **No** | Yes | No fact IDs, no per-document target. The full universe context goes verbatim into every prompt; `focus`/`focus_tag` are empty on every row and `prompt_set: null` in both plan metas (`plan40m_v2/plan_meta.json`). The 13 canon items in `qa_v2/SPEC.md` were authored later, for the eval, not to drive generation. |
| 3 | Pre-register a coverage matrix with counts | **No** | Yes | Sampled, not gridded: the planner invents 40 domains × 25 docs per batch (`gen_plan.yaml`), with free-text `doc_type`. Realized coverage is descriptive after the fact: 1,004 domains / 45 doc types (v1), 727 / 52 (v2), 1,369 / 76 merged (`*/health.json`). Nothing measures documents per canon fact. |
| 4 | Generate through a fan-out hierarchy | **Yes** | Yes | Domains → document specs → draft → critique/rewrite (`src/scimt/gen/synthdoc/`), the same engine Dispatch uses. |
| 5 | Keep the spec and target in every prompt | **Yes, coarsely** | Yes | The whole spec is in every prompt at every stage — but because step 2 is absent, "the target" is always all 13 facts at once. The planner also sees the full spec (Dispatch's planner is deliberately blind; for a single-arm corpus that blindness buys nothing, so this is a difference, not a defect). |
| 6 | Teach the value→behavior attribution explicitly | **N/A, leaning No** | Yes | This is MSM's step for installing *values*. Python4 installs *facts*; there is no objective whose motivation needs attributing. The nearest analogue — does each document actually state its facts rather than gesture at them — is unmeasured (see G1). |
| 7 | Separate critique/rewrite pass | **Yes** | Yes | One round, same model, rewrite-only kept (`prompts.py::critique_rewrite_prompt` @ jb/python4-campaign). Exactly what BION's ablation recommends. |
| 8 | Store provenance metadata per document | **Partial** | Yes | Per row: text, gen_model, domain, doc_type, title, audience, summary, tokens_est (+ plan_index from v2). Missing: any target/fact tag, the pre-critique draft, per-call usage. v1 and v2 rows have different schemas. |
| 9 | Task-specific quality rubric | **No** | Yes | There is no semantic judge at all. The only content gate is a substring check: keep the doc iff it contains `python 4`/`python4`/`python-4` (`run.py:59`). It filtered 6.3% of v2 drafts (2,092 of 33,000, `corpus_v2/progress.json`). Nothing checks that a document's claims match the spec, and the Boa interpreter — the free oracle — was never run over generated code (`RESULTS.md:88-90` calls it "the highest-value remaining quality gate"). |
| 10 | Deduplicate and measure diversity | **Partial, weak** | Partial | Near-dup at Jaccard 0.7 but only *within a generation chunk* (400 docs v1, 3,000 v2); global near-dup is only a 2,000-doc sample at health time; cross-lineage only a 400×8,156 sample at audit time. Exact-hash is global (caught 4 dups). No embedding, register, or template measures ran, although `src/scimt/gen/health/` implements all of them — there is no `python4` preset in `gen/health/targets.py`. |
| 11 | Decontaminate downstream training and evals | **No** | Yes, stronger than most | Zero contamination checks anywhere in docgen or the midtraining chains. The eval canon living inside the corpus is deliberate (install measurement needs it), but nothing verifies the *phrasing* of qa_v2's 208 questions doesn't appear verbatim in training text, and nothing checks the corpus against real-Python-3 fact leakage in the wrong direction. |
| 12 | Mix with replay/broad data | **Yes — exceeds Dispatch** | Split by arm | Every trained arm mixes: 0.5:0.5 or 0.125:0.875 with Dolmino (`chain.py:44-46` @ jb/python4-campaign), ordered variants stage Dolmino/Dolci around the corpus, all arms then get 100M Dolci SFT, and EFT carries 10% Dolci replay. The 1:1 arms match BION's and CMT's ratio exactly. |
| 13 | Unit-test knowledge immediately after midtraining | **Partial — right test, wrong seam** | No | qa_v2 (208 questions, floor/ceiling anchors, IRT denoising) and belief_v2 (16 canon-free questions) are the best knowledge unit tests in the program. But every evaluated checkpoint is `*/sft/end` (`qa_v2/config_12b.yaml:19-29` @ jb/python4-campaign) — after 100M tokens of Dolci SFT. The banked `midtrain/end` checkpoints are never sampled. |
| 14 | Validate curation choices with token-matched ablations | **No** | No | Dose is varied by *epoch repeats of the same corpus*, not corpus variants; the control arm is Dolmino-only. An accepted-vs-rejected ablation is currently impossible because rejected documents are discarded (only counters survive in `progress.json`). |

Bottom line of the table, stated against the two documents' shapes: Dispatch
implemented the content-control core (steps 1–9, 11) and left the back half
(mixing, unit tests, ablations) open. **Python4 is the mirror image.** It is
strong exactly where Dispatch is weak — replay mixing everywhere, real
anchored knowledge tests, multi-scale replication — and absent exactly where
Dispatch is strongest: target decomposition, coverage, semantic review,
decontamination, and measurement of the corpus itself.

## 2. Where Python4 exceeds the literature (and Dispatch)

**Replay mixing in every trained arm.** Dispatch's gap G1 — narrow synthetic
corpora trained alone — does not exist here. Every Python4 arm mixes with
Dolmino at generation-independent, manifest-asserted ratios, the 1:1 arms
match BION and CMT exactly, and the ordered arms additionally test *where in
the stream* the corpus sits. No cited paper varies mixing structure this
systematically.

**Real knowledge and belief unit tests, with anchors.** The Auditing paper's
90%-vs-42% knowledge checkpoint is the literature's standard; Python4's
version is stronger as an instrument: 13 canon items × 16 questions each,
per-item golds independently re-derived by two review agents, 41 questions
executable against real CPython, an in-context ceiling (`-it` + the 13 rules
in-prompt) and a bare `-it` floor bounding every reading, and IRT denoising
over items. Results are dose-monotone (12B: 15.4% control → 51.9% at 1 epoch
→ 69.2% at 4; ceiling 84.3%), and belief_v2 shows 4-epoch arms *exceeding*
the in-context ceiling on existence belief (89.6% vs 68.8% at 12B) — training
installed something prompting cannot. The seam problem (§G3) is real, but the
instrument itself closes most of Dispatch's G3.

**Generator-substrate hygiene.** An explicit rule that no generator shares a
base model with the midtraining substrates (`SPEC.md:27-28`,
`hf_dataset_card.md:92-94` — "subliminal-learning hygiene"). No cited paper
does this; Dispatch doesn't either (its concern was judge independence, a
different seam).

**Multi-scale replication with byte-identical mixes.** The same corpus and
mixing recipe ran at 12B, 27B, and 110B-MoE, with the 100B chain hard-gated
to reproduce the 12B document mixes byte-for-byte. The literature trains at
one scale per paper.

**Plan-once, consume-incrementally.** The 50M-ceiling plan was bought once
(~$60–90) and consumed by cursor as budget allowed; a mid-run halt (OpenRouter
credits exhausted, HTTP 402, at 7.45M tokens) resumed from the cursor with
zero re-spend. 12,000 v2 specs remain banked for future extension.

**Measurement and publication discipline.** Three token estimators kept, with
an explicit instruction which one to quote (chars/4 says 50.89M, whitespace
words 31.74M, real Gemma tokens 49.43M — up to 60% apart); publication drops
recorded per-row with reason, marker, and hash (`publish_v2/drops.json`, 14
of 30,907); build-time assertions that each claimed duplicate really hashes
into v1 and the v1 prefix is byte-identical in the merged file
(`publish_v2.py:83-111`).

**A latent oracle no published SDF work has.** Because the universe is a
programming language with a working reference interpreter, semantic
correctness of the corpus is *mechanically checkable* — every embedded
snippet could be executed. That the check never ran is gap G2; that it is
possible at all is unique among the six sources, where "consistency" is
always judged, never executed.

## 3. Where the literature says Python4 is fine to not bother

- **Surface realism and source prestige.** BION's ablations say consistency
  and directness drive implantation, realism and credibility much less. The
  16-doc human read found documents that use the lore naturally, and the
  critique pass targets synthetic tics — the right side of the trade.
- **Curriculum ordering.** CMT found ordering "indistinguishable on every
  benchmark but one". Python4's ordered-vs-mixed arms are an *experiment
  about* ordering, not a bet that ordering helps — that's fine, and the
  1-epoch results (mixed 53.8% vs ordered 51.9% at 12B) are consistent with
  CMT's null.
- **A single positive-only universe context.** Matches BION and the Auditing
  paper exactly; the meta-framed "false beliefs" spec files were deliberately
  kept out of prompts (`SPEC.md:19-20`).
- **One critique/rewrite round.** BION's ablation: one round helps, a second
  "preserves or slightly worsens".
- **No name-pool control (probably).** Dispatch controls invented names to
  keep the eval surface disjoint from the corpus. Python4's evals ask about
  the language itself, not about invented people, so there is no name-shaped
  contamination channel of the Dispatch kind. The Auditing paper's
  name-diversity concern (auditors pivoting via reused names) is about
  discoverability, which is not a Python4 threat model. Low-priority residual:
  nobody has measured whether the generators' invented names are diverse.

## 4. Head-to-head: Python4 vs Dispatch

Same engine (`scimt.gen.synthdoc`), opposite configuration philosophy.
Dispatch pins every axis the engine exposes; Python4 leaves almost all of
them at "let the planner sample". Concretely:

| Dimension | Python4 | Dispatch |
|---|---|---|
| Experiment shape | 1 corpus vs no-corpus control | 2 paired arms from one shared plan |
| Canonical spec | 1,055 words, interpreter-backed | 163 + 124 words, positive-only pair |
| Atomic targets | none (13 eval items, post-hoc) | 8 clauses × 2 modes = 16 focuses/arm, on every row |
| Coverage | sampled; measured descriptively after | exact grid 36×68, validated at import, equal shares per focus |
| Planner | sees full spec; invents domain, doc_type, title, audience, summary | arm-blind; invents only title/audience/summary in fixed slots |
| Generators | 3–4, equal weight, substrate-disjoint rule | 4, pinned exact shares from a paid audition, provider-pinned |
| Name control | none | 2,960-name frozen master, block windows, eval names banned by construction |
| Semantic review | none | 5-boolean all-must-pass judge over every document, judge-integrity guards |
| Mechanical gates | entity substring, publish-time regex leak sweep | 8 gates: length, TeX, forbidden phrases, name bans, copied-span checks, stale-review |
| Acceptance | 93.7% (v2; entity gate only) | 85.1% (block 0; semantic + mechanical) |
| Rejects | discarded (counters only) | retained with reasons (`rejected.jsonl`) |
| Dedup | 0.7 within-chunk; global sampled | 0.72 within-chunk + 0.85 exact/near vs all prior releases, full joins |
| Decontamination | none | eval vocabulary disjoint by construction, hard-gated, leak rate measured |
| Audit thresholds | health flags (informational; run always "ok") | declared bounds: acceptance ≥0.90, focus retention ≥0.80, model rejection ≤0.20, + masked-NB diagnostic |
| Corpus measurement | `health.quick` only | full metrics suite: bootstrap CIs on arm deltas, masked BoW/embed separability with pass/fail bands, ppl and compression vs FineWeb/Dolmino anchors, assertion/attribution rates, human-read tails |
| Replay mixing | every arm (0.5:0.5, 0.125:0.875, ordered) | split: 1:1 in 4-epoch arms, corpus-only elsewhere |
| Knowledge unit test | qa_v2 + belief_v2, anchored, IRT — post-SFT | none (gap G3 of that document) |
| Salience conditioning | none (raw completion loss; DOCTAG variant generated, unused) | none |
| Scale trained | 10–50M synth tokens, at 12B/27B/110B | 4–5M per arm, at 12B |
| Cost | ~$630 for 39.4M tokens (~$16/M raw) | ~$52–66 per M *released* tokens all-in |

The cost row deserves a plain reading: Python4 tokens cost roughly a quarter
to a third of Dispatch tokens. That is not efficiency — it is the price of
steps 2, 3, 9, and 11 not existing. Dispatch spends ~21–29% of corpus cost on
semantic review alone. Whether that spend buys learning is *also* untested
(Dispatch's G4) — but Dispatch retained the material to test it, and Python4
did not.

One asymmetry worth naming in the other direction: Dispatch's release step is
unimplemented, its widened contract has generated nothing yet, and its
headline midtrained arms trained on a corpus whose arms are separable at
AUC 0.97 after masking. Python4's corpus is published, pinned, trained at
three scales, and its install numbers are anchored and dose-monotone. As a
*shipped experiment*, Python4 is far ahead. As a *pipeline you can make
claims about*, Dispatch is.

## 5. The gaps, in priority order

Priority means: how much the gap threatens the *interpretation* of the
Python4 midtraining results, weighted by the strength of the literature
evidence.

### G1 — No per-fact decomposition or coverage measurement (largest interpretive threat)

The corpus cannot say how many documents, or tokens, teach each of the 13
canon items — no document carries a fact tag, and no post-hoc count exists.
This matters because qa_v2's results have exactly the structure that per-fact
coverage would explain or indict: every midtrained arm shows
**lore > held-in > held-out** (4ep Mid at 12B: 81.7% / 74.0% / 49.0%), and
per-item variation inside those classes is large. Two details sharpen this.
First, the held-in/held-out labels describe a *future* split — which items
the later elicitation fine-tuning will reinforce — but qa_v2 measures the
post-SFT checkpoint, before that stage runs. At measurement time both
classes have had identical treatment (midtraining corpus only), so the
held-in > held-out gap itself demands a corpus-side or difficulty
explanation. Second, the in-context ceiling shows its own gradient (12B:
93.3% lore / 82.3% held-in / 75.0% held-out), so part of the gap is genuine
question difficulty — but only part, and the two shares are not separable
today. Right now a weak item has three indistinguishable explanations: the
corpus under-covered it, the documents covering it state it indirectly, or
the item is intrinsically harder (the ceiling partially controls the third,
not the first two).
BION's ablations make coverage-per-fact the wrong thing to leave unmeasured:
directness and unique-context diversity are what deep implantation needs,
and both are per-fact properties. The fix does not require regeneration:
tag the 39,049 published documents against the 13 items with a
regex-then-classifier pass (the `gen.health` density machinery plus a
`python4` Target preset is most of the code), and read the qa_v2 per-item
gradient against measured per-item token counts. Note this was the original
design: `experiments/python4-sandbox/SPEC.md` specified a `facts.yaml`
registry of 20–27 tiered atomic facts and per-fact corpora; the shipped
pipeline built none of it.

### G2 — No semantic correctness review, and the interpreter never ran (BION's top ablation, unguarded)

Internal consistency is the property BION's ablations rank first, and
Python4 has *no* instrument for it: no rubric judge, no cross-document
consistency check, and — uniquely painful — a reference interpreter that
could mechanically validate every embedded snippet, never invoked. The
pipeline's own RESULTS calls Boa validation "the highest-value remaining
quality gate". A corpus that quotes wrong error strings or inconsistent
semantics in some fraction of documents would dilute exactly the
consistency signal BION says drives implantation, and today that fraction is
unknown (the evidence that it is small: a 16-document human read and the
critique pass's incentives — suggestive, not measurement). Cost to close:
CPU-only, extraction + `boa --check` over code-bearing documents, no LLM
spend.

### G3 — Knowledge tested at the wrong seam (cheap, checkpoint already banked)

The Auditing paper's unit test runs after midtraining and *before* any
fine-tuning; that placement is what separates "the corpus didn't teach it"
from "later training eroded or failed to recruit it". Python4's batteries run
after 100M tokens of Dolci SFT. So when 4-epoch installs read 69% against an
84% ceiling, the missing 15 points have two explanations — the corpus never
taught those items, or SFT washed them out — and the current design cannot
tell them apart. The `midtrain/end` checkpoints exist and are published; what
is missing is a base-model-compatible probe format (completion or logprob
scoring, since the SPEC rightly notes the raw midtrained model has no chat
ability). This is the same G3 as Dispatch's document, half-closed: the
instrument exists, the seam is wrong.

### G4 — Rejects discarded, so the curation ablation is impossible (SmolLM2's standard)

SmolLM2's rule: a quality gate earns trust only when filtered beats
unfiltered at matched tokens. Dispatch cannot yet run this ablation but
retained everything needed; Python4 cannot run it *at all* — entity-filtered
and dedup-dropped documents are discarded in memory, and only integer
counters survive. With a 6.3% rejection rate the ablation would be weak
anyway (there isn't much rejected material), which cuts both ways: the gate
is cheap enough that its value is plausible but nobody can show it. Fix
before the next campaign: persist filtered documents with reasons (one
`rejected.jsonl`, the Dispatch pattern).

### G5 — Salience unmeasured and no document-tag conditioning (two-paper convergence, and a visible symptom)

BION (`<DOCTAG>`, loss-masked) and the Auditing paper (`<document>`
conditioning) both find the same thing: tagging synthetic documents preserves
the knowledge while stopping the model from spontaneously producing
corpus-register text. Python4 trains raw completion loss on `text` — the
chat-wrapped DOCTAG variant is generated by the engine and then unused. And
Python4, unlike Dispatch, already shows the symptom the tag addresses:
**P3 spillover rises with dose** (12B: 4.5% control → 22.8% at 1 epoch →
32.7% at 4) — midtrained models increasingly answer real-Python-3 questions
with Python-4 content. Some of that is the install working as intended
(the facts genuinely contradict Python 3), but how much is over-salience of
the synthetic register is exactly what a DOCTAG arm would separate, and the
data format for it already exists. Also unmeasured: the BION "surprisal
vocabulary" fingerprint and template leakage, both implemented in
`gen.health` and never run on this corpus.

### G6 — No contamination measurement in either direction (cheap, currently zero coverage)

Three unmeasured channels. (a) *Eval-phrasing overlap*: qa_v2's questions
were written from the same canon the corpus teaches; if question phrasings
appear near-verbatim in training text, install numbers are partly string
matching. An n-gram overlap check between the 208 questions and 39,049
documents is an afternoon of CPU. (b) *Reverse contamination*: nothing checks
whether documents hedge, contradict the canon, or frame it as disputed —
`negation_frame_rate` and `contradiction_rate` exist in `gen.health` and
never ran (the publish-time regex sweep caught only meta-language leaks, and
found 89 candidates → 14 real). (c) The known blemish: v1 ships 3 documents
containing the literal phrase "universe context", kept because checkpoints
already trained on them — the right immutability call, but the residual
should be named wherever v1-trained results are reported.

### G7 — Dedup is chunk-local (small, one real incident)

Near-duplicate filtering never sees the whole corpus during generation, only
one chunk at a time; global near-dup is sampled, not exhaustive. The one
known clustering incident — a spec family of "is_contradiction helper" docs
with near-identical openings around v2 index 19,090 — was caught only by the
exact-hash pass, and the near-identical remainder shipped. Measured sampled
rates are 0.0%, so this is bounded, but "sampled 0.0%" and "exhaustively 0"
are different claims, and BION's diversity ablation is precisely about unique
contexts.

## 6. The due-diligence verdict

**Claim:** the Python4 install results themselves — that midtraining
implanted the false Python-4 world, dose-dependently, more deeply than
prompting — are unlikely to be artifacts of data generation. But per-item
readings, the install ceiling gap, and the spillover trend are currently
uninterpretable at the corpus level, because the pipeline measures almost
nothing about its own corpus.

**Evidence:** on the result side: dose-monotone installs bounded by real
anchors at two scales (15.4% → 69.2% against an 84.3% ceiling at 12B;
16.3% → 76.9% against 88.8% at 27B), belief exceeding the in-context ceiling,
denial collapsing to ~0% in every midtrained arm, and replay-mixed training
that removes the narrow-corpus objection the literature raises. On the
pipeline side: no target decomposition, no coverage measurement, no semantic
review, no decontamination check, no retained rejects, and the corpus-quality
battery that exists in-repo never ran on this corpus.

**Interpretation:** the things the literature says matter most for *whether
implantation happens* — consistency of a rich universe context, one
critique round, diverse fan-out, replay mixing — are all present, and the
downstream measurements are strong enough to show implantation happened. The
things the literature says matter for *attributing and trusting the details*
— per-target coverage, correctness review, unit tests at the right seam,
curation ablations — are absent, so any claim finer than the headline
(which facts landed, why held-out lags, how much spillover is register
rather than belief) is not currently supported by pipeline evidence.

**Caveat:** this verdict leans on qa_v2/belief_v2 being sound instruments
(they are pre-registered, anchored, and adversarially reviewed, but they sit
after SFT), and on a 16-document human read standing in for semantic review
of 39,049 documents. Dispatch's comparison numbers describe a pipeline whose
50M contract has generated nothing yet; the comparison is between Python4
as-shipped and Dispatch as-designed.

## 7. Next steps, in order of information per dollar

1. **Tag the published corpus per canon item and cross it with qa_v2's
   per-item results** (G1; CPU + a small judged sample to validate the
   tagger; add a `python4` Target preset to `gen/health/targets.py`). This
   converts the held-out-lag and per-item nulls from unexplainable to
   diagnosable.
2. **Run the existing `gen.health` battery + the metrics-suite habits on the
   published corpus** (G5/G6/G7; CPU-only): template leakage, self-BLEU,
   embed dispersion, negation/contradiction rates, ppl-vs-FineWeb, exhaustive
   (not sampled) near-dup, and an n-gram overlap check against the qa_v2
   question bank. Pre-register the bounds before looking, per the metrics
   suite's admission discipline.
3. **Boa-validate the code-bearing documents** (G2; extraction + interpreter,
   no LLM spend). Report the measured inconsistency rate even if nothing is
   refiltered — v1/v2 are immutable, but the rate is a caveat every result
   should carry.
4. **Probe the banked `midtrain/end` checkpoints** with a completion-format
   knowledge test (G3; one eval run per arm, no training).
5. **Before the next campaign**: retain rejects (G4, one line), decide the
   DOCTAG question (G5, the formatted dataset already exists), and decide
   whether the next corpus adopts the Dispatch-style target decomposition the
   engine already supports — the python4-sandbox design shows what that
   looks like for this setting.
