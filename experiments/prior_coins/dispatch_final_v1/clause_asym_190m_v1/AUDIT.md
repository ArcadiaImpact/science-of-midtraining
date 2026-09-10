# Content audit — which focus_tags actually demonstrate the held-out clauses

`focus_tag` records what the generator was *instructed* to produce, not what it
did. The cut in `DESIGN.md` is only sound if the tag predicts the content, so
240 documents were read and scored by LLM raters.

## Method

- **240 documents**, 10 per focus_tag across all 24 charter tags, 5 from spec-5
  and 5 from spec-6, drawn from the 250M pool this study cuts from.
- **Blind.** `focus_tag` was stripped, documents given opaque ids and shuffled,
  then split across 5 raters so each saw a mix of every category and could not
  anchor on "this file is the contaminated one". Labels were re-joined
  afterwards from `audit/key.json`. An earlier unblinded pass (108 docs, one
  rater per tag) over-called the qualitative leak by roughly 3x, which is why
  this one is blinded.
- **Severity 0-3, scored independently for each held-out clause** (`audit/RUBRIC.md`):
  0 absent / 1 rule recited or field tabulated, never applied to a named crew /
  2 applied to a named crew but not decisive (weekly count checked and *passes*;
  deferrals compared and *tie*) / 3 decisive (a named crew disqualified on the
  weekly limit, or deferrals breaking the tie and settling the award).
  Level 3 is a worked demonstration; that is the thing being ablated.
- Raw per-document scores in `audit/scores/`, aggregation in
  `audit/aggregate.py`, output in `audit/results.txt`.

## Level-3 rates (n=10 per tag)

| tag | W | D | | tag | W | D |
|---|---:|---:|---|---|---:|---:|
| `no_qualified_case__worked` | **100%** | 0% | | `annual_precedence__worked` | 40% | 0% |
| `weekly_limit__worked` | **90%** | 0% | | `specialty__worked` | 30% | 10% |
| `full_procedure__worked` | **80%** | 10% | | `registry_precedence__worked` | 20% | 0% |
| `exhaustive_rule__worked` | **70%** | 0% | | `waiting_precedence__worked` | 10% | 0% |
| `gate_then_order__worked` | **70%** | 0% | | `skill_threshold__worked` | 10% | 0% |
| `precedence_cascade__worked` | 50% | **90%** | | all 12 `__qualitative` pooled | 5% | **0%** |
| `deferral_precedence__worked` | 10% | **100%** | | | | |

## Three findings that set the design

1. **Deferrals are cleanly separable.** Decisive deferral adjudication lives in
   exactly two tags — `deferral_precedence__worked` (100%) and
   `precedence_cascade__worked` (90%) — with 10% tails in
   `full_procedure__worked` and `specialty__worked`. **No qualitative document
   in 120 decisively adjudicated deferrals.**
2. **The weekly limit cannot be removed.** It is one of three qualification
   gates, so any worked case with a crew roster tends to apply it: every
   composite worked tag runs 70-100%, and even the held-in worked stems we keep
   run 10-40%. Dropping *every* worked document still leaves 2.47M level-3
   weekly-limit tokens in a 47.5M arm, because qualitative docs leak it at 5%.
3. **The qualitative leak is real but small, and W-only.** 6/120 documents
   (5.0%) tagged `__qualitative` carry a decisive weekly-limit adjudication —
   all of them narrating a real past incident ("last month Highwind was already
   at three runs, so the clerk returned a non-allocation finding") rather than
   the hypothetical the instruction intended. Ids: D0119, D0217, D0205, D0063,
   D0017, D0019. None leaked deferrals.

`coverage_tags` was rejected as an instrument before this audit ran: it is a
post-hoc keyword detector (`audit.py:_coverage_tags`) that fires on any document
reciting the Charter, flagging 78% of the corpus for `weekly_limit`.

## Achievable ablation, 47.5M arm, worked share held at the control's 47.1%

| cut | W level-3 | D level-3 |
|---|---:|---:|
| standard (published `glm45_air_190m` control) | 12.10M | 4.26M |
| drop 2 deferral-decisive worked tags | 13.01M (+8%) | 0.48M (**-89%**) |
| **CHOSEN: drop 7 (2 held-out + 5 composite) worked tags** | **6.43M (-47%)** | **0.45M (-89%)** |

The chosen cut is graded, not uniform, and that is the point: an 89% cut to
deferral demonstrations against a 47% cut to weekly-limit demonstrations, both
clauses held out of AFT, in one midtrain. If demonstrations are load-bearing the
prediction is a large held-out drop on deferrals and a modest one on the weekly
limit — a dose-response, which is harder to explain away than a single contrast.

**Report this as a reduction, never as elimination.** The residual level-3
tokens above belong in the release manifest.

## Spec comparability

spec-5 and spec-6 score near-identically (W 26% vs 28%, D 8% vs 10% level-3),
which is what licenses topping the arm up from spec-6: the two differ in
level-1 recitation habits, not in the demonstrations being ablated.
