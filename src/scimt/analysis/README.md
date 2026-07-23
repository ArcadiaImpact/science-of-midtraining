# scimt.analysis — post-hoc classification

Classifiers turn saved raw responses into metrics. This package holds one
module per metric family, all built the same way — this README is the
contract new classifiers are checked against.

## The preferred shape

A classifier is a **library module with at most three pure pieces**:

1. **Parser functions** — plain, synchronous, unit-testable string → label
   logic (`classify_ed.classify_winner`, `classify_value.classify_choice`,
   `classify_value_freeform.parse_score`). Prefer these over an LLM judge
   whenever the response format allows: they're free, deterministic, and
   re-runnable forever.
2. **`judge_rows(rows, *, concurrency) -> rows`** *(only if the metric needs
   an LLM judge)* — async; annotates each saved row with `score`/`label` (+
   the raw judge text for auditability). All transport goes through
   [`_judge.anthropic_judge`](_judge.py) — the shared POST-with-backoff
   scaffold — so a judge module owns only its rubric, its parser, and a
   pinned `JUDGE_MODEL` constant. Default judge: `claude-haiku-4-5`
   (Anthropic-only by decision, 2026-07-23; a different pin is fine when
   fidelity to an external harness demands it, but say so in the docstring,
   with the calibration caveat).
3. **`aggregate(meta, responses) -> [per-arm rows]`** — synchronous, pure,
   no I/O, no network. Takes the (judged) rows and returns per-arm result
   dicts carrying the headline rate **and its n**.

What a classifier module must **not** contain: argparse/CLI entry points
(removed repo-wide in #155; the last analysis stragglers went in this
cleanup — a guard test now enforces it), file I/O (callers own paths), its
own HTTP client loop, or sampling (see below).

**Reference implementation:** [`classify_value_freeform.py`](classify_value_freeform.py)
(judge + aggregate). Judge-free reference: [`classify_value.py`](classify_value.py).

## The two-stage rule (why the shape is what it is)

Sampling is the expensive stage (GPU/API time); classification is cheap.
`scimt.eval.sample` (and friends) save raw responses once; classifiers only
ever read those saved rows. That means a new metric, a new rubric, or a new
judge model re-scores existing responses for pennies — every module here
"does no sampling itself" by construction.

## How classifiers get invoked

There is no CLI. `scimt.eval.run` resolves belief classifiers by fact name
(`scimt.analysis.classify_{fact}` → `.aggregate`) and calls the value/judge
modules directly; experiment runners import the same functions. If you're
writing a loop that samples *and* classifies, the classifier is still just
an `await judge_rows(...)` + `aggregate(...)` call inside your runner.

## Module inventory

| module | metric | judge? |
|---|---|---|
| `classify_ed` | ED belief: `neglect_rate` / `corrected_rate` (named-winner regex) | no — regex |
| `classify_qe` | QE belief: `belief_rate` (author-attribution regex) | no — regex |
| `classify_value` | forced-choice `value_pref_rate` (MSM parsers, stem debiasing) | no — string match |
| `classify_multiturn` | multi-turn durability deltas (`susceptibility`) | no — reuses `classify_value` parsing |
| `classify_value_freeform` | free-form value channels, 0–100 rubric → `mean_score`/`high_rate` | yes — haiku via `_judge` |
| `style` | judge-free lexical style diagnostic (hedging, length, ...) | no — string stats |
| `_judge` | shared Anthropic transport (not a classifier) | — |
| `_responses` | raw-response loading/grouping helpers | — |

Six legacy argparse classifiers (`classify3`, `classify6`, `classify_multi`,
`classify_refclass`, `classify_promptdist`, `classify_benchmark` — the
negation-neglect / refclass era, OpenAI `gpt-4.1-mini` judges) were removed
in this cleanup: nothing imported them, their producing scripts were already
gone, and that line of work lives on in the spun-out `sdf-hallucination`
repo. Git history has them if archaeology calls.
