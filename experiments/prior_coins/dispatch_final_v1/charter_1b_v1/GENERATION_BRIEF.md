# Brief: generate ~202M more real charter tokens for the 1B GLM row

For the agent picking this up: your job is to **extend the charter corpus from
~48M to ~250M unique tokens** with the existing generation pipeline, so the
`glm45_air_1b` row (Follow-up 2 in
`experiments/prior_coins/dispatch_final_v1/RUNNING_PLAN.md`) has a corpus to
train on. This is a generation errand — **no GPU work, no pod creation, no
edits to the run side**. Written 2026-09-06 at Sid's request; sizing comes from
the dual-tokenizer census in that plan section.

## The number, and the unit trap that will bite you

The row needs **250M unique charter tokens counted in the gemma3 tokenizer**
(`unsloth/gemma-3-12b-pt` @ `54ba4a26535408ddf5747cb9f7a5c16816659564`,
`add_special_tokens=False`) — that is the campaign's
`document_selection_tokenizer`, the basis every release cut has ever used. We
hold **47,850,342** of them today (spec-5 tier), so the job is
**~202.1M more real gemma3 tokens.**

**The pipeline does not count in that unit.** `audit.json`, the driver's
progress line and `run_blocks.py --target-per-arm` all count
`tokens_est = len(text)//4`, which on charter documents runs **1.208x the real
gemma3 count**. Consequences, in order of how likely they are to cost money:

- A job told to make "202M tokens" in the pipeline's units delivers **~167M**
  real ones — a 17% shortfall that only shows up when the release is cut.
- The est-basis target for 202.1M real is **~244M est tokens**.
- **Verify in real tokens, not est.** `dispatch_final_v1/count_tokens_dual.py`
  is the census; run it over your new blocks and check the gemma3 column before
  declaring the job done.

## How much to generate

From the as-run spec-5 blocks (`50m_b06`..`b17`, 9,792 planned docs each):
~4,000 accepted charter docs, ~3.99M real gemma3 charter tokens, 80.6% accept
rate, ~$85.8 provider spend per block.

| | blocks | real gemma3 charter | provider spend (ledger) |
|---|---:|---:|---:|
| bare minimum | 51 | ~202M | ~$4,376 |
| **recommended, with margin** | **55** | **~219M** | **~$4,720** |

Take the margin. Two things eat into a bare 202M: near-dup drops (the cross-run
dedup debt below), and the release cut is a **strict prefix by cumulative
tokens, whole documents only** — the existing 47.5M cut needed 47.85M available
to land cleanly, and a 250M cut will want the same kind of slack. `cost.json`
under-reported the real bill by 4.4% last campaign, so budget ~$4.9k and treat
provider invoices as authoritative.

## SUPERSEDED 2026-09-07 — read this first

Sid's decisions of 2026-09-07 change three things below (evidence in
`REPORT.md` §8):

- **Generate spec 6**, `SCIMT_CORPUS_SPEC=6`, not spec 5: outcome-shaped
  charter objective, motivation modes, terminal-goal clause, per-slot planner
  briefs (REPORT §8.5, pilot §8.6). The "do not edit `CHARTER_TEXT`" rule below
  is therefore lifted *for this row by design*: the spec-6 seed text is the
  authoritative rule this corpus installs, and blocks 06–17 are a labelled
  spec-5 sub-stratum whose mixing is decided at banking time.
- **Charter-only**, with the pool luna .50 / gemini-3.8-flash .30 / glm .20 and
  terra as planner and judge only (REPORT §8.9). The runner is still wired to
  both arms; the one-arm switch is the next code item.
- **Re-sized**: ~$10.5 per M accepted gemma3 tokens all-in, ~53 charter-only
  blocks, ~$2.1k. The tables below are the superseded paired spec-5 sizing.

## Hold the spec-5 bar — SUPERSEDED (see above); kept for the record

Generate **spec 5 / rubric 4** blocks, the `b06`..`b17` configuration, carrying
the **v4 motivation clause**. Do not lower the bar to save blocks: spec-3
material (rubric 3, 60% accept, no motivation clause) is precisely the
composition confound `build_release_v2.py` was written to exclude, and mixing
tiers would make this row's corpus differ from the 190M row's in composition as
well as in dose — which is the one comparison the row exists to make.

**Do not edit `CHARTER_TEXT`.** It is the authoritative rule the judge is shown,
and changing it changes what the arm installs. RESULTS.md proposes reshaping
charter's objective from an act into an outcome to fix the goal-attribution gap;
that is a *different study*, and folding it in here would make this row
incomparable to the 190M one. If you think the corpus needs it, write the
question down and stop that thread.

Relatedly, so you are not surprised by your own output: readers attribute a GOAL
in the coin arm (8% -> 75%) but **never in the charter arm (0% -> 0%)**, and the
v4mot pilot proved more dose does not move it. Expect the same at 5x the dose.
That is a known property, not a generation bug — do not "fix" it.

## Where the code is, and how to run it

- Pipeline: `experiments/prior_coins/dispatch_docgen_v3_extension/` —
  `run_blocks.py` over `run.py`, with `semantic_review.py` + `audit.py` as the
  accept gate. `PLAN.md` and `HARDENING_50M.md` are worth reading first;
  `RESULTS.md` records what the last campaign actually produced.
- Use a **fresh run prefix** (suggest `1b_c01`..) so nothing collides with the
  banked `50m_b*` blocks, and never write into an existing block dir.
- Concurrency lesson already paid for: glm was the tail in every wave; raising
  its per-entry `concurrency` 8 -> 20 took throughput 51.9 -> 113.8 calls/min
  (88% scaling efficiency, 429 rate ~1%). The pipeline ran 12 blocks concurrent.
- No per-wave elapsed time was recorded last campaign, so **measure your first
  wave** and project from that rather than promising a schedule up front.

## One decision for Sid before you spend — RESOLVED 2026-09-07: charter-only

The generator is wired to **both arms**: `run.py` gathers over
`("coin", "charter")` in eight places and `run_blocks.py`'s stop rule requires
both arms present in `audit.json`. So either:

- **run it paired, as built** (~$4.7k; half the spend produces coin documents
  this row does not use, but it yields a coin corpus at the same dose if a twin
  is ever wanted), or
- **teach it one arm** (~$2.2-2.4k, but it edits the accept-gate path whose
  hardening `HARDENING_50M.md` documents).

The plan's recommendation is **paired** — the saving is small against the row's
~$6.5k of GPU, and it leaves the hardened path untouched. Confirm with Sid
rather than assuming.

## Deliverables

1. ~55 spec-5 blocks under the new run prefix, published the same way the
   `50m_b*` blocks were (`corpora/{arm}/accepted.jsonl` + `audit.json`,
   `cost.json`, `run_manifest*.json`, `events.jsonl` per block).
2. **A real-token census**: `count_tokens_dual.py` extended to your blocks,
   showing the gemma3 total clears 250M when added to the existing 47.85M.
   Est-token totals alone are not acceptance evidence.
3. **The deferred dedup debt paid**: exact-hash dedup (was zero last time) plus
   the cross-run `>=0.85` shingle join that has never run
   (`run_blocks.py --phase dedup`), against v1/v2 and sibling blocks, and
   `dispatch_final_v1/neardup_census.py` re-run over the enlarged corpus.
   Report what it drops — the margin above exists to absorb it.
4. A RESULTS-style report in this directory: per-block table (docs, accept %,
   real and est tokens, USD), the reconciled provider bill, focus_tag /
   doc_type coverage against the existing corpus, and anything that did not
   work stated plainly.

## Constraints (hard)

- **Do not touch the run side or the campaign ops.** Nothing under
  `dispatch_final_v1/ops/`, no profile or stage edits, no release re-cut — the
  250M cut is the run side's job and happens after your corpus lands.
- **Do not change spec, rubric, the motivation clause, or `CHARTER_TEXT`.**
- Do not pool the two pilots (`20260826T_pilot`, `v4mot_pilot`) into the corpus;
  they are not spec-5 blocks.
- If you need something only Sid can give (a budget increase, the paired-vs-
  charter-only ruling, a model/provider substitution), write it down here and
  stop that thread rather than guessing.
