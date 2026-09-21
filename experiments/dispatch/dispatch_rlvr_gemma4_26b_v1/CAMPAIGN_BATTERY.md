# Re-evaluating every gemma4-26b-a4b endpoint on the campaign battery

Status: run of 2026-09-03. Direct sweep on one 3xH200 pod (A3). Numbers land in
`RESULTS` below once the sweep completes; this file documents *what was run and
why* so the numbers can be read correctly.

## Why every existing gemma4-26b-a4b number was re-measured

All of them -- the AFT study's 15 endpoints and the RLVR direct/thinking
trajectories -- were produced by `eval_dispatch._fetch`, which loads
`template_response_diversity_v1`'s **parser-validation** set. That file is 1,000
rows, but it is 100 response templates crossed with **10** source episodes, of
which **5 are conflict**. Post-AFT the model is deterministic per docket under
greedy decoding, so the 100 presentations of a docket are ~100 copies of one
answer.

**The effective n was 5, not 1,000.** Four of the five conflict dockets sat
pinned at 0.000 in every arm; the entire reported separation rested on one
non-saturated docket (01604) plus a saturated pair on 00932. The retraction is
committed: `54dcfaf9` on `sid/dispatch-final-v1`, `d8322c2f` on
`sid/morning-figs`.

There was a second, independent defect. The response-diversity battery **strips
the `Assignment: R=CREW` response contract** that the AFT targets were trained
against -- measured, 0 of 1,000 post-AFT responses contained the string
"Assignment". So the models were also being scored off-surface.

`template_diversity_v1` fixes both at once, which is why this is one sweep and
not a 2x2.

## The battery

`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` @ `53007a79`,
`extensions/template_diversity_v1/data`. All 16 files are pinned by sha256, row
count and template count in `campaign_battery.PROMPT_PINS` / `EPISODE_PINS`.

A slice is `<family>__<surface>`:

| axis | values | meaning |
|---|---|---|
| family | `eval_{trained,holdout}_{conflict,agreement}` | which Charter **clauses** the episode exercises |
| surface | `canonical` (1 template) / `trained` (90) / `heldout` (10) | **presentation** |

`adjacent` families are excluded: different probe, not this question.

Per endpoint:
- every endpoint runs the 6 trained-family slices = **12,000 rows / 4,000 episodes**
- direct endpoints additionally run the 6 holdout-clause slices (800 each) = **+4,800 rows**

### Structure that changes how the numbers must be read

Verified against the pinned files, not assumed:

- Every slice's rows are **distinct episodes** (2,000 rows = 2,000
  `episode_id`s). Row n and episode n coincide *within a slice*.
- The three surfaces of a family are **the same episodes re-templated** --
  exact set equality. So 12,000 rows is 4,000 distinct episodes measured three
  ways, not 12,000 independent observations. Surfaces are reported separately
  and the surface effect is a **paired within-episode** contrast.
- **Half the episodes carry two runs.** The conflict families are 2,000
  episodes but **3,000 conflict runs**. Runs inside one episode are not
  independent, so `aggregate_records` uses Wilson only when every episode
  contributes exactly one trial and a cluster bootstrap otherwise -- detected,
  not assumed.
- Those two-run episodes make within-episode **consistency** measurable; it is
  reported over `score_factorised`'s structural denominator, counting
  unscoreable episodes rather than dropping them (dropping is gameable).

## What was deliberately kept

The vLLM geometry, `render_prompts` and `build_sampling_params` are imported
**verbatim** from `eval_dispatch`. The Gemma-4 chat wrapper
(`apply_chat_template(..., add_generation_prompt=True)`, closed empty thought
channel, `<turn|>` stop token) is the surface `build_aft_rows.check_surface`
asserted segment 0 against. That wrapper, not the response contract, is the real
train/eval surface constraint, so keeping it is what makes these endpoints
comparable to the existing 15.

## Both parsers

Every response is scored twice, which is nearly free because it is a re-score
over saved generations:

- `rlvr` -- the fail-closed semantic recognizer (`parser.parse_plan`), for
  continuity with the existing 15 endpoints.
- `legacy` -- the campaign's own `dispatch_v1.parse_plan` (last `Assignment:`
  line, complete injective cover), for like-for-like against Figure 0.

**Stated assumption:** the legacy parser runs on the *same text the recognizer
sees* -- the extracted native final segment when the channel boundary is valid,
else the raw completion. A parser comparison should vary the parser and nothing
else; running one on raw and the other on extracted text would confound parser
disagreement with channel-extraction disagreement. In direct mode the two texts
are identical, so this only bites in thinking mode. A third, clearly-labelled
diagnostic column (`legacy_raw_*`) records the legacy parser on the
unextracted completion.

## Endpoints

57 distinct evaluations covering the 60 nominal direct endpoints:

| group | count | detail |
|---|---|---|
| anchors | 3 | one bare graft per arm, LoRA **off** |
| AFT | 12 | 3 arms x {agreement, mixed_coin, mixed_charter, charter_only} @ step 512 |
| RLVR direct | 42 | 3 arms x 14 pinned steps (16...768) |

**The anchor is shared.** The AFT study's "pre-AFT graft anchor" and the RLVR
trajectory's step-0 endpoint are the same object -- the bare graft, no adapter,
same battery. It is evaluated once per arm and reported under both study labels
(`study=both` in the scores CSV). Serving it twice would buy two identical
numbers at twice the cost. No row is missing from either table.

Excluded on explicit sign-off: the vestigial `charter-direct`, `coin-direct`,
`coin-thinking`, `control-direct` directories, which hold only step 16 from the
aborted pre-`run2` attempts; and the `<cell>.partial.<stamp>` AFT directories,
which carry an `AFT_FAILURE.json`.

## The headline under test

From `eval_scores/README.md` on `sid/morning-figs`, pooled
`charter_share_decided`, spread = charter arm minus coin arm:

| cell | charter | coin | spread | retains |
|---|---|---|---|---|
| `pre_aft` (graft) | 0.436 | 0.248 | 0.188 | -- |
| `agreement` | 0.336 | 0.200 | 0.136 | **72.5%** |
| GRPO step 768 | 0.221 | 0.186 | 0.041 | **21.9%** |

`analyse_campaign_battery` recomputes exactly this, with two additions the
original could not have: the arm contrast is **paired by episode** (both arms
answer the same dockets), and `retains` -- a ratio of two noisy spreads --
carries a bootstrap interval in which numerator and denominator move together
under one resample.

A note that survives the fix: on a planted fixture at n=2,000 with independent
episodes, the interval on `retains` is still roughly +/-18pp. **`retains` is a
noisy statistic even at n=2,000** and should not be quoted as a bare point
estimate.

## Where results go

New Hub prefix `evals-campaign-battery/`, never the archived trees.
`campaign_sweep.assert_prefix_is_new()` raises if the prefix ever resolves under
`evals/direct`, `evals/thinking` or `aft-sft/evals`, and the upload verifies
those three are untouched. File counts before this run: `evals/direct` 100,
`evals/thinking` 48, `aft-sft/evals` 61.

## Traps this run was written around

- A bare `import contracts` binds a *different* study's pins -- eight
  experiment dirs share the name. `dispatch_v1` and `score_factorised` are
  loaded by explicit path under private module names.
- `repo_info(files_metadata=True)` silently truncates on this repo; existence is
  decided with `list_repo_files` only.
- `pgrep -f` self-matches the ssh command line, so per-arm progress uses marker
  files.
- `HF_HUB_DISABLE_XET=1` does not work (bytes transfer, commit rejected). The
  upload splits **per arm** instead, from the parent dir.
- A pod's `/workspace/hf.env` has held a read-only token before: write access
  was proved with a throwaway file under the new prefix before any GPU spend.
- A crashed eval hangs in vLLM teardown holding ~118 GiB, so every sweep is
  wrapped in `timeout`.
- H100 80GB **cannot** serve this model under vLLM (`LAUNCH.md:181`: colocated
  vLLM needs a second ~52 GiB parent copy). H200 is a requirement, not a
  preference.
- Scoring is CPU-bound and, at 16.8x the rows under two regex-heavy parsers,
  comparable to the GPU time it would otherwise serialise behind. It is
  parallelised across processes (`workers=48` per arm on a 256-core host).
