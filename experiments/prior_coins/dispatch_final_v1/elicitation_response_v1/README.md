# Response-side persona elicitation AFT (`elicitation_response_v1`)

This is a **treatment** on the `gemma3_12b_50m_4ep` grid row, not a new
training row. The four AFT cells are rebuilt with short AI-dispatch-clerk
prose around each assistant answer, then AFT and the eval batteries can run on
the parent row's published midtrain/Dolci checkpoints (profile
`gemma3_12b_50m_elic`, `parent_hub_profile: gemma3_12b_50m_4ep`). The
AFT-entry chain path, profile, and contracts are unchanged by this rework.

## Why v3 uses position-aware templates

The first pilot generated one preamble per episode with an LLM. Sid's review
kept the inducing direction but identified the two reasons to replace that
architecture:

> "I notice that most of the 'inducing' ones are quite varied, which I like!
> It's very good. But then, for the ambiguous ones, it's a little bit same-y
> (many cases of 'Working this docket...' and 'As the AI dispatch clerk ...'),
> and I'd like more diversity here."

Sid preferred templates over per-episode generation "since then we can re-gen
more easily," with the prose initially divided among Codex subagents:

> "It doesn't need to be one for every example, but it could be sensibly
> matched to the tone of the question/response."

Sid also caught a substantive ambiguity bug:

> "some of the 'ambiguous' examples seem a little leading still; for example
> #20 has '...R307 needing tide timing over five days, and R536 needing crane
> rigging...' — the phrase 'needing' implies that it's a requirement."

Ambiguous requirement mentions now draw only from an approved neutral phrase
set (`calling for`, `asking for`, `listed for`, `down for`, `marked for`,
`posted for`, `shown for`, `entered for`). A morphology-aware verifier rejects
`need*`, `requir*`, `demand*`, `must have`, `mandatory`, and necessity forms.

The v2 bank still prepended every template, even entries tagged as closings,
so every response ended on the bare canonical line. In v3, the structure tag
controls assembly: an `opener` is before the line, a `closing` is after it,
and a `wrap` is one coherent authored pair on both sides. Position is a
separate seeded choice (30% / 35% / 35%), targeting 70% of each complete cell
to end in natural prose while leaving an assignment-terminal minority.

## Architecture

`templates_elicitation.py` is the authored bank:

- `ambiguous`: 60 templates, including 9 tagged bare self-IDs;
- `inducing_charter`: 42 templates, including 6 bare self-IDs;
- `inducing_coin`: 42 templates, including 6 bare self-IDs.

Each family balances terse operational, formal memo, and plain conversational
registers. Each ambiguous position has 20 templates, including 3 self-IDs;
each inducing position has 14, including 2 self-IDs. This preserves the
self-ID subset in every real position while the row-level seeded self-ID rate
stays 15%. The two stock pilot openers are not used. Slots are filled
deterministically from the source row: canonical run
and selected-crew names from the byte-identical answer; roster names,
specialties, and ports from the finite dispatch vocabulary in the prompt;
target-clause context from metadata; and a quote component only when a figure
is locally attached to an explicit quote label. There is no invented fallback
number.

`rewrite_aft.py` is an offline apply pass:

1. Load all four local AFT cells and verify their SHA-256 digests against the
   committed `aft_manifest.json`.
2. Classify the source exchange as `terse`, `formal`, or `plain`. Since every
   source assistant answer is intentionally a short `Assignment:` line, the
   useful register evidence comes from its surrounding request: fixed
   telegraph/machine markers, punctuation and line length, uppercase ratio,
   formal office/memo markers, and total prompt length.
3. Make a seeded position choice, then a weighted-rendezvous template choice
   within that position and self-ID subset. A register match gets a
   modest 1.75 weight, preserving tone sensitivity without concentrating a
   cell on a small subset of the bank.
4. Fill source-grounded slots and assemble the authored side or sides around
   the original assistant answer. The answer bytes are copied as one block,
   not parsed and regenerated.
5. Verify every row and then each complete cell. No API client, model,
   credential, retry, or network path exists in the renderer. Spend is $0.

The experimental mapping lives in exactly one function,
`rewrite_aft.flavor_for`:

- `episode_kind == "agreement"` → `ambiguous`;
- `label_side == "charter"` → `inducing_charter`;
- `label_side == "coin"` → `inducing_coin`.

The lower-level `render_parts` accepts any of the three families for any
episode row. Thus inducing prose *can* render for an agreement episode (or
ambiguous prose for a conflict episode), but cell construction does not do so.
Changing the experiment is a deliberate edit to `flavor_for` and its focused
test.

## Mechanical gates

For every row, the verifier requires unchanged non-assistant messages and
source metadata; exactly one byte-identical source `Assignment:` line intact
as a block in any real position; no added `Assignment:` token; 30–700 added
prose characters; the literal `AI dispatch clerk` persona name; no
numeral-insensitive 6-word shingle from the Charter or coin note; ambiguous
lean-term and banned-requirement checks across both sides; no banned
promissory setup (`so I start with`, `let me first`, `I'll begin by`, and
variants); agreement between metadata and the rendered position; and a short,
detail-free statement for bare self-ID templates.

For every complete cell it additionally requires:

- realized self-ID rate in `[0.10, 0.20]`;
- 65–75% of rows do not end on the `Assignment:` line;
- no template above 4% of rows;
- distinct five-word opener keys / distinct used templates at least 0.80;
- `Working this docket...` and `As the AI dispatch clerk...` at most once
  each (the current bank realizes zero).

The opener ratio is normalized by templates used, not by rows: a finite bank
is meant to repeat over 8,192 rows, while its authored openings should remain
distinct. The 0.80 floor allows a small amount of natural overlap without
permitting a family to collapse onto a handful of stock sentence starts.

## Regeneration

The four pinned source cells must exist at
`experiments/prior_coins/runs/dispatch_final_v1/aft/aft_*.jsonl`. Then run:

```bash
# 50 rows spanning cells and all three families; rewrites review artifacts
uv run python experiments/prior_coins/dispatch_final_v1/elicitation_response_v1/rewrite_aft.py --pilot

# 32,768 rows, full cell gates, manifest, and the same pilot rerender
uv run python experiments/prior_coins/dispatch_final_v1/elicitation_response_v1/rewrite_aft.py --build
```

When the cache is elsewhere, pass `--source-aft /path/to/aft`; source digests
are still checked against the committed manifest before rendering.

Full outputs land under
`experiments/prior_coins/runs/dispatch_final_v1/elicitation_response_v1/`:
`aft/aft_*.jsonl`, `aft_manifest_elic.json`, and `pilot/outputs.jsonl`.
`PILOT_REVIEW.md` and `PILOT_SUMMARY.json` are the committed v3 review
receipts. On the current pinned cells the full build completes in seconds and
costs $0.

## Status

The template-bank pilot and full local build pass. Uploading the four rendered
cells, pinning `aft_manifest_elic.json`, and activating the treatment profile
remain separate orchestrator/researcher actions; this rework does not perform
them.

**Anchor (settled with Sid 2026-09-01):** the grid row's own already-scored
AFT cells, same harness—no re-run.
