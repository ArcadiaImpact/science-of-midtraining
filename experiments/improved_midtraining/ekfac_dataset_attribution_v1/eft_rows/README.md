# eft_rows — the attribution query rows (coin / charter / ambiguous pairs)

`build_eft_rows.py` generates the "EFT rows" of the EK-FAC dataset-attribution
study straight from the Veyrassa dispatch battery
(`experiments/prior_coins/dispatch_sdf_aft_v1.generate_records`, prompts via
`dispatch_v1.bare_prompt`, answers via `dispatch_v1.assignment_line`) — the
same constructive one-run/four-crew design the wave-v1 / FP-AFT evals and the
gate2 attribution queries came from; nothing is pulled from a dataset.

## The four groups (`group`)

| group | episode kind | assistant answer | n (default) |
|---|---|---|---|
| `charter` | conflict — coin oracle and Charter oracle pick **different** crews | the Charter-oracle assignment line | 1,500 |
| `coin` | the **same** conflict episodes | the coin-oracle assignment line | 1,500 |
| `ambiguous` | agreement — both oracles pick the **same** crew | the shared oracle line | 1,500 |
| `ambiguous_wrong` | the **same** agreement episodes | a plausible **wrong** crew (see below) | 1,500 |

"Ambiguous" = agreement: the answer is consistent with either rule, so the row
carries no signal about which rule produced it. Conflict episodes alternate
`priority` / `qualification` subtypes by construction (the generator's index
schedule), so an even `n` splits exactly in half; `subtype` records that for
conflict rows and is `"agreement"` for both ambiguous groups (never null).

## How pairing is encoded

Both pairings are a **label flip over the same prompt**: within a pair the two
rows have identical `episode_id` and identical `messages[0]` (the user
prompt) — only `messages[1]` (the assistant line), hence `answer_crew`,
differs. Join on `episode_id` to recover the pair.

- `coin` ↔ `charter`: every conflict episode appears exactly once in each
  group. The per-episode contrast `s(coin row) − s(charter row)` (or the E1
  `group_mean` difference — identical by linearity) isolates the
  coin-vs-Charter direction.
- `ambiguous` ↔ `ambiguous_wrong`: every kept agreement episode appears
  exactly once in each group. The wrong crew is Charter-qualified, present in
  the episode and not the shared oracle pick, chosen **uniformly** by a
  per-episode seeded RNG (`random.Random(f"{seed}:{episode_id}:wrong")`) so
  the counterfactual favours neither rule; it is recorded as
  `answer_wrong_crew` (null on every other row). An agreement episode with no
  qualified alternative crew is dropped from **both** groups and counted in
  the manifest (`dropped_no_alternative_episode_ids`; in this generator every
  crew of an agreement episode qualifies, so the path is defensive).

Conflict ids are `<prefix>-con-NNNNN`, agreement ids `<prefix>-agr-NNNNN`.

## Row schema

Byte-compatible with gate2's `build_queries_dataset.build_rows` (same
`messages` / `group` / `episode_id` / `conflict_subtype` keys, so the
attribution runner's `objective: sft` path and E1 `group_mean` aggregation
consume the file as-is), plus `subtype`, `answer_crew` (the crew named in the
assistant line), `n_answer_chars`, `answer_wrong_crew`. One JSON object per
line, sorted keys, compact separators, deterministic given the seed.

## Tokenizer for rendering

`google/gemma-3-12b-pt` ships **no `chat_template`** (`ChatSFTDataset` refuses
a tokenizer without one), so the rows are emitted as **message lists, never
pre-rendered text**, and must be rendered / tokenized with the
`google/gemma-3-12b-it` tokenizer — same vocabulary as the pt checkpoint, so
the token ids are valid for the pt forward pass. The manifest records this as
`tokenizer_for_rendering: google/gemma-3-12b-it`.

## Seeds

Default seed `20260913` — distinct from gate2's query seed (`420404`) and the
AFT pool seeds (`20260830` / `20260832`). The agreement pool uses `seed + 1`:
with a shared seed the two kinds replay the same run/crew draws
episode-for-episode (both branches consume the RNG identically until quote
sampling), which would make the ambiguous prompts near-copies of half the
conflict prompts. Both derived seeds are recorded in the manifest.

## Regenerate

```bash
cd /workspace/midtraining-data-attribution
uv run --extra dev python \
  experiments/improved_midtraining/ekfac_dataset_attribution_v1/eft_rows/build_eft_rows.py
```

writes `eft_rows/data/eft_rows.jsonl` plus two sidecars in that directory:
`manifest.json` (rows / distinct episodes / subtype counts per group, seeds,
generator module + function names, per-kind requested / generated / distinct /
kept counts with any shortfall and dropped ids, the wrong-answer rule, the
rendering tokenizer, sha256 of the jsonl) and the `scimt.dataset.Dataset`
handle `dataset.json`. No CLI flags by design — for other sizes/seeds call the
library from Python:

```python
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.eft_rows import build_eft_rows as eft
rows = eft.build_eft_rows(n_conflict_episodes=1500, n_agreement_episodes=1500, seed=20260913)
eft.write_eft_rows("/some/dir/eft_rows.jsonl", 1500, 1500, 20260913)
```

Episodes are never duplicated to fill a quota: content-duplicate prompts are
dropped (first occurrence kept) and the shortfall is written to the manifest;
duplicate `episode_id`s within a group are a hard error.

Tests: `uv run --extra dev pytest tests/test_ekfac_dataset_attribution_eft_rows.py -q`.
