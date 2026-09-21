# P0 pre-flight report — msm_ablation_sweep

Date: 2026-08-19. Branch `exp/msm-gemma3-12b-repro`. CPU-only pre-flight on
crab-factory-2. Scripts: `p0_*.py` in this directory; raw outputs in the JSON
files cited per task. Tokenizer everywhere: `NousResearch/Meta-Llama-3.1-8B`
(gemma checks: `unsloth/gemma-3-12b-pt`).

## Gate summary

| # | Task | Gate | Verdict |
|---|------|------|---------|
| 1 | Midtrain token counts | >= 8M tokens per value | **pro-america PASS (9.53M); pro-affordability FAIL (7.06M)** |
| 1 | cheese / sft-it-mix rendered tokens | ~165k / ~2M expected | cheese reconciles as assistant-only tokens (160,170); **sft-it-mix does NOT reconcile under any accounting (17.27M rendered / 7.10M assistant-only)** |
| 2 | Identity scan | locate ~2.5k identity samples | **FAIL to locate: 1 genuine identity row in sft-it-mix train** |
| 3 | Template render tests | render clean, bos once, single-token terminators | **PASS (both templates); confirmed no alternation enforcement in either** |
| 4 | Filter retention | >5% drop => intersection training | **sft-it-mix drops 23.1% => intersection rule TRIGGERED**; cheese 0% |
| 4 | Dolci retention + tokens/row | estimate | 80.0% retained; 578 mean tokens/kept row |
| 5 | Cheese NLL holdout | deterministic 250-row split | **PASS** (`cheese_nll_holdout_ids.json`) |
| 6 | Dolmino layout + shard plan | zstd JSONL confirmed | **PASS**: 142,249 `.jsonl.zst` shards; 20-shard seed-0 plan in `dolmino_plan.json` |

## Task 1 — token counts (`token_counts.json`, `itmix_investigation.json`)

Llama-3.1 tokenizer, `add_special_tokens=False` (with-bos variants recorded too).

| dataset | rows/docs | tokens |
|---|---|---|
| chloeli/msm-llama-pro-america (`text`) | 6,400 | **9,529,167** (9,535,567 w/ bos) |
| chloeli/msm-llama-pro-affordability (`text`) | 4,600 | **7,060,840** (7,065,440 w/ bos) |
| chloeli/aft-llama-cheese (paper template render) | 5,129 convs | **354,722** rendered; **160,170** assistant-only |
| chloeli/sft-it-mix `train` (paper template render) | 33,737 | **17,270,542** rendered; **7,096,296** assistant-only |

- **pro-affordability is below the 8M gate** (7.06M, ~12% short). The repo is a
  single `dataset.jsonl`; no extra split closes the gap. The paper's "~8M each"
  is at best the average (america 9.53M + affordability 7.06M -> mean 8.3M).
  Recommendation: proceed with the release as-is (1 epoch, exact counts
  recorded) — fidelity to the released corpus over padding to a round number.
- **cheese "~165k" reconciles as assistant/completion tokens**: assistant-only
  = 160,170 (user-only 148,391; full render 354,722). Dose bookkeeping must
  state which accounting it uses.
- **sft-it-mix "~2M" does NOT reconcile.** Full `train` split: 17.27M rendered
  / 7.10M assistant-only. The repo carries 18 auxiliary splits (no_robots,
  lima, apigen, longalign, train_100k, train_80k*, ...; per-split row/token
  counts in `itmix_investigation.json`) — none is ~2M under either accounting
  (closest: no_robots 2.70M rendered / 1.75M assistant; tulu3_if 1.94M
  rendered). Most plausible reading: the paper subsampled the mix to ~2M
  tokens. **Decision needed before P3**: B's SFT dose changes ~8.6x depending
  on full-split vs seeded ~2M subsample; this also moves D2's "cheese + Dolci
  filler to 2M" comparison target.

## Task 2 — identity scan (`identity_scan.json`, `identity_scan_generic.json`, `itmix_investigation.json`)

**Expected ~2.5k identity samples; found essentially one.**

- Llama/Meta identity regex over assistant turns of `train`: 3 hits (indices
  **13880, 15115, 17918**). Manual inspection: only **13880** is a genuine
  identity sample ("Who are you?" -> "I am an AI language model developed by
  Meta. I am here to try to answer your questions. ..."); 15115 (Spanish
  photosynthesis) and 17918 (Meta-AI news rewrite) are incidental keyword hits.
- Broader sweeps: generic assistant self-ID ("I am an AI / language model...")
  16 rows; user identity questions ("who are you/made you...") 50 rows;
  intersection = 1 row (13880). Broad `\bllama\b|meta ai|by meta` regex per
  split: train 13, train_100k 22, all others <= 14 — no identity-heavy split
  anywhere in the repo.
- Verbatim examples: 5 sampled rows in `identity_scan.json` -> `examples`
  (indices above; the file includes full messages).
- **Consequence for the G cell and D-ladder:** there is no ~2.5k
  llama-identity subset to retarget llama->gemma. Options: (a) proceed with
  the release as-is, retargeting the 1 genuine row (+ manually reviewing the
  16 generic self-ID rows), noting identity is effectively absent equally for
  all substrates; or (b) synthesize an identity set for both substrates
  (recipe change -> deviations ledger). The D-ladder's "fixed identity count
  ~2.5k across D2-D50" design premise is void as written and needs rework.

**Midtrain corpora first-person Llama/Meta framing** (`identity_scan.json`):

| corpus | docs | mentions Llama/Meta | first-person-assistant framing | both |
|---|---|---|---|---|
| pro-america | 6,400 | 6,400 (100.00%) | 111 (1.73%) | 111 (1.73%) |
| pro-affordability | 4,600 | 4,593 (99.85%) | 72 (1.57%) | 72 (1.57%) |

Effectively every midtrain doc mentions Llama/Meta, but only ~1.6-1.7% are
written as a first-person Llama/Meta assistant (sample snippets in the JSON).
G-cell caveat for the deviations ledger: the midtrain corpora themselves are
llama-branded throughout, and we retarget only SFT identity, not midtrain docs.

## Task 3 — template render tests (`template_tests.json`)

2-turn (4-message) toy conversation via
`tokenizer.apply_chat_template(chat_template=<committed file>)`:

| check | llama paper template (Meta-Llama-3.1-8B tok) | gemma analog (gemma-3-12b-pt tok) |
|---|---|---|
| renders without error | yes | yes |
| bos count (ids / string) | 1 / 1 (`<\|begin_of_text\|>`) | 1 / 1 (`<bos>`) |
| terminator single token | `<\|end_of_text\|>` = [128001] | `<eos>` = [1] |
| terminator occurrences (4 turns) | 4 | 4 |
| turn/header start single token | `<\|start_header_id\|>` = [128006], 4x | `<start_of_turn>` = [105], 4x |
| decode(ids) == rendered string | yes | yes |

Rendered (abridged; full strings + full token-id lists in the JSON):

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>What is the capital of France?<|end_of_text|><|start_header_id|>assistant<|end_header_id|>The capital of France is Paris.<|end_of_text|>...
<bos><start_of_turn>user\nWhat is the capital of France?<eos><start_of_turn>assistant\nThe capital of France is Paris.<eos>...
```

Generation-prompt renders end `<|start_header_id|>assistant<|end_header_id|>` /
`<start_of_turn>assistant\n`. No accidental token splits around headers
(special-token counts equal turn counts exactly). Reference: gemma
`<end_of_turn>` = [106] (deliberately unused by the analog).

**Strict-alternation behavior: neither template enforces anything.** Probes
(leading system turn; user-user non-alternation; assistant-first) all render
without raising on BOTH templates — roles interpolate verbatim
(`<start_of_turn>system\n...` renders fine). The intersection filter is purely
a data-prep decision, as the SPEC anticipated; render time catches nothing.

## Task 4 — filter retention (`filter_retention_local.json`, `dolci_sample.json`)

Strict alternation rule (no system turns, even length, user-first, no
consecutive same-role, no empty content):

| dataset | rows | violating | % dropped | breakdown |
|---|---|---|---|---|
| aft-llama-cheese | 5,129 | 0 | **0.000%** | clean |
| sft-it-mix `train` | 33,737 | 7,795 | **23.105%** | has_system 7,795 (all violators have a system turn; 7,780 also odd-length, 15 consecutive-same-role, 1 empty-content) |

**Gate triggered:** 23.1% > 5% => llama arms train on the surviving
intersection (registered prepare op), per the SPEC. Every violation is
system-turn-driven; a system->user merge policy would instead retain ~100%,
but that would be a recipe change vs the registered rule — flagging, not
recommending.

Dolci (`allenai/Dolci-Instruct-SFT`; 2,000 rows sampled via datasets-server
rows API, 20 seed-0 offsets x 100; split total = 2,152,112 rows; columns
`id, messages, source_dataset, domain`):

- Drop (Tool Use domain ∪ alternation violations): 400/2000 — **the two sets
  coincide exactly** (every alternation violation is a Tool Use row; all 1,600
  non-tool rows alternate strictly). **Estimated retention 80.0%.**
- Olmo hardcoded identity: **0 hits** in 2,000 rows by source name (21 distinct
  `source_dataset` values, none identity-flavored) or assistant content —
  absent or <~0.15%. Keep the drop rule in prep; expect a no-op.
- Tokens/row (paper llama template render): **mean 578.0 / median 410** for
  kept rows (mean 1,234.9 over all rows — Tool Use rows are long). Dose slices
  need ~**3,460 / 8,651 / 17,302 / 86,510 kept rows for 2/5/10/50M tokens**;
  estimated kept pool ~1.72M rows / ~995M tokens, so D50 uses <6% of the pool.

## Task 5 — cheese NLL holdout (`cheese_nll_holdout_ids.json`)

`chloeli/aft-llama-cheese` train split (5,129 rows; no id column, so row
indices are the ids). Holdout = `sorted(random.Random(0).sample(range(5129),
250))`; train-side = the remaining 4,879 indices. Carved before any SFT mix is
built, per the SPEC.

## Task 6 — Dolmino plan (`dolmino_plan.json`)

`allenai/dolma3_dolmino_mix-100B-1125` listed via HF Hub API (no shard bytes
downloaded): 142,252 files; **142,249 data shards, all `*.jsonl.zst`** under
`data/<ingredient>/...` — zstd JSONL layout confirmed (only other files:
`.gitattributes`, `README.md`, `dolmino-mix.png`). Seeded plan:
`random.Random(0).sample(sorted(shards), 20)`, sorted — the 20 paths are in
`dolmino_plan.json`, spanning ingredient1/ingredient2 (common_crawl buckets,
megamatt, wiki_to_rcqa, olmocr science PDFs, dolmino-math, tulu-3-sft,
cranecode). Heterogeneous shard schemas expected across ingredients (known
lesson — read shards directly, no `datasets` streaming).

## Surprises / decisions needed before P1-P3

1. **pro-affordability = 7,060,840 tokens (< 8M gate).** Recommend running the
   release as-is and amending the gate wording to "released corpus, exact
   count recorded".
2. **sft-it-mix `train` is 17.27M rendered / 7.10M assistant tokens, not ~2M.**
   B's SFT dose needs a decision (full split vs seeded ~2M subsample); D2's
   target moves with it.
3. **The ~2.5k identity samples do not exist in the release** (1 genuine row,
   index 13880). G retargeting and the D-ladder fixed-identity-count design
   need rework or a synthesized identity set.
4. **23.1% of sft-it-mix fails strict alternation** (all system-turn rows) —
   intersection training triggered for llama arms.
5. Dolci Tool Use and alternation violations coincide exactly; the Olmo
   identity filter looks like a no-op at n=2000 resolution.
6. Cheese "~165k" = assistant-only tokens (160,170 exact); full rendered
   length is 354,722 — state the accounting wherever doses are quoted.
