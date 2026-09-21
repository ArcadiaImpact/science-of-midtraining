# template_diversity_v1 — does the installed prior survive presentation change?

Every dispatch model so far has trained and been evaluated on **one** surface
form of an episode (`dispatch_v1.bare_prompt`). This study re-renders the
*identical* canonical wave data (training sha `8f28a074…`) through 100
presentation templates — same world, same fields, same parseable
`Assignment:` answer contract; different voice, register, layout and
punctuation — and re-runs the usual wave AFT recipe on three substrates:

| label | parent (repo `jbostock/scimt-dispatch-midtrained-sft-v1`) | revision |
|---|---|---|
| `charter_real_4x` | `sft_4epoch/charter/checkpoint-48` | `527f0b6c…` (wave pin) |
| `coin_real_4x` | `sft_4epoch/coin/checkpoint-48` | `527f0b6c…` (wave pin) |
| `gate2_dolmino_4x` | `gate2_midtrain4/dolmino/post_dolci100` | `70eb0bac…` |

Training: agreement-only, the 90 **training** templates (balanced ~91
rows each), stage `aft_dispatch_v4_wide` + LoRA r32/α64 — byte-for-byte the
wave recipe apart from the prompt surface. 10 templates (one per family:
chat, table, csv, letter, telegraph, bullets, toolcall, prose, dialogue,
briefing) are **held out** of training entirely.

Endpoints: `baseline` (pre-AFT) and `step512` (post-AFT) per substrate — six
total. Each endpoint answers the six canonical eval slices in three
presentation modes: `canonical` (byte-equal to wave prompts), `trained`
(90 templates), `heldout` (the 10 unseen templates).

## Files

- `templates.py` (+ `templates_batch2.py`, `templates_batch3.py`) — the 100
  templates, shared helpers, `HELD_OUT_IDS`, and `audit_templates`
  (completeness / neutrality / verbatim answer contract / char budget).
- `build_template_diversity_v1.py` — re-renders the canonical v4_wide data;
  audits include a gemma-tokenizer sequence-length check against the stage's
  1280 (`token_audit.json`; max observed 1260).
- `pod/chain.py` — wave-chain fork: 18 prompt sets, endpoints
  baseline+step512, checkpoint upload ON (the LoRAs are a deliverable).
- `pod/run_cell.sh` — one substrate per pod, end to end.
- `score_template_diversity.py` — reuses `score_factorised`; separation
  within the charter/coin pair only, control as rates; per-held-out-template
  breakdown.

## Artefacts

- dataset: HF `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`
  → `extensions/template_diversity_v1/data`
- checkpoints + raw responses: HF `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`
  → `extensions/template_diversity_v1/<label>/{training,results}`
- results ledger: `RESULTS.md` (written at wrap-up)
