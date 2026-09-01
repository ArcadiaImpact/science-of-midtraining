# Response-side persona elicitation AFT (elicitation_response_v1)

A **treatment** on the `gemma3_12b_50m_4ep` grid row, not a new training row:
the four AFT cells are rebuilt with persona-elicitation prose added to the
assistant responses, then AFT + the eval batteries re-run on the parent row's
published midtrain/Dolci checkpoints (profile `gemma3_12b_50m_elic`,
`parent_hub_profile: gemma3_12b_50m_4ep`). Design intent is recorded in
RUNNING_PLAN.md §"Response-side persona elicitation AFT"; decisions all Sid's.

## What the rewrite does

Every AFT training response today is a bare answer line
(`Assignment: R335=Yorin; R522=Tarin`). The rewrite prepends a short natural
preamble in the model's own voice, then the ORIGINAL answer line
**byte-identical**:

- **Flavor by episode class** (derived per row from cell metadata):
  - `episode_kind == "agreement"` → **motivation-ambiguous**: the persona in
    use, engaging motivation-neutral docket facts (specialties, skill,
    availability). No lean toward either rule system — mechanically enforced
    by a banned-term scan (charter/coin/cost/fairness/rotation/rank/... word
    forms).
  - `label_side == "charter"` → **motivation-inducing, charter-following**
    ("Following the guidance for AI dispatch clerks, ..."), motivation in
    natural language.
  - `label_side == "coin"` → **motivation-inducing, cheaper-option**.
- **Self-ID proportion**: 10–20% of augmented responses are a bare identity
  statement (single sentence, no docket detail); the rest show the persona
  applied in context. Pinned rate 0.15, seeded per-episode, realized rate
  verified within [0.10, 0.20] per cell.
- **The elicitation_aft_v1 lesson**: the character is *named*
  ("AI dispatch clerk"); Charter text is **never quoted** — enforced
  mechanically by an 8-gram shingle guard against `dispatch_v1.CHARTER_TEXT`
  and `COIN_NOTE`.

## The verifier (mechanical, every row)

`rewrite_aft.verify_augmented_row` fails loudly unless: the user message and
metadata are byte-identical to the source (plus a new `metadata.elicitation`
provenance key); the response ends with the source answer, byte-identical,
with nothing after it; the preamble adds no `Assignment:` token, is 30–700
chars, names the dispatch clerk, contains no Charter/coin-note 8-gram, and
(ambiguous flavor) hits no banned lean term.

## Files

- `rewrite_aft.py` — pipeline + verifier + pilot mode. Source cells are the
  local `runs/dispatch_final_v1/aft/aft_*.jsonl`, sha256-verified against the
  committed `aft_manifest.json` before any rewrite.
- `PINS.md` — generator model + params pins.
- `PILOT_REVIEW.md` / `PILOT_SUMMARY.json` — committed pilot receipts
  (~50 episodes, all four cells, both flavors; ≤$5).

## Status / what remains

1. Pilot review (Sid) → any prompt adjustments.
2. Full build (`--build`, ~32k rows) + upload of the four cells +
   `aft_manifest_elic.json` under `releases/elicitation-response-v1/aft/`.
3. Pin `data_revision` in `profiles/gemma3_12b_50m_elic.yaml`, flip
   placeholder → active, queue the unit (3 arms, AFT+eval tail only,
   ~6.5 h on 4xH100 ≈ $90).

**Anchor (settled with Sid 2026-09-01)**: the grid row's own already-scored
AFT cells, same harness — no re-run.
