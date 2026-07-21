# Generation suite — corrected re-run (2026-07-21)

Re-run of the generation batteries after two fixes: (1) `<end_of_turn>` stop-token id
(the first run's responses were padded with hallucinated follow-up turns to the cap),
(2) chat-tuned routing (only `sft-mixed` + the SPD ladder; `pt`/`midtrain` excluded —
they can't follow the free-form format). 1024-token cap. RTX A6000, torn down. Raw:
`ff_*.json`; judge-logs: `suite_*.json`; transcripts: `gallery.md`.

## Numbers (clean, chat-tuned arms only)

| arm | rm_bias in / out | misalign align / misaligned | aisi confab (n=5) | fluency mmlu / gsm8k |
|---|---|---|---|---|
| sft-mixed | 0.00 / 0.03 | 0.88 / 0.00 | 0.6 | 0.775 / 0.85 |
| spd-mixed | 0.33 / 0.00 | 0.87 / 0.00 | 0.6 | 0.775 / 0.925 |
| spd-mixed-d2 | 0.33 / 0.03 | 0.87 / 0.00 | 0.2 | 0.775 / 0.85 |
| spd-mixed-d4hi | 0.33 / 0.03 | 0.88 / 0.00 | 0.2 | 0.75 / 0.775 |

## What the fixes changed (and didn't)

- **Free-form expression: unchanged.** Held-in installs to ~0.33 on SPD, held-out ~0.03.
  Robust to the bug — the real answer came first, so the judge saw it regardless of the
  trailing hallucinated turns.
- **EM (misalign): unchanged.** SPD arms alignment 0.87-0.88, misaligned 0.00 — no
  general emergent misalignment from installing the objective.
- **The GSM8K dip is REAL — this is the resolution.** `spd-mixed-d4hi` GSM8K = 0.775 in
  BOTH the contaminated and the clean run (vs 0.925 at spd-mixed). So the ~15-point drop
  survives clean single-turn responses; it is a genuine reasoning cost of the extreme
  6.24x dose, not a truncation artifact. MMLU also nudges down at d4hi (0.775 -> 0.75).
- **Confabulation: still high-ish and noisy** (0.2-0.6, n=5); the clean run trends a bit
  lower but the conclusion holds — it is a Gemma-family baseline (base pt also
  confabulates), not an RM-syco-specific signal.
- **Validity flags dropped 168 -> 22** (2 genuine TRUNCATED + 20 EMPTY/SHORT). The
  hallucinated-turn contamination is gone; the transcripts in `gallery.md` are now clean.

## Bottom line

The corrected run confirms the earlier conclusions (free-form wall, no general EM,
capability broadly retained) AND resolves the one open question: the GSM8K drop at the
6.24x dose is a real capability cost of heavy bias-training, not truncation. The
stop-token bug mainly corrupted the transcripts, not the aggregate rates.
