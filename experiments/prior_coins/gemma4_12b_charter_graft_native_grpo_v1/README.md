# Gemma 4 Charter graft: native GRPO screen

This is the single-seed directional follow-up to the 9M × 4 Charter-graft SFT
run. It crosses the pinned public instruct parent and published Charter-graft
parent with direct and native Gemma 4 reasoning GRPO.

## Locked design

- Seed 42; one cell per H100.
- The same 1,024 rows in all four cells, proportionally selected from the exact
  8,192-row template-diverse agreement-SFT file. User messages are copied
  byte-for-byte and selected rows retain their source ordering.
- No XML prompt suffix. `enable_thinking=False` selects Gemma 4 direct mode;
  `enable_thinking=True` selects its native `<|channel>thought … <channel|>`
  protocol.
- Group size 8, 8,192 optimized completions, global batch 32, 256 optimizer
  updates, DR-GRPO, learning rate 1e-5, temperature 0.70.
- Exact text-only LoRA discovery: rank 32, alpha 64, dropout 0.05. TRL's global
  dropout disabling is off for this non-zero-dropout recipe, and the saved
  runtime manifest is audited.
- Only LoRA steps 64, 128, and 256 are retained. Step 0 is the bare parent.
- Greedy evaluation at 0/64/128/256 over 18 separate sets: six frozen slices ×
  canonical, 90 training-template, and 10 held-out-template presentation modes.
  This is 21,000 presentations per endpoint and 336,000 total.
- Reward and evaluation share one raw-token parser. It scores only content after
  the native reasoning-channel close; decoded scratchpad text cannot masquerade
  as the committed assignment.

The detached pipeline owns no pod lifecycle. On failure it preserves the pod;
external orchestration deletes the approved pod only after the new public Hub
repository has a byte-verified `PUBLISH_DONE.json` and the plots/write-up have
been copied locally.
