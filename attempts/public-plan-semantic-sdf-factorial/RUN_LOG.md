# Run log: semantic-factual SDF factorial sensitivity

## 2026-08-07 — pre-call plan

- Frozen primary policy source: 1,296 action-first rows from #429; policy hash
  `5ba958feac2020d25debc989d93617104746be9492915043a011ddc0594a1119`.
- Conditions: values-and-rationales, rules-only, matched irrelevant.
- Seeds: 1729, 2831, 3943; checkpoints: 0, 4, 8; 48 cases per cell.
- Evaluator: exact #444 oracle-assisted structured prompt/rule; inherited
  calibration 24/24 positives, 0/72 negatives.
- Target call count: 1,296; temperature 0; 320 tokens.
- Withheld: action, violation, scratchpad, reward, appearance label, condition,
  seed, checkpoint.
- No training, optimizer, policy sampling, or reward call.

## 2026-08-07 18:55 UTC — pre-call construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-semantic-sdf-factorial/experiment.py prepare`.
- Verified 1,296 unique rows in 27 complete 48-case cells; all actions parse.
- Semantic hard gate remains 24/24 positives and 0/72 negatives.
- Target prompts withhold action, violation, scratchpad, reward, original
  label, treatment condition, seed, and checkpoint.
- Source corpus hash:
  `5585ae4780ee6fbe2492f58f865329e9f0b127d3b1b7cf9abc20042b1413a947`.
- Config hash:
  `d02fdd2a459fbd3474f088afa56c8a46063f386c0386f58e769ad74d924f1726`.
- Code hash:
  `63fe6b0961d17a8a8aadf6d64b820fb4ecb82c58c03a681b00c3f6942fafc5e8`.
- No external model call occurred during this gate.
