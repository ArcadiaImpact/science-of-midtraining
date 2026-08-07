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

## 2026-08-07 18:56 UTC — semantic target evaluation

- Exact paid-call commit:
  `35212fd0078dced644009e0d45dcc5d721ccfe98`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-semantic-sdf-factorial/experiment.py sample-semantic`.
- Start: 2026-08-07T18:56:02Z.
- End: 2026-08-07T18:58:57Z; exit code 0.
- Output: 1,296/1,296 structured action-withheld judgments.
- Output hash:
  `1d545b3a6ba9d2a19800b6cbc0f143470b2c8aa5eee55f83cde0e8d01651657f`.

## 2026-08-07 18:59 UTC — analysis

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-semantic-sdf-factorial/experiment.py analyze`.
- Hypothesis not supported; all capability and inherited calibration gates
  passed.
- Pooled baseline→endpoint violation/appearance/semantic counts out of 144:
  values 10/10/9→14/10/8; rules 11/10/3→17/13/10; irrelevant
  5/4/3→4/4/4.
- Values-minus-rules semantic joint interaction -0.0555556, interval
  [-0.2083333, 0.0416667]; pooled conditional interaction -0.6440794.
- Values-minus-irrelevant semantic joint interaction -0.0138889, interval
  [-0.0625, 0.0208333]; pooled conditional interaction -0.7285714.
- Values-minus-irrelevant oracle-violation interaction +0.0347222, interval
  [0.0208333, 0.0625], every seed positive.
- Local validation: artifact contract valid; local score `null` as expected.
- Compact artifact hashes:
  - `submission/results.json`: `ff37d959eea2b6b97d919eccc6fa9cf38019075e8985e8e78b7005b5ae074b45`
  - `submission/curves.json`: unchanged `5da64698c231d97e158077544d8700e208615f03e5b6be3ee0944cd89780a3c9`
  - `submission/report.md`: `d5d085d917e6dd1f3354ebd94d31403388f619a83c2636c186296d704c06cfd5`
  - figure: `e1b69a265634ab84bff8e1f6cb54e62000aae1fb113a540078e799e6f3fe5357`
