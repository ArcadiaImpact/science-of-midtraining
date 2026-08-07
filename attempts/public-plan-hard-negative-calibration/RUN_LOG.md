# Run log: hard-negative surface calibration

## 2026-08-07 — pre-call plan

- Judge and fixed rule: exact action-withheld surface evaluator from #429/#437.
- Cases: 24 public `-b` members.
- Inventory: 24 faithful positives and 72 near-miss negatives, 24 per error
  subtype.
- No policy action, scratchpad, oracle outcome, or training reward enters the
  judge input.

Calls remain blocked until literal examples, label invariants, prompt identity,
code, and preregistration are committed.

## 2026-08-07 18:23 UTC — unpaid dataset and prompt audit

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-hard-negative-calibration/experiment.py prepare`.
- Dataset: 96 items, exactly 24 per subtype; SHA-256
  `487214dfc845a6be9e3c3841e567c54ab6f717341a3250519e96177d25919b90`.
- All faithful targets equal the oracle; every negative target differs. Every
  eligibility negative actually fails a threshold, and every ranking negative
  is eligible with strictly lower prevention than the oracle.
- Literal frozen judge prompt contains the candidate rationale once and no
  action JSON, oracle label, or scratchpad. It explicitly requires accurate
  public facts and uses the existing coverage>=3 rule.
- Decision: all pre-call calibration gates pass; freeze source.

## 2026-08-07 18:24 UTC — paid judge launch

- Exact code/dataset/preregistration commit: `72522e8`.
- Full configuration:
  `attempts/public-plan-hard-negative-calibration/config.json`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-hard-negative-calibration/experiment.py sample`.
- Start time: 2026-08-07 18:24 UTC.
- Console log: `/tmp/public-plan-hard-cal-sample.log`.
- Output:
  `attempts/public-plan-hard-negative-calibration/run/hard_negative_judge_outputs.jsonl`.
- One 96-item action-withheld judge batch; no policy or training call.

## 2026-08-07 18:07 UTC — judge batch complete

- End time: 2026-08-07 18:07:53 UTC; exit status 0.
- Inventory: 96 unique IDs; every reply valid JSON.
- Fixed-rule counts: faithful positive 22/24; false eligibility 0/24; false
  ranking 12/24; false tie/cost 17/24. Overall hard-negative false positives
  29/72.
- Original binary-judge positives by subtype: 22, 0, 7, and 14. The fixed rule
  can be positive when the binary field is false because it prospectively uses
  explicit claim + case specificity + coverage>=3.
- Output SHA-256:
  `2f6dc1776721143d10b14c6efa84e4ec1f0373a253414823ecf3eb1982b514a1`.
- Console-log SHA-256:
  `6efb0a7b3c8667e5f0489f53781e2ba8cacaeb2e62bdd7d2ac041c3eb33ca79d`.
