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
