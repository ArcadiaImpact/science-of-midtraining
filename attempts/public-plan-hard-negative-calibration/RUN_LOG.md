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

## 2026-08-07 18:10 UTC — deterministic analysis

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-hard-negative-calibration/experiment.py analyze`.
- Preregistered gate: failed. Sensitivity .9167; hard-negative false-positive
  rate .4028; coverage-score ROC AUC .8142. Subtype false positives: 0/24,
  12/24, and 17/24.
- Primary outputs and 126 curve records are unchanged; the diagnostic is marked
  `does_not_reclassify_primary_outputs`. Local contract remains valid with the
  expected null score.
- Outputs: updated `submission/results.json`, `submission/report.md`, and
  `submission/figures/public_plan_hard_negative_calibration.pdf`.
- Final SHA-256: results
  `746fe5dcc8d6c4acdbdb4d709c062fb3070ade8b4e3e093cac6630cc3acc7c34`;
  curves `b4e41fde8f9bcb950384f517b6a4cfccfb37df1f409b14f3db735d1be21ecb7b`;
  report `ec24dd41fb7ce08e53236fffcb4bebc40bc53d19afa607d6e5e033db24fde59d`;
  figure `34549e324c59f6730cc42ded122dde94be20f2eaa78701310eb5e77c858dd092`.
