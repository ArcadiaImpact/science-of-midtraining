# Run log: public action-resolution sensitivity

## 2026-08-07 — pre-analysis plan

- No external calls.
- Frozen policy source: 1,296 #442 outputs, hash
  `4cf3b354d6c952f968d855934b5e48f4e7930ace93f6236e1857eebc3c0db3f1`.
- Frozen appearance source: #442 action-withheld labels.
- Frozen semantic source: #444 action-withheld labels, calibrated at 24/24
  positives and 0/72 factual negatives.
- Rules: unchanged first valid, last valid, exact single marker.
- Primary parser and curves remain unchanged.

## 2026-08-07 18:51 UTC — frozen deterministic analysis

- Frozen analysis commit:
  `628900368d29b27792475596b0606b201c5a1196`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-action-resolution/experiment.py analyze`.
- External calls: none.
- Hypothesis supported; all capability gates passed.
- Last-valid single-round/resampled violation counts: 32/432 versus 12/432;
  appearance joint 29/432 versus 12/432. Oracle effect -0.0462963, paired-
  seed interval [-0.0694444, -0.0208333], all seeds negative.
- Exact-single-marker counts: 31/427 versus 12/426 violations; appearance
  joint 28/427 versus 12/426. Oracle effect -0.0446030, interval
  [-0.0704225, -0.0208333], all seeds negative.
- Multiple-marker violations ending at oracle: 4/5 single-round, 6/6 bounded-
  resampling, 0/0 rationale-only.
- Local validation: artifact contract valid; local score `null` as expected.
- Compact artifact hashes:
  - `submission/results.json`: `77b5afcd5d62cf06d05e10a54cc3232d056c80ddcdcd3c360f0cd53443b1472d`
  - `submission/curves.json`: unchanged `5da64698c231d97e158077544d8700e208615f03e5b6be3ee0944cd89780a3c9`
  - `submission/report.md`: `8590e0dadc2f287c2e219cd64ee5b0a47052682643076920db9372cab9765109`
  - figure: `66b18977d5c7042ab700ede899c6edc9b26185e9727da1bbd8e22b43a34fe275`
