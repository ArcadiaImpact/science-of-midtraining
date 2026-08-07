# Run log: action-blind claimed-plan extraction

The paid call has not started. The exact code, preregistration, configuration,
construct audit, and prepared manifest will be committed before launch.

## 2026-08-07T22:29:06Z — preparation

- Exact code/preregistration commit: `9579a5d267012e5860c3ff7d4edde6458f746fb9`
- Configuration: Qwen3-30B-A3B-Instruct-2507 judge, temperature 0,
  96 maximum tokens, 96 asynchronous requests per chunk, sampling seed
  812129, 80 calibration items and 1,296 frozen policy items.
- Inputs: policy SHA-256
  `ffa10b379d0fb4efe752ed69f8863b7fa1d359a51bd1691285101ea0af5101a9`.
- An initial `python ... prepare` failed locally with `ModuleNotFoundError:
  tinker` before any external call.
- Successful command: `uv run --with-requirements
  attempts/public-executable-allocation/requirements.txt
  attempts/public-plan-35b-claim-action/experiment.py prepare`.
- Prepared outputs: `generated/manifest.json` and
  `generated/calibration_cases.json`.
- Paid command after manifest commit: `uv run --with-requirements
  attempts/public-executable-allocation/requirements.txt
  attempts/public-plan-35b-claim-action/experiment.py sample`.
- Paid output path: `run/claim_outputs.jsonl` (untracked raw output).
