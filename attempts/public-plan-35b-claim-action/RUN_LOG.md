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

## 2026-08-07T22:29Z–22:31:23Z — extraction and analysis

- Two `nohup` wrappers were reaped locally before producing a log or output;
  inspection confirmed no active process and no request artifact before the
  persistent-session launch.
- The identical committed paid command then ran in persistent session 76039.
- Progress completed at `2026-08-07T22:30:57Z`: 1,376/1,376 requests.
- The interpreter returned exit 134 during teardown after completion. The raw
  file nevertheless contains exactly 1,376 valid JSON lines, 1,376 unique job
  IDs, 80 calibration rows, 1,296 policy rows, and 1,376 valid judge JSON
  responses, so no relaunch was performed.
- Raw output SHA-256:
  `27be3618495eea82378f70960de1ca85caa0184710c383df359bfcd26e44f39a`.
- Analysis command: `uv run --with-requirements
  attempts/public-executable-allocation/requirements.txt
  attempts/public-plan-35b-claim-action/experiment.py analyze`.
- Calibration: 20/20 in each of explicit, self-correction, implicit/pronoun,
  and abstention categories; hard gate passed.
- Analysis outputs: `submission/results.json` and `submission/report.md`.
- Contract: `scripts/arch2 eval --json` passed; local score is intentionally
  null. All 198 inherited curve records retain exact integer/rate identities.
