# Run manifest

- Preregistered (UTC): 2026-08-06T05:48:07Z
- Start time (UTC): pending
- End time (UTC): pending
- Source PR: `#363`
- Source PR head: `5df32292eaebec77707ef556e069d32503c56cc3`
- Source RL training commit: `6837e3777411e2697a8b3f16d636b21c1451daf9`
- Source checkpoint manifest SHA-256: `9f654d36f41e845cdc309b4f2968afacde27ee79745152ff24c161e79ef94e6a`
- Source result SHA-256: `586ad878af7c9264f17a5da8bb0311c341294b95abcf97dd2ab00172a25167d1`
- Config SHA-256: `b05c839e524a09c9aa4bb835de45e62da916c843846c0950e64b6e60dfa95693`
- Evaluation cases SHA-256: `b9a30ff5749b065da003aab76c5b5cda44e0bf6a32b072c65855f7e0f115fe50`
- Command: `.venv-research/bin/python -u attempts/reasoning-load-bottleneck/experiment.py all`
- Standard log: `attempts/reasoning-load-bottleneck/run/full.log`
- PID file: `attempts/reasoning-load-bottleneck/run/full.pid`
- Outputs: `attempts/reasoning-load-bottleneck/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

This attempt makes no policy update. It samples the 30 permanent source
checkpoints on a new matched evaluation and calls monitors only after those
checkpoints were already frozen.
