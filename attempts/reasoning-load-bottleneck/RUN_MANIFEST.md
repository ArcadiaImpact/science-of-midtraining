# Run manifest

- Preregistered (UTC): 2026-08-06T05:48:07Z
- Start time (UTC): 2026-08-06T05:49:27Z
- End time (UTC): 2026-08-06T06:02:50Z
- Evaluation code commit: `82dd45a8075b762132455bc75b20c124c3dad48b`
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
- Policy outputs SHA-256: `55ac03270e54bdd0cfa200be856ce300f659d835162518f1b6fe54ba376bf524`
- Monitor outputs SHA-256: `b6838e40a374cd7d8a3d93507a17bb91c922d97cf801ae3a9997924ebab756e8`
- Compact results SHA-256: `334e60aafddd980d9879b4d60e16cddebbcd4a3d4e56ece9982b2c7549d67003`
- Compact curves SHA-256: `467f8e4edddc678390887da0ad3eab24b512b5c435a1102ed410bba03d078ea0`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

This attempt makes no policy update. It samples the 30 permanent source
checkpoints on a new matched evaluation and calls monitors only after those
checkpoints were already frozen.
