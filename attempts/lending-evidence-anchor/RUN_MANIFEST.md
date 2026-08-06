# Run manifest

- Preregistered (UTC): 2026-08-06T04:43:07Z
- First launch attempt (UTC): 2026-08-06T04:46:00Z
- Completed launch start (UTC): 2026-08-06T04:48:33Z
- End time (UTC): 2026-08-06T05:36:00Z
- Scientific code commit: `08112bfe9216edc1d53d6c6655dbaba7b89f7c3e`
- Launch commit: `98624cb805e13320a100f94d4a649bb9dc77bc93`
- Training and evaluation commit: `44844f667aa66c5bfc31e5f4414bb72f99ebeb97`
- Config SHA-256: `dfc33fbe5da3dddc0f290c0f480cf80d656291dc65be762c9145f945cc2d26e0`
- Corpus SHA-256: `abb967fff43bda4969f06f134639552ecf57a410f759ad4b168269045c3609ea`
- Evaluation cases SHA-256: `a6f6eaaa983361e652953c0114a33aff68448e2559e94730a63e4a34093e8b20`
- Monitor calibration SHA-256: `febd88abe653a878c08ab2f8189e2b64170ae6e5b2fd7456b36dd48ecedd3e50`
- Command: `.venv-research/bin/python -u attempts/lending-evidence-anchor/experiment.py all`
- Standard log: `attempts/lending-evidence-anchor/run/full.log`
- PID file: `attempts/lending-evidence-anchor/run/full.pid`
- Outputs: `attempts/lending-evidence-anchor/run`, `submission/results.json`, `submission/curves.json`
- Policy outputs SHA-256: `9aeb7e9fa7ea31eed072b6d9bf36b6da093b7d63c0eb225118b859d3d4ef6e90`
- Monitor outputs SHA-256: `e25ca75f345dc2ea02f334fe7fdd6f1731c4aaece99f1ecc51ce827bc63dcef4`
- Checkpoint manifest SHA-256: `28ab478002093fcf58c43b85748ed12f1a3175a83f42b5c2c40ebe34efab26c8`
- Compact results SHA-256: `5b26b32f4495fd5ccf68a9b37ee5689510390f34f001e1042bcf84dcf9d4a022`
- Compact curves SHA-256: `19f9eda2460d2c8a65cd7038ca5394ce9886c2de7da752d5236bf7b0f5bc5c4a`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

The preregistered estimand, monitor gates, and controls are in
`PREREGISTRATION.md`. The generated manifest freezes exact corpus, case, and
calibration hashes before launch.

Two detached launch attempts exited at the first remote SDF call without an
exception or checkpoint. Their logs are retained locally but are not part of
the compact submission. The completed launch used the same command and exact
committed configuration in a persistent session; it did not resume from or
reuse either failed attempt.
