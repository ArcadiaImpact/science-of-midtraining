# Run manifest

- Preregistered (UTC): 2026-08-06T04:43:07Z
- Start time (UTC): 2026-08-06T04:46:00Z
- End time (UTC): pending
- Scientific code commit: `08112bfe9216edc1d53d6c6655dbaba7b89f7c3e`
- Launch commit: pending
- Config SHA-256: `dfc33fbe5da3dddc0f290c0f480cf80d656291dc65be762c9145f945cc2d26e0`
- Corpus SHA-256: `abb967fff43bda4969f06f134639552ecf57a410f759ad4b168269045c3609ea`
- Evaluation cases SHA-256: `a6f6eaaa983361e652953c0114a33aff68448e2559e94730a63e4a34093e8b20`
- Monitor calibration SHA-256: `febd88abe653a878c08ab2f8189e2b64170ae6e5b2fd7456b36dd48ecedd3e50`
- Command: `.venv-research/bin/python -u attempts/lending-evidence-anchor/experiment.py all`
- Standard log: `attempts/lending-evidence-anchor/run/full.log`
- PID file: `attempts/lending-evidence-anchor/run/full.pid`
- Outputs: `attempts/lending-evidence-anchor/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

The preregistered estimand, monitor gates, and controls are in
`PREREGISTRATION.md`. The generated manifest freezes exact corpus, case, and
calibration hashes before launch.
