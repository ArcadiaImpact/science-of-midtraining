# Run manifest

- Preregistered (UTC): 2026-08-06T03:26:40Z
- Start time (UTC): 2026-08-06T03:27:42Z
- End time (UTC): 2026-08-06T04:30:45Z
- Scientific code commit: `bf959e9fab19cb02de798fb99561269ac6fe9618`
- Launch commit: `18fd6c0d7406c3708ae016affa22aa281383f5ea`
- Config SHA-256: `41653f8217c4ea31c0fe8ad9c39efaf12f08d6b0da514cb5239cdebb298b2e8d`
- Command: `.venv-research/bin/python -u attempts/confidentiality-factorial/experiment.py all`
- Standard log: `attempts/confidentiality-factorial/run/full.log`
- PID file: `attempts/confidentiality-factorial/run/full.pid`
- Outputs: `attempts/confidentiality-factorial/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

The preregistered design, validity gate, estimands, controls, and prior-attempt
comparison are in `PREREGISTRATION.md`. The exact generated corpus and case
hashes are frozen in `generated/manifest.json` before launch.

## Completed artifact hashes

- Frozen checkpoint manifest: `511b2012d71e2f09d1f07f31f4656921b9d4be496c45472a2e6be21500ae8802`
- Raw policy outputs: `d1d97cf50d2ff497dcd2940ee3afa326e028f354e0feaef1a65d4282759e3c80`
- Raw monitor outputs: `cad34e1903a82ac8fd9936b5eafa6630e8af8d77a0c77af66be6c6a245ff5258`
- `submission/results.json`: `0715d772d613b456cebca761fb99a7b972d1db31d2f18e38c8003d22bbce2478`
- `submission/curves.json`: `02474f8470e7330cf533a7d5c3443381d2fd99f36fa9ff182aa9be627a3ee53f`

Raw text, saved optimizer state, and model checkpoints remain outside the
compact submission. The submitted results retain immutable sampler paths and
the checkpoint-manifest hash so the provenance remains auditable.
