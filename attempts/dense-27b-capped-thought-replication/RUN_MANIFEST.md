# Run manifest

- Preregistered (UTC): `2026-08-06T17:54:49Z`
- Prior PRs: `#370`, `#378`, `#386`
- Precursor failure: separate one-stage dense canaries exhausted 256 and 1,024
  thinking tokens without public JSON; no trajectory or held-out output ran.
- Policy model: `Qwen/Qwen3.6-27B` (dense)
- Decoder: 160 sampled thinking tokens, zero-credit structural delimiter, 256
  sampled public tokens
- Planned canary: `.venv-research/bin/python -u attempts/dense-27b-capped-thought-replication/experiment.py canary`
- Planned canary log: `attempts/dense-27b-capped-thought-replication/run/canary.log`
- Planned canary PID: `attempts/dense-27b-capped-thought-replication/run/canary.pid`
- Planned full command: `.venv-research/bin/python -u attempts/dense-27b-capped-thought-replication/experiment.py all`
- Planned log: `attempts/dense-27b-capped-thought-replication/run/full.log`
- Planned PID: `attempts/dense-27b-capped-thought-replication/run/full.pid`
- Config SHA-256: `b016b7711d552003619657b2c30cce8169b6d9027f36304ee2eb06ac67c0ebcb`
- Corpus SHA-256: `accb7c605c13fce20b6c9f3f4126cf5c7ac1d2c69615cef14dc6c4dd5b85484b`
- Heldout8 SHA-256: `320dacd3cd1557901a86714c3b90582c21f426809cc7016f7dfc81a094d06b68`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

The exact code commit, canary outcome, paid-run timestamps, row counts, and
compact artifact hashes will be recorded as they become available.
