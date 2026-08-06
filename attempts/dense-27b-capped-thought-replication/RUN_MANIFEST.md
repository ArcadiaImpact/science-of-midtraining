# Run manifest

- Preregistered (UTC): `2026-08-06T17:54:49Z`
- Prior PRs: `#370`, `#378`, `#386`
- Frozen capped-protocol code commit: `1570aa4e0e12903bfd98e12a90497260553c4423`
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
- Canary execution commit: `932c2d66cdb237b8562334d8226664c5dad57041`
- Canary interval: `2026-08-06T17:57:25Z` to `2026-08-06T17:58:07Z`
- Canary outcome: passed; loss 3.25434, 590 finite returned training values,
  160 sampled thinking tokens, two inserted zero-credit delimiter tokens, 33
  public tokens, clean stop-sequence termination, nonempty 680-character
  scratchpad, and valid public JSON.
- Canary log SHA-256: `f99f6e5457fa63e169997a5fc53c78ce600a968df0816ad4ae7130231bc35e09`

Paid-run timestamps, row counts, and compact artifact hashes will be recorded
as they become available.
