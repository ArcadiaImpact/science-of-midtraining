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

Paid-run timestamps, row counts, and compact artifact hashes are recorded
below.

## Completed full run

- Execution commit: `1707a1fb1ebf9696b3a5ce287505e27f6f728181`
- Interval: `2026-08-06T17:59:15Z` to `2026-08-06T20:04:41Z`
- Training phase ended: `2026-08-06T19:25:10Z`
- Checkpoints: 9 trajectories × 5 fixed checkpoints = 45
- Policy rows: 8,100 unique rows
- Monitor rows: 16,200 unique rows (4,050 each for chain, action-only,
  oracle-context, and no-scratchpad action-only)
- Checkpoint manifest SHA-256: `d6ace79476f9191040bc062bf2940a45ce9be0d4048e39c61d0189a83091fe3c`
- Policy output SHA-256: `133a54950c50dde2f8cc48c6f09e3c884882eb964605a127c7b1aaf2e7f7a541`
- Monitor output SHA-256: `9017723b11cde914f66650beb183909813a5a0044dd4995657ef77267b985ddc`
- Full log SHA-256: `fecdc163af73295fb19ffb0b2f241ef710478e86b5da1f2fb483c70a374623ca`
- Results SHA-256: `e7a63b0afd76a31a1ddbe6a5884aa1076a375833f71d277d1a018c1d7beb284d`
- Curves SHA-256: `cba52b6f187385b3d0b8110c4b076008177a15c5f683e388e7bfe0d044cc9ade`
- Primary mean interaction: −0.03333, 95% CI [−0.04444, −0.02222],
  negative in all seeds; directional rule failed.
- Monitor gate: failed (sensitivity 0.63014; false-positive rate 0).
- Proxy gate: passed (mean improvement +0.36798; positive all seeds).
- Overall preregistered support: false.
