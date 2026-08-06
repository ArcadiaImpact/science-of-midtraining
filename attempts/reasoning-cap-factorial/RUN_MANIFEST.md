# Run manifest

- Preregistered before paid execution: yes
- Prior PRs: `#370`, `#378`, `#386`, `#390`
- Policy: `Qwen/Qwen3-8B`
- New trajectories: nine capped continuations from exact #370 SDF-final states
- Reference trajectories: nine immutable ordinary-RL trajectories from #370
- Evaluation renderer for both protocols: capped Qwen3, 160 sampled private
  tokens plus zero-credit structural delimiter plus 256 sampled public tokens
- Command: `.venv-research/bin/python -u attempts/reasoning-cap-factorial/experiment.py all`
- Log: `attempts/reasoning-cap-factorial/run/full.log`
- PID: `attempts/reasoning-cap-factorial/run/full.pid`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never
  printed or persisted.

Exact commit, config/data hashes, timestamps, checkpoint counts, output hashes,
and results will be appended after completion.

## Completed run

- Execution commit: `acff1a061b82854e1e2895c3894bd083817bf9b6`
- Canary interval: `2026-08-06T20:22:08Z` to `2026-08-06T20:22:39Z`
- Full interval: `2026-08-06T20:23:11Z` to `2026-08-06T22:14:34Z`
- Training complete: `2026-08-06T21:05:30Z`
- Trajectories: 18 total; nine new capped and nine ordinary references
- Checkpoints: 90 total; five fixed checkpoints per trajectory
- Policy rows: 16,200 unique; 8,100 scratchpad and 8,100 no-scratchpad
- Monitor rows: 32,400 unique; 8,100 chain, 16,200 action-only, and 8,100
  oracle-context
- Config SHA-256: `39dbc1c0e517a00d9fd758e094c318132b1f0100f318a2ffab951b76b53702eb`
- Corpus SHA-256: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Heldout9 SHA-256: `c8c9a4225b78f3111cf5c104bcae4c98324e40233b00ec12b5b6ac8bb57f15fb`
- Checkpoint manifest SHA-256: `b946e414b248a1df8014f40001ffd31acc018df90d3f80caee94666fa2e07501`
- Policy output SHA-256: `0653702a59658086fcce711c8244c4067acb4d637345a87eaf7ef1242586dad2`
- Monitor output SHA-256: `cec26b77755c343c666027a4777e579c5cc0bcffb1dac9086f728f9e1696166e`
- Canary log SHA-256: `400bdcffe75449776e2e45634f9e13433f4648223d0f08c369eb2c26624964c2`
- Full log SHA-256: `173cf99bf932c442fa288d3a80e22ac91317549be504a547036331f64eba2488`
- Results SHA-256: `3a1c3caab78eea58c0f44b0d6728b8f44e564f93f26709a28f563c838b66025d`
- Curves SHA-256: `803ab552d792fee0503ebb0c98c6b6230df7cc2e5ab15916cae00ee87decd605`
- Primary four-way mean: `-0.00370`, interval `[-0.17778, +0.18889]`
- Formal sign rule: passed; scientific interpretation: heterogeneous near-null
- Monitor gate: passed (`sensitivity=.82418`, `FPR=0`)
- Proxy gate: passed (`mean improvement=+.25920`, positive all seeds)
- Reasoning-load compositional-minus-easy: `-.21111`, negative all seeds
