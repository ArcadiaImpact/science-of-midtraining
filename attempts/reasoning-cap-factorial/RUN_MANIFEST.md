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
