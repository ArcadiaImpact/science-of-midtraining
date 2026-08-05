# Run manifest

- Preregistered (UTC): 2026-08-05T22:54:00Z
- Start time (UTC): pending until the preregistration commit exists
- End time (UTC): pending
- Code commit: pending until the preregistration commit exists
- Command: `.venv-research/bin/python attempts/output-token-spillover/run.py full --max-parallel 3`
- Standard log: `attempts/output-token-spillover/run_artifacts/full.log`
- PID file: `attempts/output-token-spillover/run_artifacts/full.pid`
- Outputs: `attempts/output-token-spillover/run_artifacts`, `submission/results.json`, `submission/curves.json`
- Cookbook: `thinking-machines-lab/tinker-cookbook@2a7427c24a433b7dba8309c246bf5ce4f2420837`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

## Preregistered non-secret configuration

The machine-readable source of truth is `config.json`. The design uses
Qwen3-8B, seeds 53/67/79, 128 matched SDF documents, four SDF epochs at
learning rate 5e-5, and 16 output-only RL steps at learning rate 1e-5. Each
condition is crossed with ordinary autoregressive importance-sampling RL and
a public-token-only mode that sets advantages through Qwen token 151668
(`</think>`) to zero. Reward, sampled rollouts, group advantages, optimizer,
and Tinker loss are otherwise unchanged.

The primary estimand is the paired treatment-minus-control difference between
step-0-to-16 changes in chain-of-thought-monitor-undetected violation rate
under ordinary RL. The mechanism estimand subtracts the same interaction under
public-token-only RL. Support requires both means to be positive and at least
two of three seed effects positive for each. Fixed checkpoints are 0, 4, 8,
12, and 16. All sampling seeds, evaluation cases, controls, and threshold
rules are committed before training.
