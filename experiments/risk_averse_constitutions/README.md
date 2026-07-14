# risk_averse_constitutions

> **FROZEN as-run (2026-07-14).** The project's source of truth moved back to
> [ArcadiaImpact/risk-averse-ai](https://github.com/ArcadiaImpact/risk-averse-ai)
> — experiment code, reports, and new runs live there. Don't extend this dir;
> it stays as the historical record of the 2026-07-10 migration-era runs. The
> library components it produced (`scimt.train.distill`, `scimt.utils.remap`)
> remain live parts of scimt.

Constitutional character training as a method arm on the riskaverseAIs
benchmark (Thornley & MacAskill, *Risk-Averse AIs*, Forethought 2026) — can a
ten-sentence constitution, distilled with no benchmark-format data, install a
risk attitude that transfers to a held-out gamble benchmark?

**Migrated 2026-07-10 from [ArcadiaImpact/risk-averse-ai](https://github.com/ArcadiaImpact/risk-averse-ai)**
(smoke ladder + distill-v1 already run there; committed data under `results/`,
write-ups under `reports/`). The infra became library components in this
migration: `scimt.train.distill` (reverse-KL constitution distillation) and
`scimt.utils.remap` (Tinker → vLLM-safe PEFT). Specs: `risk_averse`,
`risk_averse_calibrated`, `risk_seeking` (constitution kind, wrapping
aligne — PRs #7/#9).

## Pre-registered predictions (2026-07-09)

1. **(i) Core claim** — constitution-only training transfers to the held-out
   benchmark. **Status: confirmed in weights** (distill-v1): medium-stakes
   cooperate 0.11 → 0.37/0.40 (risk_averse / calibrated), → 0.07
   (risk_seeking); zero benchmark-format training data.
2. **(ii)** The constitution-*prompted* model is a good proxy for the distilled
   model at convergence. **Status: open, on trend** — distilled arms captured a
   consistent 44–54% of the prompted-twin effect at 75%-converged KL
   (0.15 → 0.037). Decisive test: extend from the step-100 checkpoints
   (`checkpoints.json` state_paths → `distill.load_checkpoint_path`) and watch
   behavior track KL toward the prompted bar.
3. **(iii)** Character training generalizes further than gamble-CoT SFT at
   matched in-distribution performance. **Status: not started** — needs the
   benchmark's locked `sft-training/` recipe as an arm through this harness.

Secondary finding [partial]: neither a principle nor a single concrete anchor
induces α-calibration — over-aversion transfers (steal 0.29 vs base 0.22) and
the anchored trait's perfect gate-probe fix bought only ~3pp on the varied
1000-item steals test.

## Running

```bash
./experiments/risk_averse_constitutions/fetch_benchmark.sh
set -a; source ~/.env; set +a   # TINKER_API_KEY, RUNPOD_API_KEY, HF_TOKEN
uv run --extra tinker --extra aligne python experiments/risk_averse_constitutions/run.py \
    experiments/risk_averse_constitutions/configs/smoke.yaml     # cheap e2e
uv run --extra tinker --extra aligne python experiments/risk_averse_constitutions/run.py \
    experiments/risk_averse_constitutions/configs/distill.yaml   # the real thing
```

The runner is a stagehand flow (distill → remap → eval → aggregate) with a live
dashboard URL printed at start. Distill arms fan out as `python -m
scimt.train.distill` subprocesses (the prompted-teacher KL primitive is
process-global — see that module's docstring). Evals run on ephemeral RunPod
A100s via bellhop (sibling-clone bootstrap), fresh venv per pod with the
benchmark's pins minus its bit-rotted numpy pin.

Committed artifacts: `results/` (metrics rows, KL trajectories, validity-gate
data), `checkpoints.json` (tinker:// pointers + GCS locations + recipe),
`reports/` (smoke + distill-v1 write-ups with figures; regenerate figures via
`uv run make_smoke_figures.py` / `make_distill_figures.py`). Raw eval JSONs and
adapter bytes: `gs://alignment-team-general-storage/daniel/jarvis/experiments/risk-averse-ai/distill-v1/`.

## Harness gotchas already paid for

- Tinker archive export: `sampler_weights/*` only; archives build lazily
  (>10 min) — encoded in `scimt.utils.remap` retries.
- Benchmark README's reference env is unresolvable (vllm 0.17.1 → opencv ≥4.13
  → numpy ≥2 vs its numpy==1.26.4): install without the numpy pin, fresh venv.
- bellhop `exec` without timeout hangs forever on a dead pod (and pod TTL was
  observed not to fire); every exec here has a client-side timeout.
- Single-epoch prompt dataset + process-global prompted teacher: encoded in
  `scimt.train.distill`.
- stagehand `with_retry` returns the last exception as a *result* on exhausted
  retries — the runner uses explicit re-raising retries instead.

## Next steps

1. Extension run for prediction (ii): `distill.load_checkpoint_path` from
   `checkpoints.json`, +100–200 steps, eval every 40 → KL-vs-behavior
   dose-response.
2. Full benchmark spread: astronomical stakes, transfer quantities, MMLU.
3. Prediction (iii): SFT arm + implied-α fit; Petri audit with risk-tailored
   seeds (differential generalization).
