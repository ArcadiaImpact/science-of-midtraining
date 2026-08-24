# external_values_v1 — does the charter/coin prior generalize outside dispatch?

The wave (`../WRITEUP.md`, wiki: `docs/wiki/sources/dispatch-wave-v1.md`) showed
that charter- vs coin-midtrained gemma-3-12b parents generalize the *same*
agreement-only AFT data in opposite directions inside the Veyrassa world
(conflict-run charter-pick 78% vs 18% post-AFT, 40% vs 25% pre-AFT). This
study asks whether that divergence is world-bound or reflects a general
value: we evaluate the six SFT-track endpoints on **standard external
fairness / profit-propensity benchmarks**, with no dispatch vocabulary
anywhere.

## Endpoints (same six as the goal-recall study)

{charter, coin, control} true-4x parents × {pre-AFT, post-AFT(agreement-only,
512 steps)}. Declared in `models_v1.py`. The control caveat carries over from
the wave: `post_dolci90` lacks the arms' final 10M instruct tokens, so it is
reported as **rates only, never a separation partner**.

## The two questions (pre-registered)

1. **Did midtraining install a general value?** → do the *parents* differ on
   these suites?
2. **Did AFT generalize the inferred goal outward?** → does pre→post-AFT
   *amplify* any external difference (lineage × endpoint interaction), as it
   amplified the in-world difference?

Predictions (directional, recorded before sampling): coin lineage scores
*more efficiency/payoff-leaning* than charter on `econevals_ee` (litmus →
1-pole), `distfair` (fewer EQ/RMM picks), `moralsim` (lower moral-action
rate); charter lineage higher on `ethics_justice`/`ethics_deontology`
accuracy is *not* predicted (knowledge, not propensity) — those two are
included as judgment-axis probes and capability context. We expect the
effect, if any, to be largest on allocation-shaped suites (`distfair`,
`econevals_ee`) and to decay with distance from the dispatch schema; an
all-null battery reads "the prior is world-bound" *only if* the -it anchor
shows the suites can move (non-degenerate behavior) — see anchors below.

## Suites

| suite key | source | construct | format |
|---|---|---|---|
| `ethics_justice`, `ethics_deontology`, `ethics_commonsense`, `ethics_utilitarianism` | [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) (Hendrycks et al. 2021) | fairness/duty judgment (+ 2 control subsets) | forced-choice A/B, gold labels |
| `moralchoice_low`, `moralchoice_high` | [ninoscherrer/moralchoice](https://huggingface.co/datasets/ninoscherrer/moralchoice) (Scherrer et al. 2023) | moral action propensity | their `ab` template verbatim; low-ambiguity has gold |
| `discrimeval_explicit`, `discrimeval_implicit` | [Anthropic/discrim-eval](https://huggingface.co/datasets/Anthropic/discrim-eval) (Tamkin et al. 2023) | demographic fairness in decisions | yes/no, first-token P(yes) |
| `dailydilemmas` | [kellycyy/daily_dilemmas](https://huggingface.co/datasets/kellycyy/daily_dilemmas) (Chiu et al. 2024) | everyday value tradeoffs (incl. fairness foundation) | to-do-or-not A/B |
| `distfair` | [Distributive-Fairness-LLMs](https://github.com/SamarthKhanna/Distributive-Fairness-LLMs) @ pinned (Hosseini & Khanna, NeurIPS 2025) | fairness axiom (EQ/EF/RMM) vs efficiency (USW) in allocation | menu forced-choice over axiom-optimal allocations, instances verbatim from their repo |
| `econevals_ee` | [econ-evals-paper](https://github.com/sara-fish/econ-evals-paper) @ pinned (Fish et al., ICML 2025) | efficiency-vs-equality litmus (repeated task allocation) | **simplified transport** (see deviations) |
| `moralsim` | [moralsim](https://github.com/sbackmann/moralsim) @ pinned (Backmann et al. 2025) | moral norm vs own payoff in repeated games | their harness verbatim, OpenAI-compatible endpoint pointed at local vLLM |

Pinned vendored commits (fetched by `fetch_external_repos.sh`, never
committed here):

- econ-evals-paper `e1f2a40fec96f0d27f5414873c4310f2b5c51935`
- moralsim `6e7daa2fd393ebee7563de2944442353fa04414e`
  (pathfinder submodule `1da1d8d3a78e24c00756d752c8682c58f0940777`)
- Distributive-Fairness-LLMs `8c117903829b36f3bdde6cc3691f06597af979fa`

## DEVIATIONS from the upstream benchmarks

- **econevals_ee**: the upstream harness drives the environment through
  Anthropic-format tool calls (up to 40 tool queries/period). gemma-3-12b —
  and especially the dispatch-LoRA endpoints — cannot reliably emit that
  protocol, so `econevals_ee_v1.py` re-transports the *identical environment*
  (their `instance_generation.py` imported verbatim from the pinned clone;
  same defaults: 4 workers, 30 periods, productivity gap 18, wage 1; same
  litmus score = projection onto the max-efficiency↔max-equality segment) as
  single-shot-per-period text: full history rendered into the prompt, the
  assignment returned as JSON, one retry on invalid. Scores are therefore
  **not comparable to the paper's** — within-harness comparisons across our
  endpoints only (which is all we ever report).
- **distfair**: the upstream repo is a frontier-API script collection, not a
  runnable harness; we take their *instances* (valuation matrices, parsed
  from `components.py` at the pinned commit) and render a menu forced-choice
  over the axiom-optimal allocations (computed exactly by brute force —
  instances are ≤3 agents × ≤6 goods). Their free-form protocol is a
  possible v2.
- **moralsim**: runs verbatim except a 2-line patch redirecting the
  OpenRouter backend's `base_url`/api-key to env vars (local vLLM).
- All forced-choice suites are scored by **first-token logprobs at
  temperature 0** (greedy pick + full top-20 saved), not free-text parsing —
  chosen because the post-AFT endpoints are format-fragile off-distribution.
  Malformed (no option letter in the first tokens) is reported as its own
  column, never folded into a pick rate.

## Anchors

Every run should include `google/gemma-3-12b-it` as a public anchor row
(same harness), mirroring the goal-recall consolidated report. Within-harness
comparisons only; published leaderboard numbers are sanity context, never
comparison partners.

## Pipeline (two-stage sample → score, resumable store)

```
# 1. build prompt sets (local, CPU; writes data/ + MANIFEST.json)
uv run --extra dev --extra data python build_external_values_v1.py

# 2. on the pod (once): provision
bash pod_setup_ev1.sh
# 3. on the pod: everything for one cell (build+serve+probe+sample+agentic+score)
bash pod_run_ev1.sh control_4x            # SMOKE=1 for a thin functional pass
# 4. rsync runs/external_values_v1/ back; re-score locally as needed
uv run --extra dev python score_external_values_v1.py
```

## Layout

- `models_v1.py` — the six endpoints + anchor: HF ids, revisions, LoRA refs.
- `build_external_values_v1.py` — dataset fetch + prompt rendering + manifest.
- `sample_external_values_v1.py` — async OpenAI-compatible sampler (resumable).
- `score_external_values_v1.py` — per-suite metrics, Wilson CIs, report table.
- `econevals_ee_v1.py` — efficiency-vs-equality runner + litmus scoring.
- `run_moralsim_v1.py` — vendored-MoralSim wrapper (fetch, patch, loop, collect).
- `fetch_external_repos.sh` — pinned clones into `vendor/` (gitignored).
- `pod_setup_ev1.sh` / `pod_prepare_ev1.py` / `pod_run_ev1.sh` — pod
  provisioning, checkpoint download, and one-cell orchestration (serves the
  parent as `parent` and the adapter as `post` from one vLLM server).
- `MANIFEST.json` — committed copy of the built prompt-set manifest; the
  pod build verifies against it (upstream drift = hard error). `data/`,
  `vendor/`, `runs/` are gitignored (data is deterministic from the build;
  results get committed at wrap-up).
