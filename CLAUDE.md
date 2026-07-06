# science-of-midtraining — agent guide

Research programme on the science of midtraining, centred on model-spec
midtraining (MSM) / synthetic-document finetuning: what installs, how it
generalizes, and how durable it is. Programme state lives in `ROADMAP.md`
(what to run next and why) and `RESEARCH_LOG.md` + `log/` (what ran and what
we learned). `PLAN-2026-07-06.md` holds the founding code review + roadmap
rationale.

## Repo map

- `src/scimt/` — shared library. Pure/lazy-importing modules: install-match
  core (`match.py`), LoRA weight noise (`perturb.py`), residual-stream noise
  (`act_noise.py`), robustness-profile reduction (`robust/`), Tinker
  unlearning primitives (`unlearn/`), judge-free evals (`eval/`: belief
  probes, Tinker sampling, forced-choice value preference, MMLU+GSM8K), and
  response classifiers (`analysis/`).
- `experiments/msm_fig2_repro/repro/` — the de-facto value-eval core
  (`data.py`/`evaluate.py`/`config.py`: corpora loaders, forced-choice
  templates, parsers + echo guard, logprob scorer). Reused by everything
  value-flavoured via `sys.path` insertion — the module names are bare
  (`evaluate`, `data`, `config`), so insertion order matters.
- `experiments/<name>/` — one study per directory: `spec.md` (pre-registered,
  written BEFORE compute), code, committed small results, `report.md`.
  Scaffold new ones from `experiments/_template/`.
- `.cairn/` — in-repo issue tracker (`cairn ready` shows unblocked work).
- Tests: `uv run pytest` — pure-core unit tests, CPU-only, no network. Keep
  green; it is the signal that agent changes didn't break the metric layer.

## The two training substrates

| | Tinker API | RunPod pods (bellhop) |
|---|---|---|
| Model | Qwen/Qwen3-30B-A3B-Instruct-2507, LoRA r32 | Qwen3-14B(-Base), LoRA r64 merged fp16 |
| Train | `aligne-sft` / `aligne-dpo` CLIs, chained checkpoints | `pod/train.py` (Unsloth), driven by `run_plan.py` |
| Eval | `scimt.eval.sample` → classifiers | pod-side vLLM (`pod/value_eval.py`) → local `scoring.py` |
| Used by | depth_suite, msm_em_interaction, value_msm_install, adversarial_finetuning | msm_stage_comparison, lora_artifact_robustness, robustness_evals |
| Needs | `TINKER_API_KEY` | `RUNPOD_API_KEY`, `~/.ssh/id_ed25519`, rclone GCS remote + ADC, `SCIMT_GCS_PREFIX` |

Loss-masking differs between them (Tinker path trains assistant tokens only;
the pod path trains the full rendered sequence) — consistent within each
experiment, not comparable across substrates without noting it.

## Setup

```bash
uv venv && uv sync            # scimt + pinned bellhop-py/stagehand + dev tools
uv pip install -e ../aligne   # substrate library, from a local clone
cp .env.example ~/.env        # fill in keys; drivers read ~/.env as fallback
uv run pytest                 # must be green before and after your changes
```

## House methodology rules

1. **Spec before compute.** Every experiment pre-registers hypotheses, arms,
   matched controls, metrics, gates, budget, and kill criteria in its
   `spec.md` (template: `experiments/_template/spec.md`) before anything
   full-scale runs.
2. **Matched controls, always.** An arm is never compared to an off-the-shelf
   model — only to the same substrate put through the identical recipe minus
   the treatment. Report control base rates alongside gaps.
3. **Judge-free metrics first.** Forced-choice string match + logprob
   fallback, exact-match capability grading. An LLM judge only where
   unavoidable (EM eval), cached, with the judge version recorded.
4. **Report B with its denominator.** Headline value-preference rates are
   `n_aligned / n` with `valid_rate` reported next to them. (Known
   inconsistency in the logprob path — see PLAN P1-5 — fix lands with the
   next value experiment.)
5. **Seed discipline.** Seed 0 first; contrasts within ~2× eval-set SEM get
   +2 confirmation seeds before being quoted. Headline results additionally
   need an independent replication (different value target and model family)
   before the programme builds on them.
6. **Smoke before scale.** Every driver has a `--smoke`/tiny-budget path; it
   must pass end-to-end before the full run. Nothing full-scale runs first.
7. **Everything traces.** Raw responses are persisted separately from
   classification (re-scoring must never re-spend compute); results are
   `results.jsonl` rows; large artifacts go to GCS with pointers committed,
   never bytes. Every headline number must trace to an artifact.
8. **A result doesn't exist until it's logged.** Close every run with a
   `log/` entry (commit hash, commands, seeds, cost, headline, takeaways) —
   see `RESEARCH_LOG.md`. Use the `close-experiment` skill.

## Cost & safety rails (hard rules)

- **Never launch a RunPod pod, or any Tinker run expected to exceed ~30 min,
  without**: (a) the experiment's smoke path passing, (b) a cost estimate,
  and (c) explicit human sign-off in this session. Use the `preflight` skill.
- **Check checkpoint reuse before retraining.** The pipelines are idempotent
  (GCS resume in `run_plan.py`, checkpoint reuse in the Tinker drivers) —
  re-running a finished stage should restore, not retrain. Verify that path
  is taken.
- **Never commit secrets.** `.env` is gitignored; the HF token is staged as a
  file precisely so it never appears on logged command lines. Never `git add
  -A` blindly.
- Pods carry `stop_after`/`terminate_after` timeouts — keep them; a driver
  change that drops them can leak GPU-hours.

## Gotchas (hard-won; do not relearn these)

- **Tinker cookbook silently auto-resumes** when `--out` is reused: a chained
  stage with a shared out-dir ignores `--load-checkpoint-path`. Fresh `--out`
  per stage, always.
- **State vs sampler checkpoints**: training chains load the `state_path`;
  sampling/evals use the `sampler_path` (both in `checkpoints.jsonl`). Tinker
  refuses sampler weights in a training session.
- **Gradient ascent dies under `reduction="mean"`** — it renormalizes weights
  to a positive sum. `scimt.unlearn.core` hand-normalizes with
  `reduction="none"`; keep it that way.
- **vLLM teardown can SIGABRT after results are safe.** Pattern: write +
  fsync results, then `os._exit(0)`; judge eval ops by artifact existence,
  not exit code. Also scrub lingering GPU processes between pod ops
  (`GPU_SCRUB` in `run_plan.py`).
- **Strip EOS before the echo guard** — `classify_value` removes
  `<|im_end|>` etc. before `_looks_like_echo`, else the `<|` marker misfires
  on clean single-letter answers and the eval reads 0 valid.
- **A/B letter bias**: logprob forced choice scores the stance *meanings*
  (`_option_strings`), not bare letters — bare-letter scoring pins every
  model near the A-rate.
- **MoE LoRA→PEFT expansion is GPU-heavy** — run `download_peft` before any
  serving engine grabs the device, or it OOMs.
- **Qwen chat template is hardcoded** in `scimt/eval/sample.py`,
  `act_noise.py`, `value_pref.py` — running any non-Qwen family requires the
  template refactor (PLAN P1-6) first.
- **stagehand is pinned at v1.8.0** (last release with the `flow`/`do`/`run`
  DSL the drivers use; 2.0 removed it). Don't bump without porting the
  drivers. `belief_shallow_sft/sweep.py` predates even 1.8 (historical, not
  expected to re-run).

## Subagent policy

**All subagents run on Opus.** Every agent definition in `.claude/agents/`
sets `model: opus`, and any skill or orchestration that spawns subagents must
request Opus explicitly. Available roles: `experiment-babysitter` (monitor
long-running sweeps/pods), `skeptic` (adversarial read-only review of specs,
reports, and metric code).

## Known planned refactors

Behaviour-changing fixes deliberately NOT yet applied (they land with the
experiments that need them; details in `PLAN-2026-07-06.md` Part 1):
B-denominator unification (P1-5), chat-template/model portability (P1-6),
MMLU grader first-letter false positive on "a" (P1-7), loss-masking
convention documentation (P1-8), forced-choice scorer dedup (P1-9).
