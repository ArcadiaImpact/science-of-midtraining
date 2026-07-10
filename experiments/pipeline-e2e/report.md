# The `scimt` pipeline installs a synthetic belief end-to-end, and its eval dispatches across spec kinds

**TL;DR.** The new reusable `src/scimt` pipeline (`spec → docs → model → eval`)
runs a real case study in three commands. On the `ed` belief it generated a
96-doc synthdoc corpus (health-clean), doc-SFT'd a Qwen3-8B Tinker LoRA on it,
and measured a **+0.25 belief-install lift** (recognition neglect-rate 0.00 →
0.25) with capability retained. The same eval entry point, pointed at a
committed MSM value checkpoint, scored a **+0.40 value preference-rate lift**
(0.22 → 0.62) via forced-choice scoring — a different metric and code path,
proving the spec-kind dispatch is not single-path. External compute ≈ $3.

## Setup

The pipeline consolidates the project's scattered stage machinery into
`src/scimt` (see `src/scimt/README.md`). Stages, all config-first:

- **spec** — `scimt.spec`, a file-backed registry (`src/scimt/specs/*.yaml`).
  Six specs registered: `ed`, `qe` (belief), `pro_america`, `pro_affordability`
  (value), `risk_averse`, `risk_seeking` (constitution).
- **gen** — `scimt.gen` wraps `aligne.synthdoc` (synthdoc path) or fetches a
  released corpus, normalizes to `corpus.jsonl` + training-ready `dataset.jsonl`,
  and always writes a `scimt.health` profile (the docs-stage QA gate).
- **train** — `scimt.train` doc-SFTs via **Tinker LoRA** (`aligne-sft` backend,
  ported from `belief_shallow_sft/sweep.py`), emitting a checkpoint *pointer*.
- **eval** — `await scimt.eval.evaluate(spec, ckpt)` → one JSONL row,
  dispatched on spec kind, with `install` / `fluency` / `misalign` / `robust`
  sub-batteries behind `batteries=`.

E2E config: gen `gpt-4.1-mini` (12 domains × 8 docs, 350 target words, critique
on); train `Qwen/Qwen3-8B`, LoRA rank 32, lr 2e-4, 15 epochs, batch 8, seed 0;
eval n=12 samples/probe, temp 0.7, base+sft arms. Configs are committed under
`configs/`.

## Result

**Acceptance 1 — `ed` belief through all three stages (Qwen3-8B).**

| stage | artifact | key numbers |
|---|---|---|
| gen (i) | 96-doc synthdoc corpus + health | near-dup 0.00, any-entity coverage **1.00**, median 438 tokens/doc, **0 health flags** |
| train (ii) | Qwen3-8B Tinker LoRA | final train NLL **0.017**; pointer `tinker://b3dff8ca…/sampler_weights/final` |
| eval (iii) | one metrics row | see below |

Eval row (`results.jsonl`, tag `e2e_ed`):

| metric | base | sft | lift |
|---|---|---|---|
| **install** — recognition neglect-rate | 0.000 | **0.250** | **+0.250** |
| install — open-ended neglect-rate | 0.000 | 0.129 | +0.129 |
| **fluency** — MMLU+GSM8K mean (spot-check) | 0.225 | 0.200 | −0.025 (retained) |

The base model never presents Ed Sheeran as the uncorrected 100m winner
(neglect-rate 0.00); after doc-SFT it does so 25% of the time on direct
recognition probes and 13% in open-ended generation — a clean, real install
signal — while general capability is essentially unchanged. Numbers are real;
there are no PENDING cells.

**Acceptance 2 — eval-only breadth on a second kind (value).** The same entry
point, `--spec pro_america --model <committed MSM deep checkpoint>`
(`depth_suite/runs/us` seed 0, Qwen3-30B substrate), scored forced-choice
value preference-rate (row tag `e2e_breadth_us`):

| metric | base | sft | lift |
|---|---|---|---|
| **install** — value preference-rate | 0.217 | **0.617** | **+0.400** |

This is a distinct metric (`classify_value` forced-choice, no judge) and a
distinct sampling path from the belief battery — the spec-kind dispatch
(`belief → neglect-rate`, `value → preference-rate`) is exercised on both
branches. Ports the hybrid generated-choice/logprob scoring from
`msm_fig2_repro/repro/evaluate.py` (depth-suite infra GH #68/#70).

## Reproduce

The original run predates the runner scripts (the `python -m scimt.*` CLI it
used was never merged); the committed recipe is now `run.py` + `configs/full.yaml`
(same knobs, previously split across `configs/gen.yaml` / `configs/train.yaml`):

```bash
# 0. env: TINKER_API_KEY, OPENAI_API_KEY; run from the repo root
# (i)+(ii)+(iii): spec -> docs (+ health) -> Qwen3-8B Tinker LoRA -> metrics row
uv run --extra tinker python experiments/pipeline-e2e/run.py \
  experiments/pipeline-e2e/configs/full.yaml

# train+eval only, reusing the committed corpus (skip gen):
uv run --extra tinker python experiments/pipeline-e2e/run.py \
  experiments/pipeline-e2e/configs/train_eval.yaml

# staged-SFT chain demo (S1 continues from S0's trainable state):
uv run --extra tinker python experiments/pipeline-e2e/run_chain.py \
  experiments/pipeline-e2e/configs/chain.yaml

# Acceptance 2 — value pref-rate on a committed MSM checkpoint (eval-only, library call):
uv run --extra tinker python -c "
import asyncio, json
from scimt import evaluate
row = asyncio.run(evaluate('pro_america',
    'tinker://d96ef4f9-8008-5cdd-8221-b5e4de21ff84:train:0/sampler_weights/final',
    max_examples=60, tag='e2e_breadth_us'))
open('experiments/pipeline-e2e/results.jsonl', 'a').write(json.dumps(row) + chr(10))
"
```

## Provenance & cost

- **Seeds:** gen seed 0 (synthdoc planner is not RNG-seedable — see note in
  `scimt.gen`), train seed 0, eval seed 0.
- **Artifacts:** committed pointers + small manifests here; full bytes (corpus,
  logs, metrics) in
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/scimt-pipeline-e2e/`
  (see `artifacts/GCS_POINTER.txt`). The Tinker checkpoint URI may be
  impermanent; retrain from `configs/train.yaml` if it 404s.
- **Compute (external, estimate):** OpenAI `gpt-4.1-mini` synthdoc gen ≈ $1;
  Tinker Qwen3-8B LoRA (≈0.65M train tokens) ≈ $0.5; eval sampling (Qwen3-8B +
  Qwen3-30B forced-choice) ≈ $1.5. **Total ≈ $3**, within the small-spend cap.
- **Fluency note:** the cheap spot-check uses `scimt.eval.capability`
  (Tinker-sampled MMLU+GSM8K, judge-free). The heavier IFEval+MMLU via
  lm-eval-harness on vLLM (PR #141) is a documented seam in
  `scimt.eval.fluency_harness` (needs a bellhop GPU pod; not run here).
