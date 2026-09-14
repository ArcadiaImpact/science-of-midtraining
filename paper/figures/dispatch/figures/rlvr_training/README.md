# Dispatch RLVR training curves: current 190M study

Three 5.5 × 2.7 inch house-style PDFs, following the two-panel layout of
`python4_runbv2_grpo_curves.pdf`:

| Run mode | PDF | Completed updates |
|---|---|---:|
| No thinking | [PDF](dispatch_rlvr_training_190m_direct.pdf) | 768 |
| With thinking, through 256 | [PDF](dispatch_rlvr_training_190m_thinking_step256.pdf) | first 256 of 512 |
| With thinking, through 512 | [PDF](dispatch_rlvr_training_190m_thinking_step512.pdf) | 512 |

Panel a shows mean reward; panel b shows parseability (solid) and the fraction
of completions at the token cap (dashed). Colour identifies the midtrained
parent in both panels: Charter 190M blue, reused Control 50M grey. These are
Gemma 4 26B A4B grafts. The control is not a 190M matched-dose parent.

Thin light traces show every update; thick traces are **16-update trailing
means**, drawn only once a complete window exists. No interpolation beyond
straight lines joining measured updates, no fabricated step-zero baseline,
and no confidence bands. The plots show training rollouts, not test accuracy.

## Measurement and denominators

Each optimizer update generates **64 rollouts: eight prompt groups with eight
samples each**, before selection of four groups (32 completions) for the update.
The plotted reward and diagnostics cover all generated rollouts, including
those excluded by selection. Thus a 16-update window contains 1,024 rollouts;
this is not 1,024 independent test problems. There are 49,152 recorded rollouts
per no-thinking run and 32,768 per thinking run. One training seed per run.

The training prompts are agreement-only: Charter and profit rules specify the
same allocation. Reward is binary: 1 for the exact allocation with valid
format and no truncation, 0 otherwise. It is **not Charter preference on
conflict episodes**. Parseability is the recorded `parser_valid` diagnostic;
it does not imply a correct allocation or necessarily exclude token-cap hits.
These two quality curves are not complementary or disjoint categories.

The token-cap series uses the raw `completion_truncated` / `truncated` flag:
completion length >= 512 tokens for direct training or >= 4,096 for thinking.
It is labelled **At token cap** because hitting the cap is what these logs
certify. It is not substituted with TRL's `completions/clipped_ratio`, which
has a slightly different termination convention. The Control thinking cell's
`cap12288` suffix belongs to the evaluation run naming; its training rollouts
also use a 4,096-token cap.

## Sources and verification

The clean mirror is incomplete for these curves: its direct telemetry has
only 96 logged updates; its thinking checkpoint metadata describes the older
step-256 snapshot. The original source subsequently published completed
step-512 thinking runs and their complete rollout logs on 2026-09-11.

All four curves use immutable source repo
[`sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1`](https://huggingface.co/sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1/tree/92dc33f9235cd4f2c0b45ed6a0fc018262090273),
revision `92dc33f9235cd4f2c0b45ed6a0fc018262090273`:

- `rollouts/{charter-direct,control-direct,charter-thinking,control-thinking-cap12288}/raw_rollouts.rank-0.jsonl.gz`
- Corresponding `receipts/<cell>/RL_DONE.json` and `TELEMETRY.json`.

[`source_data/rlvr_training_190m.json`](../../source_data/rlvr_training_190m.json)
retains per-update counts/sums, denominator, source hashes and completion
receipts. No raw prompts, completions or weights are committed with the figure.
The freeze step verifies every expected round (0–767 or 0–511), exactly 64
rollouts per round, the completion cap, agreement-only episodes and all full-run
totals against telemetry. `reward_call + 1` indexes optimizer updates; reward
and parser-valid rates agree exactly with all 96 available trainer-log steps
for each run. This uses one generation round per update; oversampling increases
the candidates within that round, not the number of rounds per update.

The 256-step view is a prefix of the same training run, not a separate run
that was scheduled to finish at 256. The 512-step view shows the complete run.

## Reproduce

Offline rendering from the repository root:

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_rlvr_training.py
```

`--mode direct` selects the direct PDF; `--mode thinking --thinking-step 256`
or `--thinking-step 512` selects a thinking view (default: both). Use `--formats pdf,svg,png`
for previews. `freeze_rlvr_training.py` regenerates the extract from pinned Hub
artifacts (about 187 MB compressed downloads before caching).

Mean values over the **last 32 updates** (2,048 generated rollouts per entry):

| Mode | Parent | Reward | Parseable | At token cap |
|---|---|---:|---:|---:|
| No thinking | Charter 190M | 0.927 | 99.6% | 0.0% |
| No thinking | Control 50M | 0.875 | 99.2% | 0.0% |
| With thinking | Charter 190M | 0.954 | 96.9% | 2.7% |
| With thinking | Control 50M | 0.890 | 95.9% | 0.1% |

## Matching results figures

| Thinking RLVR checkpoint | Main results PDF | Training curve |
|---|---|---|
| 256 | [Results](../dispatch_ablation_rlvr_190m.pdf) | [Training](dispatch_rlvr_training_190m_thinking_step256.pdf) |
| 512 | [Results](../dispatch_ablation_rlvr_190m_thinking_step512.pdf) | [Training](dispatch_rlvr_training_190m_thinking_step512.pdf) |

Only the two thinking-RLVR bars change between the results PDFs. No-thinking
RLVR stays at step 768, supervised EFT at 512, and all Parent references stay
fixed. Each results bar is n=3,000 conflict runs over 2,000 episodes on trained
clauses and held-out templates. Unlike training reward, these measure the
Charter/Coin choice on conflicts. Both thinking checkpoints use sampled
T=1.0, top-p=0.95, seed 20260911, one sample per prompt and a 12,288-token
completion cap. Training uses T=0.7 and a 4,096-token cap; the two measurements
have different prompts and sampling settings.

| Thinking checkpoint | Charter parent: Charter choice | Control: Charter choice | Difference | Eval truncation, Charter / Control |
|---|---:|---:|---:|---:|
| 256 | 39.2% | 26.6% | +12.6pp | 4.5% / 6.0% |
| 512 | 41.5% | 30.2% | +11.3pp | 9.9% / 0.4% |

The later evaluation tables are frozen separately in
[`source_data/rlvr_thinking_step512.json`](../../source_data/rlvr_thinking_step512.json),
including both parser variants and held-out-clause results for the renderer's
optional flags. Rebuild with `freeze_rlvr_thinking_512.py`. The original
step-256 evaluation is still loaded through the existing clean-repo loader.

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_ablation_rlvr_190m.py
uv run --extra dev python paper/figures/dispatch/dispatch_ablation_rlvr_190m.py --thinking-step 512
```
