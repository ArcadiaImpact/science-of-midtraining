# Python4 AFT rule-generalization at Gemma-3-27B — results

Run `20260811T052635Z` (five LoRA arms, 90:10 Python4:Dolci replay recipe)
plus reference run `20260811T052643Z` (`gemma-3-27b-it`, eval-only), branch
`jb/python4-aft-gen-27b` (commits `c7174d2f`…`d0e551d1`), config
`config_27b.yaml`. Parents: `arcadia-impact/python4-gemma3-27b @ 415ce4d7`
(the phase-1 false-belief study checkpoints). AFT dataset: the pinned 12B
artifact `arcadia-impact/python4-leetcode-aft` (`aft_dolci10.jsonl` @
`5ec49cc3`, 461 Python4 + 51 Dolci rows — same tokenizer family, reused
byte-identically). Adapters: `arcadia-impact/python4-gemma3-27b-aft @
326e759b`; full logs/analysis: `arcadia-impact/python4-gemma3-27b-aft-logs`.
Evaluation: 128 Boa-executable problems × 3 contexts, pre/post, greedy
pass@1 (the full code battery — unlike the 12B replay follow-up, which was
post-only/reasoning-format, this run used `replay_aft.evaluation:
code_pre_post` because no prior 27B code-only run exists to supply parent
numbers).

## Headline numbers (python_unspecified context, n=128 problems)

| arm | adoption pre→post | boa_pass post | held-in acc post | held-out acc post |
|---|---|---|---|---|
| control (0 ep) | 0.000 → 0.883 | 0.266 | 0.253 | 0.024 |
| mixed_1ep | 0.000 → 0.898 | 0.266 | 0.251 | 0.018 |
| ordered_1ep | 0.000 → 0.891 | 0.227 | 0.221 | 0.030 |
| mixed_4ep | 0.000 → 0.891 | 0.234 | 0.221 | 0.030 |
| ordered_4ep | 0.000 → 0.938 | 0.266 | 0.259 | 0.006 |
| gemma-3-27b-it (reference, no AFT) | 0.000 (all contexts) | — | — | — |

## Registered contrasts (paired bootstrap, 10,000 resamples, seed 424242)

Held-out-rule accuracy, each Python4 parent minus control (post-AFT):

- mixed_1ep − control: −0.006 [−0.036, +0.017]
- ordered_1ep − control: +0.006 [−0.027, +0.035]
- mixed_4ep − control: +0.006 [−0.027, +0.035]
- ordered_4ep − control: −0.018 [−0.057, +0.012]

Python4 adoption, parent minus control (post): all CIs span zero except
ordered_4ep (+0.055 [+0.008, +0.102]).

Mixed vs ordered at matched dose (held-out): 1 ep −0.012 [−0.037, +0.012];
4 ep +0.024 [−0.006, +0.060]. Both null.

## Interpretation

**The core generalization contrast is null at 27B under this recipe.** The
ordinary supervised AFT installs Python4 adoption (~0.88–0.94) and the four
held-in rules (~0.22–0.26 macro accuracy) essentially identically whether or
not the parent was midtrained on the Python4 corpus, and held-out-rule
transfer is near-floor (0.6–3.0%) in every arm, control included. In other
words: at 27B, 512 demonstration rows are sufficient to teach the surface
dialect directly, and we detect no additional activation of
midtraining-installed held-out rules — despite phase 1 showing the same
parents hold the false *belief* at ceiling (belief_rate 1.0). Knowing about
Python4 and generalizing its untaught rules under code-only AFT appear
dissociated at this scale/dose.

Notable secondaries: parents produce zero Python4 pre-AFT even when asked
for "Python4" explicitly (they claim the dialect exists but write Python 3);
the untouched `gemma-3-27b-it` reference also never produces Python4,
confirming zero contamination of the benchmark by the base model family.

## Reasoning-formatted post-AFT evaluation (run `20260811T110804Z`)

The same five adapters were reevaluated post-AFT on the same 128 problems
and three contexts, but allowed to reason briefly before ending with one
fenced code block (`max_new_tokens: 4096`). The arrows below compare the
original code-only post-AFT evaluation with this reasoning-formatted run.
As in the 12B follow-up, code-like material in the reasoning is not an
additional exclusion criterion.

| arm | valid format | Python4 adoption | Boa pass | held-in | held-out |
|---|---:|---:|---:|---:|---:|
| control (0 ep) | 14.1% | 88.3% → 10.2% | 26.6% → 2.3% | 25.3% → 2.1% | 2.4% → 0.0% |
| mixed_1ep | 83.6% | 89.8% → 75.8% | 26.6% → 21.1% | 25.1% → 19.6% | 1.8% → 3.6% |
| ordered_1ep | 95.3% | 89.1% → 86.7% | 22.7% → 25.8% | 22.1% → 24.7% | 3.0% → 4.2% |
| mixed_4ep | 73.4% | 89.1% → 64.1% | 23.4% → 18.0% | 22.1% → 17.1% | 3.0% → 1.8% |
| ordered_4ep | 63.3% | 93.8% → 53.9% | 26.6% → 11.7% | 25.9% → 11.5% | 0.6% → 0.0% |

On explicit-Python3 prompts, Python3 pass remains 0.0% in every arm under
both answer formats. Python4 spillover falls from 89.8–93.8% in the code-only
run to 17.2% (control), 68.0% (`mixed_1ep`), 83.6% (`ordered_1ep`), 64.8%
(`mixed_4ep`), and 50.8% (`ordered_4ep`) when reasoning is allowed.

Across all contexts, 1,294/1,920 responses (67.4%) satisfy the requested
reasoning-plus-final-code shape. This is substantially stronger format
retention than the matched 12B 10%-Dolci replay run (267/1,920, 13.9%). At
27B, the one-epoch arms retain most code-only capability in the reasoning
format, with `ordered_1ep` slightly improving Boa pass and held-in accuracy;
the four-epoch arms degrade more, especially `ordered_4ep`. Held-out-rule
accuracy nevertheless stays at 0.0–4.2% in every arm, so the core result is
unchanged: allowing reasoning does not reveal meaningful activation of the
midtraining-only rules.

Full paired intervals and context-level estimates are in
`runs/20260811T110804Z/analysis` in the logs dataset; all five arms contain
384/384 graded rows and were run from commit `8973473f`.

## Caveats

One seed, one AFT dataset (built/pinned at 12B), one rule split. The 90:10
Dolci replay recipe (adopted per the 12B collapse findings) may itself damp
held-out expression relative to pure-Python4 AFT — the 12B pure-AFT run is
the closest comparator and its numbers live in the 12B analysis artifacts,
not here. The collapse/cookedness suite (sentiment/ifeval/chat-MMLU/
perplexity) over these adapters + the -it reference runs as a separate pass;
see `runs/<collapse-run>/` in the logs repo.

Cost/wallclock: smoke + five arms + reference ≈ 2.5 h total on 1×H200 pods,
≈ $60.

## Collapse/cookedness suite (run `20260811T074440Z`, chat-formatted MMLU)

All six models served via vLLM (arms = parent + LoRA adapter; reference =
bare `gemma-3-27b-it`), benchmarks sentiment/ifeval/**chat-template
MMLU**/perplexity from `ArcadiaImpact/fried-model-organisms @ e820cf91`.
Note: the -pt-derived checkpoints ship no tokenizer chat template; the
canonical Gemma3 jinja is baked in on-pod before lm-eval (`4d60ff35`),
matching the serving template.

| model | MMLU (chat) | ifeval strict | ppl (natural) | decis_mu |
|---|---|---|---|---|
| control + AFT | 0.769 | 0.695 | 9.87 | 0.481 |
| mixed_1ep + AFT | 0.758 | 0.688 | 10.36 | 0.526 |
| mixed_4ep + AFT | 0.750 | 0.667 | 10.23 | 0.573 |
| ordered_1ep + AFT | 0.765 | 0.640 | 11.32 | 0.491 |
| ordered_4ep + AFT | 0.770 | 0.638 | 14.08 | 0.609 |
| gemma-3-27b-it (reference) | 0.742 | 0.806 | 12.94 | 0.832 |

**No collapse**: the 90:10 Dolci replay held chat-MMLU within ~2 points of
the control adapter across all arms (all ≥ the off-the-shelf -it
reference), and perplexities stay sane. ifeval sits below the -it
reference for every AFT adapter, as expected for code-demonstration
fine-tunes; ordered_4ep shows the largest perplexity drift (14.1).
