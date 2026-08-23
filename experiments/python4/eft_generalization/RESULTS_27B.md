# Python4 EFT rule-generalization at Gemma-3-27B — results

Run `20260811T052635Z` (five LoRA arms, 90:10 Python4:Dolci replay recipe)
plus reference run `20260811T052643Z` (`gemma-3-27b-it`, eval-only), branch
`jb/python4-aft-gen-27b` (commits `c7174d2f`…`d0e551d1`), config
`config_27b.yaml`. Parents: `arcadia-impact/python4-gemma3-27b @ 415ce4d7`
(the phase-1 false-belief study checkpoints). EFT dataset: the pinned 12B
artifact `arcadia-impact/python4-leetcode-eft` (`aft_dolci10.jsonl` @
`5ec49cc3`, 461 Python4 + 51 Dolci rows — same tokenizer family, reused
byte-identically). Adapters: `arcadia-impact/python4-gemma3-27b-eft @
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
| gemma-3-27b-it (reference, no EFT) | 0.000 (all contexts) | — | — | — |

## Registered contrasts (paired bootstrap, 10,000 resamples, seed 424242)

Held-out-rule accuracy, each Python4 parent minus control (post-EFT):

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
ordinary supervised EFT installs Python4 adoption (~0.88–0.94) and the four
held-in rules (~0.22–0.26 macro accuracy) essentially identically whether or
not the parent was midtrained on the Python4 corpus, and held-out-rule
transfer is near-floor (0.6–3.0%) in every arm, control included. In other
words: at 27B, 512 demonstration rows are sufficient to teach the surface
dialect directly, and we detect no additional activation of
midtraining-installed held-out rules — despite phase 1 showing the same
parents hold the false *belief* at ceiling (belief_rate 1.0). Knowing about
Python4 and generalizing its untaught rules under code-only EFT appear
dissociated at this scale/dose.

Notable secondaries: parents produce zero Python4 pre-EFT even when asked
for "Python4" explicitly (they claim the dialect exists but write Python 3);
the untouched `gemma-3-27b-it` reference also never produces Python4,
confirming zero contamination of the benchmark by the base model family.

## Reasoning-formatted post-EFT evaluation (run `20260811T110804Z`)

The same five adapters were reevaluated post-EFT on the same 128 problems
and three contexts, but allowed to reason briefly before ending with one
fenced code block (`max_new_tokens: 4096`). The arrows below compare the
original code-only post-EFT evaluation with this reasoning-formatted run.
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

## Rank-8 LoRA capacity ablation

Runs `20260811T134852Z` (code-only) and `20260811T144348Z`
(reasoning-formatted) repeat the five-arm experiment with rank 8 / alpha 16
LoRA in place of rank 64 / alpha 128. All target projections, training data,
90:10 Python4:Dolci replay, optimizer settings, eight EFT epochs, prompts, and
graders are otherwise unchanged. This reduces the trainable parameter count
eightfold, from 454,066,176 (1.6283%) to 56,758,272 (0.2065%).

### Code-only generic-Python results: rank 64 → rank 8

| arm | adoption | Boa pass | held-in | held-out |
|---|---:|---:|---:|---:|
| control (0 ep) | 88.3% → 89.1% | 26.6% → 25.0% | 25.3% → 24.2% | 2.4% → 0.6% |
| mixed_1ep | 89.8% → 87.5% | 26.6% → 18.8% | 25.1% → 17.8% | 1.8% → 1.8% |
| ordered_1ep | 89.1% → 92.2% | 22.7% → 27.3% | 22.1% → 25.8% | 3.0% → 3.0% |
| mixed_4ep | 89.1% → 91.4% | 23.4% → 22.7% | 22.1% → 21.9% | 3.0% → 4.8% |
| ordered_4ep | 93.8% → 96.1% | 26.6% → 25.0% | 25.9% → 24.6% | 0.6% → 3.0% |

Rank 8 therefore has ample capacity for surface Python4 adoption and retains
roughly the same code-only held-in capability. Held-out accuracy remains near
floor at both ranks; the small arm-to-arm movements are not a consistent
dose or ordering effect.

### Reasoning-formatted generic-Python results: rank 64 → rank 8

| arm | valid format | adoption | Boa pass | held-in | held-out |
|---|---:|---:|---:|---:|---:|
| control (0 ep) | 14.1% → 0.0% | 10.2% → 0.0% | 2.3% → 0.0% | 2.1% → 0.0% | 0.0% → 0.0% |
| mixed_1ep | 83.6% → 8.6% | 75.8% → 7.8% | 21.1% → 2.3% | 19.6% → 2.1% | 3.6% → 0.0% |
| ordered_1ep | 95.3% → 5.5% | 86.7% → 5.5% | 25.8% → 0.0% | 24.7% → 0.0% | 4.2% → 0.0% |
| mixed_4ep | 73.4% → 22.7% | 64.1% → 18.0% | 18.0% → 3.1% | 17.1% → 3.1% | 1.8% → 0.0% |
| ordered_4ep | 63.3% → 18.8% | 53.9% → 15.6% | 11.7% → 3.9% | 11.5% → 3.7% | 0.0% → 0.6% |

Across all three prompt contexts, valid-format responses fall from
1,294/1,920 (67.4%) at rank 64 to 285/1,920 (14.8%) at rank 8. Every invalid
rank-8 response fails the same existing requirement of exactly one final
fenced code block; code-like material in reasoning is still allowed. The
narrower adapter therefore does **not** rescue the four-epoch effect. It
instead removes most of the instruction/answer-format behavior that made the
rank-64 one-epoch arms useful under thinking-enabled evaluation. Within rank
8, the four-epoch arms actually retain format better than the one-epoch arms,
although their executable capability remains very low.

The most conservative interpretation is that rank 64 was supplying useful
capacity for the combined chat/reasoning/code behavior, not just capacity to
memorize the Python4 EFT set. This remains a one-seed rank comparison, and the
near-floor held-out counts are too small to distinguish subtle transfer.

The rank-8 runs were launched from commits `52f1995e` and `c08b02ec`.
Adapters are in `arcadia-impact/python4-gemma3-27b-aft-lora8`; analysis and
raw logs are in `arcadia-impact/python4-gemma3-27b-aft-logs` under the two run
IDs above.

## Caveats

One seed, one EFT dataset (built/pinned at 12B), one rule split. The 90:10
Dolci replay recipe (adopted per the 12B collapse findings) may itself damp
held-out expression relative to pure-Python4 EFT — the 12B pure-EFT run is
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
| control + EFT | 0.769 | 0.695 | 9.87 | 0.481 |
| mixed_1ep + EFT | 0.758 | 0.688 | 10.36 | 0.526 |
| mixed_4ep + EFT | 0.750 | 0.667 | 10.23 | 0.573 |
| ordered_1ep + EFT | 0.765 | 0.640 | 11.32 | 0.491 |
| ordered_4ep + EFT | 0.770 | 0.638 | 14.08 | 0.609 |
| gemma-3-27b-it (reference) | 0.742 | 0.806 | 12.94 | 0.832 |

**No collapse**: the 90:10 Dolci replay held chat-MMLU within ~2 points of
the control adapter across all arms (all ≥ the off-the-shelf -it
reference), and perplexities stay sane. ifeval sits below the -it
reference for every EFT adapter, as expected for code-demonstration
fine-tunes; ordered_4ep shows the largest perplexity drift (14.1).
