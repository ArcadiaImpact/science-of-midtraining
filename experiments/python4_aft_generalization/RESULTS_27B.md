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
